import cv2
import datetime
import json
import os
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from config.settings import CAMERA_INDICES, PROJECT_ROOT
from ui.renderer import draw_detections
from ai.detector import load_roi_polygon


class _PipelineAiProxy:
    """
    pipeline 모드에서 live_settings_screen / roi_setup_screen의
    'live_screen.ai' 참조를 처리하는 어댑터.
    - detection on/off: 플래그 보관 (추후 inference 스레드와 연결 가능)
    - ROI get/set: config/roi_config_cam{cam_id}.json 파일 기반
    """
    _DEFAULT_ROI = [[220, 340], [420, 340], [420, 140], [220, 140]]

    def __init__(self):
        self.detection_enabled = True

    def set_detection_enabled(self, enabled: bool) -> None:
        self.detection_enabled = enabled

    def get_roi_points(self, cam_idx: int) -> list:
        """JSON 파일에서 ROI 좌표 로드. 없으면 기본값 반환."""
        cam_id = CAMERA_INDICES[cam_idx] if cam_idx < len(CAMERA_INDICES) else cam_idx
        path = os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_id}.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) >= 3:
                return data
        except Exception:
            pass
        return [list(p) for p in self._DEFAULT_ROI]

    def set_roi_points(self, cam_idx: int, points: list) -> None:
        """ROI 좌표를 JSON 파일에 저장."""
        cam_id = CAMERA_INDICES[cam_idx] if cam_idx < len(CAMERA_INDICES) else cam_idx
        path = os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_id}.json")
        try:
            with open(path, "w") as f:
                json.dump([[int(p[0]), int(p[1])] for p in points], f)
        except Exception as e:
            print(f"ROI 저장 실패 (cam_idx={cam_idx}): {e}")

    def cleanup(self) -> None:
        pass

class LiveScreen(QWidget):
    def __init__(self, main_window, shared_states=None):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet("background-color: black;") # 전체 배경을 검은색으로 설정
        
        # 확대된 화면의 위치 번호를 저장
        self.expanded_pos_index = None
        # 화면 기본 위치 매핑
        self.cam_mapping = [0, 1, 2, 3]
        
        # 4개 카메라의 최신 프레임을 임시 보관 (ROI 설정 화면 등에서 참조)
        self.current_raw_frames = [None] * 4

        # pipeline의 SharedState 리스트 — 카메라/AI/녹화는 모두 pipeline이 담당
        self.shared_states = shared_states

        # ai: live_settings_screen / roi_setup_screen에서 참조되는 접점
        # pipeline 모드에서는 JSON 기반 프록시를 사용
        self.ai = _PipelineAiProxy()

        # 4개의 카메라 영상을 보여줄 UI 라벨(영역)을 생성
        self.cam_labels = []
        positions = [(0, 0), (400, 0), (0, 240), (400, 240)] # 4분할 화면의 각 (x, y) 시작 좌표
        for i in range(4):
            lbl = QLabel(f"CAM {i+1}\nNO SIGNAL", self)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color: #111; color: red; font-size: 20px; font-weight: bold; border: 1px solid #333;")
            lbl.setGeometry(positions[i][0], positions[i][1], 400, 240) # 가로 400, 세로 240 크기로 배치
            self.cam_labels.append(lbl)

        # 화면 중 하나를 터치했을 때 꽉 찬 전체 화면으로 띄워줄 라벨
        self.full_screen_label = QLabel(self)
        self.full_screen_label.setGeometry(0, 0, 800, 480)
        self.full_screen_label.setStyleSheet("background-color: black;")
        self.full_screen_label.setAlignment(Qt.AlignCenter)
        self.full_screen_label.hide()

        # 화면 위에 상태창 텍스트들 스타일 (반투명 검은 배경에 흰 글씨)
        overlay_style = "background-color: rgba(0, 0, 0, 150); color: white; font-weight: bold; border-radius: 5px; padding: 5px;"

        # 좌측 상단: 현재 시간을 표시할 라벨
        self.time_label = QLabel("Loading...", self)
        self.time_label.setGeometry(10, 10, 220, 30)
        self.time_label.setStyleSheet(overlay_style + "font-size: 14px;")
        self.time_label.setAlignment(Qt.AlignCenter)

        # 우측 상단: 시스템 작동 상태(녹화, 서버, AI)를 표시할 라벨
        self.status_label = QLabel("REC | 서버 ON | AI ON", self)
        self.status_label.setGeometry(590, 10, 200, 30)
        self.status_label.setStyleSheet(overlay_style + "font-size: 14px;")
        self.status_label.setAlignment(Qt.AlignCenter)

        # 버튼들의 공통 스타일 지정
        btn_style = """
            QPushButton { background-color: rgba(0, 0, 0, 150); 
            color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """

        # 좌측 하단: 메인 메뉴(1번 화면)로 돌아가는 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(1))

        # 우측 하단: 실시간 영상 설정창(6번 화면)으로 넘어가는 설정(톱니바퀴) 버튼
        self.set_btn = QPushButton("⚙", self)
        self.set_btn.setGeometry(730, 410, 60, 60)
        self.set_btn.setStyleSheet(btn_style)
        self.set_btn.clicked.connect(lambda: self.main_window.switch_screen(6))

        # 화면에 그려지는 순서정리
        self.full_screen_label.lower()
        self.back_btn.raise_()
        self.set_btn.raise_()
        self.time_label.raise_()
        self.status_label.raise_()

        # 카메라/녹화는 pipeline(capture.py, recorder.py)이 전담
        # live_screen은 SharedState에서 프레임만 읽음

        # 정해진 시간마다 함수를 반복 실행해주는 타이머 도구
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames) # 시간이 될 때마다 update_frames 함수 실행
        self.timer.start(50) # 50밀리초마다 갱신

    # 타이머에 의해 반복 실행되며, 카메라 영상을 읽고 AI 처리를 한 뒤 모니터에 그리는 핵심 함수
    def update_frames(self):
        # 화면 좌측 상단의 시계 텍스트를 실시간으로 업데이트
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(now)

        # 이번 턴에 모니터에 출력할 4개의 화면 데이터를 담을 빈 리스트
        frames = [None] * 4

        # SharedState에서 최신 프레임 + 탐지 결과를 읽어 오버레이 적용
        if self.shared_states is not None:
            for i, state in enumerate(self.shared_states):
                if i >= 4:
                    break
                with state.frame_lock:
                    if state.latest_frame is None:
                        continue
                    raw = state.latest_frame.copy()

                self.current_raw_frames[i] = raw

                with state.det_lock:
                    detections     = list(state.last_detections)
                    intrusion      = state.last_intrusion
                    warning_level  = state.last_warning_level

                cam_id = CAMERA_INDICES[i] if i < len(CAMERA_INDICES) else i
                roi_path = os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_id}.json")
                roi_polygon = load_roi_polygon(roi_path)

                frames[i] = draw_detections(
                    raw,
                    detections,
                    roi_polygon=roi_polygon,
                    intrusion=intrusion,
                    warning_level=warning_level,
                    camera_index=cam_id,
                    forklift_speed=state.forklift_speed,
                )

        # 사용자가 화면 하나를 터치해서 확대모드일 경우
        if self.expanded_pos_index is not None:
            pos = self.expanded_pos_index
            cam_idx = self.cam_mapping[pos] # 현재 터치한 위치에 할당된 실제 카메라 번호를 확인
            frame = frames[cam_idx]
            
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 800x480 전체 화면 크기에 꽉 차도록 이미지 크기 조절
                frame_resized = cv2.resize(frame_rgb, (800, 480))
                h, w, c = frame_resized.shape
                q_img = QImage(frame_resized.data, w, h, 3 * w, QImage.Format_RGB888)
                self.full_screen_label.setPixmap(QPixmap.fromImage(q_img))
                self.full_screen_label.show() # 전체 화면 표시
                
        # 확대된 화면이 없는 경우
        else:
            self.full_screen_label.hide() # 전체 화면 라벨 숨김
            for pos in range(4):
                cam_idx = self.cam_mapping[pos]
                frame = frames[cam_idx]
                
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 400x240 분할 화면 크기에 맞게 이미지 축소
                    frame_resized = cv2.resize(frame_rgb, (400, 240))
                    h, w, c = frame_resized.shape
                    q_img = QImage(frame_resized.data, w, h, 3 * w, QImage.Format_RGB888)
                    self.cam_labels[pos].setPixmap(QPixmap.fromImage(q_img))
                else:
                    # 연결된 카메라가 없을 때는 기존 화면을 지우고 NO SIGNAL 텍스트를 출력
                    self.cam_labels[pos].clear()
                    self.cam_labels[pos].setText(f"CAM {cam_idx+1}\nNO SIGNAL")
    
    # 사용자가 모니터 화면을 터치(또는 마우스 클릭)했을 때 실행되는 이벤트 함수
    def mousePressEvent(self, event):
        x, y = event.x(), event.y()
        
        # 터치한 좌표가 좌/우측 하단의 버튼 위치라면 확대 기능을 작동하지 않고 넘김
        if (x < 80 and y > 400) or (x > 720 and y > 400):
            return super().mousePressEvent(event)

        # 이미 전체 화면 상태일 때 아무 곳이나 터치하면 4분할 화면으로 돌아옴
        if self.expanded_pos_index is not None:
            self.expanded_pos_index = None
            self.full_screen_label.hide()
        else:
            # 4분할 화면일 때, 화면을 4등분하여 어느 위치를 터치했는지 판별
            if x < 400 and y < 240: pos = 0     # 좌상단 영역
            elif x >= 400 and y < 240: pos = 1  # 우상단 영역
            elif x < 400 and y >= 240: pos = 2  # 좌하단 영역
            else: pos = 3                       # 우하단 영역
            
            cam_idx = self.cam_mapping[pos]
            # 최신 프레임이 있을 때만 확대 모드 작동
            if self.current_raw_frames[cam_idx] is not None:
                self.expanded_pos_index = pos

    # 프로그램 창이 닫힐 때 — 카메라/녹화/AI 정리는 pipeline(main.py)이 담당
    def closeEvent(self, event):
        pass