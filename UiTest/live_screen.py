import cv2
import datetime
import os
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# AI 위험 감지 모듈
from ai_module.ai_detector import DangerDetector

class LiveScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet("background-color: black;") # 전체 배경을 검은색으로 설정
        
        # 확대된 화면의 위치 번호를 저장
        self.expanded_pos_index = None
        # 화면 기본 위치 매핑
        self.cam_mapping = [0, 1, 2, 3]
        
        # 4개 카메라의 AI 처리 전 '원본 프레임'을 임시 보관하는 배열
        self.current_raw_frames = [None] * 4

        # AI 감지기 객체를 생성 (AI 모델 파일 경로 및 젯슨 나노 부저 핀 번호 설정)
        self.ai = DangerDetector(model_path='ai_module/best.pt', buzzer_pin=18)

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

        # 카메라 영상 장치와 동영상 저장 객체를 담을 리스트 초기화
        self.caps = []
        self.writers = []
        
        # 동영상을 저장할 'records' 폴더가 없다면 새로 생성
        if not os.path.exists("records"):
            os.makedirs("records")

        # 파일명에 쓸 현재 시간을 문자열로 포맷팅합니다.
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 동영상 저장용 코덱 설정 (XVID 포맷)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # 4대의 카메라를 순서대로 연결하고 녹화를 준비하는 반복문
        for i in range(4):
            # 젯슨 나노용 카메라 연결 포맷인 V4L2를 사용하여 카메라 캡처 시도
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2) 
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 카메라 입력 가로 해상도
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 카메라 입력 세로 해상도
                self.caps.append(cap)
                
                # 파일 저장 경로 지정 (예: records/20260403_151956_CAM1.avi)
                filename = f"records/{now_str}_CAM{i+1}.avi"
                # 초당 15프레임(15.0)으로 영상을 기록하는 VideoWriter 생성
                out = cv2.VideoWriter(filename, fourcc, 15.0, (640, 480))
                self.writers.append(out)
            else:
                # 카메라가 없거나 연결에 실패하면 빈자리(None)를 채워 넣습니다.
                self.caps.append(None)
                self.writers.append(None)

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

        # 연결된 각 카메라에서 프레임(사진 1장)을 읽음
        for i, cap in enumerate(self.caps):
            if cap is not None and cap.isOpened():
                ret, frame = cap.read() # ret: 프레임 읽기 성공 여부, frame: 사진 데이터
                if ret:
                    # 1. AI 선을 그리기 전의 순수 원본 프레임을 저장
                    self.current_raw_frames[i] = frame.copy()

                    # 2. AI 모듈에 넘겨서 위험 구역 선을 그리고, 사람이 들어왔는지 판단 결과 받음
                    frame, _ = self.ai.process_frame(frame, i)

                    # 3. AI 선과 경고 문구가 덧그려진 프레임을 동영상 파일로 저장
                    if self.writers[i] is not None:
                        self.writers[i].write(frame)

                    # 4. 화면 출력용 리스트에 최종 프레임을 등록
                    frames[i] = frame

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
            # 터치한 위치에 연결된 카메라 화면이 정상 송출 중일 때만 확대 모드 작동
            if self.caps[cam_idx] is not None:
                self.expanded_pos_index = pos

    # 프로그램 창이 완전히 닫히거나 다른 화면으로 넘어갈 때 시스템 자원을 정리하는 함수
    def closeEvent(self, event):
        self.ai.cleanup()
        for cap in self.caps:
            if cap is not None: cap.release()
        for writer in self.writers:
            if writer is not None: writer.release()