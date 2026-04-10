import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QScroller, QDialog, QMessageBox)
from PyQt5.QtCore import Qt

# [1] 카메라 채널 순서 변경 팝업창 클래스
class CamOrderPopup(QDialog):
    def __init__(self, current_mapping, parent=None):
        super().__init__(parent)
        # 기본 윈도우 타이틀바를 없애고(Frameless) 팝업창(Dialog) 속성 부여
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #555; border-radius: 15px;")
        self.setFixedSize(500, 380)
        
        # 메인 화면에서 현재 사용 중인 카메라 배치 순서를 복사
        self.cam_mapping = list(current_mapping)

        # 전체 세로 정렬 레이아웃 설정
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 25, 25, 25)

        # 상단 타이틀
        title_label = QLabel("카메라 채널 순서 변경", self)
        title_label.setStyleSheet("color: white; font-size: 22px; font-weight: bold; border: none;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addSpacing(15)

        # 4개의 화면 위치 이름
        positions = ["채널 1 (좌측 상단)", "채널 2 (우측 상단)", "채널 3 (좌측 하단)", "채널 4 (우측 하단)"]
        
        # 각 위치별로 어떤 카메라를 띄울지 선택하는 버튼 4개를 생성
        for i in range(4):
            row_layout = QHBoxLayout() # 가로 정렬 레이아웃
            
            pos_label = QLabel(positions[i])
            pos_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold; border: none;")
            
            # 현재 매핑된 카메라 번호를 버튼에 표시
            cam_btn = QPushButton(f"카메라 {self.cam_mapping[i] + 1}")
            cam_btn.setFixedSize(110, 45)
            cam_btn.setStyleSheet("""
                QPushButton { background-color: #0055aa; color: white; font-size: 18px; font-weight: bold; border-radius: 8px; border: none; }
                QPushButton:pressed { background-color: #0077ff; }
            """)
            # 버튼을 누를 때마다 숫자가 1~4로 바뀌도록 함수 연결
            cam_btn.clicked.connect(lambda checked, idx=i, btn=cam_btn: self.cycle_cam(idx, btn))

            row_layout.addWidget(pos_label)
            row_layout.addStretch() # 가운데 빈 공간
            row_layout.addWidget(cam_btn)
            layout.addLayout(row_layout)

        layout.addStretch()

        # 하단 취소 및 적용 버튼
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(15)
        
        cancel_btn = QPushButton("취소")
        cancel_btn.setFixedHeight(50)
        cancel_btn.setStyleSheet("QPushButton { background-color: #555; color: white; font-size: 20px; font-weight: bold; border-radius: 8px; border: none; }")
        cancel_btn.clicked.connect(self.reject)

        apply_btn = QPushButton("적용")
        apply_btn.setFixedHeight(50)
        apply_btn.setStyleSheet("QPushButton { background-color: #28a745; color: white; font-size: 20px; font-weight: bold; border-radius: 8px; border: none; }")
        apply_btn.clicked.connect(self.accept)

        bottom_layout.addWidget(cancel_btn)
        bottom_layout.addWidget(apply_btn)
        layout.addLayout(bottom_layout)

    # 버튼을 누를 때마다 카메라 번호가 순서로 바뀌는 함수
    def cycle_cam(self, idx, btn):
        self.cam_mapping[idx] = (self.cam_mapping[idx] + 1) % 4
        btn.setText(f"카메라 {self.cam_mapping[idx] + 1}")



# [2] 실시간 설정 메인 목록 창 클래스
class LiveSettingsScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 상단 타이틀
        self.title_label = QLabel("실시간 녹화 설정", self)
        self.title_label.setGeometry(300, 10, 200, 30)
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        # 터치 스크롤이 가능한 영역 생성
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(0, 50, 800, 350)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # 마우스나 터치 드래그로 화면을 위아래로 넘길 수 있게 해주는 PyQt5 내장 기능
        QScroller.grabGesture(self.scroll_area.viewport(), QScroller.LeftMouseButtonGesture)

        # 스크롤 영역 안에 들어갈 실제 리스트 내용물 틀
        self.list_widget = QWidget()
        self.list_widget.setStyleSheet("background-color: transparent;")
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setAlignment(Qt.AlignTop)
        self.list_layout.setSpacing(10)
        self.scroll_area.setWidget(self.list_widget)

        # 좌측 하단 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet("""
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """)
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(0)) # 0번(실시간 화면)으로 이동

        # 리스트에 들어갈 개별 설정 항목들을 추가하는 부분
        # 실시간 화면의 AI 감지기에서 현재 ON/OFF 상태를 가져옴
        initial_detection_state = self.main_window.live_screen.ai.detection_enabled
        self.add_toggle_row("AI 위험구역 감지 기능", initial_detection_state, self.toggle_detection)
        
        # 카메라 1~4 구역 설정 화면으로 넘어가는 버튼들
        self.add_button_row("CAM 1 위험구역 설정", "설정하기", lambda: self.open_roi_setup(0))
        self.add_button_row("CAM 2 위험구역 설정", "설정하기", lambda: self.open_roi_setup(1))
        self.add_button_row("CAM 3 위험구역 설정", "설정하기", lambda: self.open_roi_setup(2))
        self.add_button_row("CAM 4 위험구역 설정", "설정하기", lambda: self.open_roi_setup(3))
        
        self.add_toggle_row("위험 경고음 (부저) 소리", True)
        self.add_toggle_row("서버 클라우드 자동 백업", False)
        
        self.add_button_row("카메라 채널(1~4번) 순서 변경", "설정하기", self.show_cam_popup)
        
        self.add_button_row("녹화 화질 및 프레임 설정", "설정하기")
        self.add_button_row("녹화 파일 자동 분할 시간", "5분 (변경)")
        self.add_button_row("카메라 밝기 / 명암 조절", "설정하기")
        self.add_button_row("시스템 초기화", "초기화")


    # 버튼 클릭 시 실행될 기능(함수)들
    # AI 모듈의 사람 감지 기능을 전체적으로 켜고 끔
    def toggle_detection(self, enabled):
        self.main_window.live_screen.ai.set_detection_enabled(enabled)

    # 위에서 정의한 카메라 순서 변경 팝업창을 화면 중앙에 출력
    def show_cam_popup(self):
        current_mapping = self.main_window.live_screen.cam_mapping
        popup = CamOrderPopup(current_mapping, self)
        
        if popup.exec_() == QDialog.Accepted:
            self.main_window.live_screen.cam_mapping = list(popup.cam_mapping)

    # 위험구역 선 그리기 화면(8번)으로 이동하는 함수
    def open_roi_setup(self, cam_idx):
        # 실시간 화면에서 해당 카메라의 가장 최근 원본 사진을 가져옴
        raw_frame = self.main_window.live_screen.current_raw_frames[cam_idx]
        
        # 만약 카메라가 꺼져있어서 사진이 없다면, 검은색 가짜 화면을 만듬
        if raw_frame is None:
            raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(raw_frame, f"CAM {cam_idx + 1}", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(raw_frame, "NO SIGNAL (TEST MODE)", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 구역 설정 화면으로 사진과 카메라 번호를 넘겨주고 화면을 전환
        self.main_window.roi_setup_screen.set_base_frame(raw_frame.copy(), cam_idx)
        self.main_window.switch_screen(8)


    # UI 줄(Row)을 생성하는 도우미 함수들
    # ON/OFF 스위치가 있는 가로 한 줄을 만들어서 리스트에 추가함
    def add_toggle_row(self, title_text, initial_state, callback=None):
        row_frame = QFrame()
        row_frame.setStyleSheet("background-color: #222; border-radius: 8px; border: 1px solid #444;")
        row_frame.setFixedHeight(70)
        layout = QHBoxLayout(row_frame)
        layout.setContentsMargins(20, 10, 20, 10)

        title = QLabel(title_text)
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; border: none;")

        toggle_btn = QPushButton("ON" if initial_state else "OFF")
        toggle_btn.setFixedSize(100, 40)
        self.update_toggle_style(toggle_btn, initial_state)
        
        # 버튼을 누를 때마다 ON과 OFF가 뒤집히는 로직
        def on_clicked():
            is_on = toggle_btn.text() == "ON"
            new_state = not is_on
            toggle_btn.setText("ON" if new_state else "OFF")
            self.update_toggle_style(toggle_btn, new_state)
            if callback:
                callback(new_state)

        toggle_btn.clicked.connect(on_clicked)

        layout.addWidget(title, 1)
        layout.addWidget(toggle_btn)
        self.list_layout.addWidget(row_frame)

    # 일반 버튼(설정하기 등)이 있는 가로 한 줄을 만들어서 리스트에 추가함
    def add_button_row(self, title_text, btn_text, callback=None):
        row_frame = QFrame()
        row_frame.setStyleSheet("background-color: #222; border-radius: 8px; border: 1px solid #444;")
        row_frame.setFixedHeight(70)
        layout = QHBoxLayout(row_frame)
        layout.setContentsMargins(20, 10, 20, 10)

        title = QLabel(title_text)
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; border: none;")

        action_btn = QPushButton(btn_text)
        action_btn.setFixedSize(120, 40)
        action_btn.setStyleSheet("""
            QPushButton { background-color: #555; color: white; font-size: 16px; font-weight: bold; border-radius: 5px; border: none; }
            QPushButton:pressed { background-color: #777; }
        """)
        
        if callback:
            action_btn.clicked.connect(callback)

        layout.addWidget(title, 1)
        layout.addWidget(action_btn)
        self.list_layout.addWidget(row_frame)

    # ON/OFF 상태에 따라 버튼의 색상과 모양을 변경함
    def update_toggle_style(self, btn, is_on):
        if is_on:
            btn.setStyleSheet("QPushButton { background-color: #0055aa; color: white; font-size: 18px; font-weight: bold; border-radius: 20px; border: 2px solid #0077ff; }")
        else:
            btn.setStyleSheet("QPushButton { background-color: #444; color: gray; font-size: 18px; font-weight: bold; border-radius: 20px; border: 2px solid #555; }")