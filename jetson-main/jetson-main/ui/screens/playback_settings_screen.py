from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QScroller)
from PyQt5.QtCore import Qt

class PlaybackSettingsScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # 화면의 전체 배경색을 검은색으로 강제 지정하는 필수 옵션
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 상단 중앙에 표시될 타이틀 텍스트
        self.title_label = QLabel("영상 재생 및 관리 설정", self)
        self.title_label.setGeometry(300, 10, 200, 30)
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        # 설정 항목이 많아질 경우를 대비해 스크롤이 가능한 영역을 만듬
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(0, 50, 800, 350)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # 터치 모니터나 마우스 드래그를 이용해 스마트폰처럼 화면을 넘길 수 있게 해주는 기능
        QScroller.grabGesture(self.scroll_area.viewport(), QScroller.LeftMouseButtonGesture)

        # 스크롤 영역 내부에 들어갈 실제 설정 항목들의 뼈대(리스트 위젯)를 만듬
        self.list_widget = QWidget()
        self.list_widget.setStyleSheet("background-color: transparent;")
        self.list_layout = QVBoxLayout(self.list_widget) # 항목들을 세로로 일렬 배치
        self.list_layout.setAlignment(Qt.AlignTop)       # 위에서부터 차곡차곡 쌓이도록 정렬
        self.list_layout.setSpacing(10)                  # 각 항목 사이의 간격을 10픽셀로 설정
        self.scroll_area.setWidget(self.list_widget)

        # 좌측 하단에 배치할 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet("""
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """)
        # 뒤로가기 버튼을 누르면 영상 재생 목록 화면)으로 돌아감
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(2))

        # 리스트에 들어갈 개별 설정 항목들을 추가하는 부분입니다.
        self.add_toggle_row("저장 공간 부족 시 오래된 영상 자동 삭제", True)
        self.add_toggle_row("USB 백업 시 파일 자동 압축", False)
        
        self.add_button_row("기본 재생 배속 설정", "1.0 배속")
        self.add_button_row("영상 정렬 기준 변경", "최신순")
        self.add_button_row("전체 녹화 영상 삭제 (포맷)", "삭제하기")

    # UI 줄(Row)을 생성하는 도우미 함수
    # 제목과 ON/OFF 토글 버튼이 있는 가로 한 줄을 생성
    def add_toggle_row(self, title_text, initial_state):
        row_frame = QFrame()
        row_frame.setStyleSheet("background-color: #222; border-radius: 8px; border: 1px solid #444;")
        row_frame.setFixedHeight(70)
        
        layout = QHBoxLayout(row_frame) # 가로 정렬
        layout.setContentsMargins(20, 10, 20, 10)

        # 왼쪽: 설정 항목의 이름
        title = QLabel(title_text)
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; border: none;")

        # 오른쪽: ON/OFF 상태를 보여주는 버튼
        toggle_btn = QPushButton("ON" if initial_state else "OFF")
        toggle_btn.setFixedSize(100, 40)
        self.update_toggle_style(toggle_btn, initial_state)
        
        # 버튼을 클릭하면 텍스트와 색상이 바뀌도록 연결
        toggle_btn.clicked.connect(lambda: self.toggle_state(toggle_btn))

        layout.addWidget(title, 1)
        layout.addWidget(toggle_btn)
        self.list_layout.addWidget(row_frame)

    # 제목과 일반 실행 버튼이 있는 가로 한 줄을 생성
    def add_button_row(self, title_text, btn_text, callback=None):
        row_frame = QFrame()
        row_frame.setStyleSheet("background-color: #222; border-radius: 8px; border: 1px solid #444;")
        row_frame.setFixedHeight(70)
        
        layout = QHBoxLayout(row_frame) # 가로 정렬
        layout.setContentsMargins(20, 10, 20, 10)

        # 왼쪽: 설정 항목의 이름
        title = QLabel(title_text)
        title.setStyleSheet("color: white; font-size: 18px; font-weight: bold; border: none;")

        # 오른쪽: 작업을 실행할 버튼
        action_btn = QPushButton(btn_text)
        action_btn.setFixedSize(120, 40)
        action_btn.setStyleSheet("""
            QPushButton { background-color: #555; color: white; font-size: 16px; font-weight: bold; border-radius: 5px; border: none; }
            QPushButton:pressed { background-color: #777; }
        """)
        
        # 이 버튼을 눌렀을 때 실행할 함수가 전달되었다면 연결
        if callback:
            action_btn.clicked.connect(callback)

        layout.addWidget(title, 1)
        layout.addWidget(action_btn)
        self.list_layout.addWidget(row_frame)

    # 토글 버튼 클릭 시 ON을 OFF로, OFF를 ON으로 뒤집는 함수
    def toggle_state(self, btn):
        is_on = btn.text() == "ON"
        new_state = not is_on
        btn.setText("ON" if new_state else "OFF")
        self.update_toggle_style(btn, new_state)

    # ON/OFF 상태에 맞춰 버튼의 색깔을 바꿔주는 함수
    def update_toggle_style(self, btn, is_on):
        if is_on:
            btn.setStyleSheet("QPushButton { background-color: #0055aa; color: white; font-size: 18px; font-weight: bold; border-radius: 20px; border: 2px solid #0077ff; }")
        else:
            btn.setStyleSheet("QPushButton { background-color: #444; color: gray; font-size: 18px; font-weight: bold; border-radius: 20px; border: 2px solid #555; }")