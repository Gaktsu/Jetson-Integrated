from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt

class SettingsScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 상단 중앙에 표시될 타이틀 텍스트 라벨
        self.title_label = QLabel("시스템 설정", self)
        self.title_label.setGeometry(300, 10, 200, 30)
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.settings_widget = QWidget(self)
        self.settings_widget.setGeometry(100, 80, 600, 300)
        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.settings_layout.setSpacing(20) # 버튼과 버튼 사이의 간격

        # 설정 항목 버튼들의 공통 디자인 스타일
        setting_btn_style = """
            QPushButton { background-color: #333; color: white; font-size: 20px; font-weight: bold; border-radius: 8px; padding: 15px; border: 1px solid #555; }
            QPushButton:pressed { background-color: #555; }
        """

        # 1. 저장소 포맷 (모든 영상 삭제) 버튼 생성 및 스타일 적용 / 해당 기능 미구현
        self.btn_format = QPushButton("저장소 포맷 (모든 영상 삭제)")
        self.btn_format.setStyleSheet(setting_btn_style)
        
        # 2. 시스템 재부팅 버튼 생성 및 스타일 적용 / 해당 기능 미구현
        self.btn_reboot = QPushButton("시스템 재부팅")
        self.btn_reboot.setStyleSheet(setting_btn_style)

        # 만들어진 두 개의 버튼을 세로 정렬 레이아웃에 차례대로 넣음
        self.settings_layout.addWidget(self.btn_format)
        self.settings_layout.addWidget(self.btn_reboot)
        
        # 버튼들을 위쪽으로 밀어 올리고, 남는 아래쪽 공간을 빈 공간(Stretch)으로 채워줌
        self.settings_layout.addStretch()

        # 좌측 하단에 배치할 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet("""
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """)
        # 뒤로가기 버튼을 누르면 메인 메뉴 화면으로 돌아감
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(1))