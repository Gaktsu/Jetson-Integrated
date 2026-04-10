from PyQt5.QtWidgets import QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt

class InfoScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # 화면의 전체 배경색을 검은색으로 지정
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 상단 중앙에 표시될 타이틀 텍스트
        self.title_label = QLabel("시스템 정보", self)
        self.title_label.setGeometry(300, 10, 200, 30) # 가로 위치 300, 세로 위치 10에 너비 200 크기로 배치
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter) # 글자를 라벨 영역의 정중앙에 정렬

        # 화면에 띄워줄 실제 정보 텍스트 내용
        info_text = """
        장비 모델: Jetson Nano
        소프트웨어 버전: v1.0.0
        운영체제: Ubuntu 18.04
        네트워크 상태: 연결됨
        카메라: 4채널 연동 완료
        """
        
        # 위에서 작성한 텍스트를 화면에 표시할 라벨 생성
        self.info_label = QLabel(info_text, self)
        self.info_label.setGeometry(50, 80, 700, 300)
        self.info_label.setStyleSheet("color: white; font-size: 22px; line-height: 1.5; background-color: #222; border-radius: 10px; padding: 20px;")
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 좌측 하단에 배치할 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet("""
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """)
        # 뒤로가기 버튼을 누르면 메인 메뉴 화면으로 돌아감
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(1))