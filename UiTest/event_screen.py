from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt

class EventScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # 화면의 전체 배경색을 검은색으로 지정
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 이벤트 로그 항목이 많아질 경우를 대비해 스크롤이 가능한 영역을 만듬
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(0, 50, 800, 350)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        # 스크롤 영역 내부에 들어갈 실제 이벤트 항목들의 뼈대를 만듬
        self.list_widget = QWidget()
        self.list_widget.setStyleSheet("background-color: transparent;")
        
        # 항목들을 세로로 일렬 배치하기 위한 레이아웃 설정
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setAlignment(Qt.AlignTop) # 위에서부터 차곡차곡 쌓이도록 정렬
        self.list_layout.setSpacing(10)            # 각 이벤트 항목 사이의 간격을 10픽셀로 설정
        
        # 완성된 리스트 레이아웃을 스크롤 영역 안에 넣음
        self.scroll_area.setWidget(self.list_widget)

        # 상단 중앙에 표시될 타이틀 텍스트
        self.title_label = QLabel("이벤트 로그", self)
        self.title_label.setGeometry(300, 10, 200, 30)
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)

        # 좌측 하단에 배치할 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet("""
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """)
        # 뒤로가기 버튼을 누르면 메인 메뉴 화면으로 돌아감
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(1))

        # 화면이 처음 만들어질 때, 기록된 이벤트 내역을 불러와서 화면에 채워 넣음
        self.load_events()

    # 기록된 이벤트(위험 감지 로그 등)를 불러와서 화면에 하나씩 그려주는 함수
    def load_events(self):
        # 현재는 화면이 어떻게 보이는지 테스트하기 위해 임시 데이터를 넣었음
        events = [
            "2026-04-03 15:30:22 - 충격 감지 (CAM 1)", 
            "2026-04-03 14:15:00 - 접근 경고 (CAM 3)", 
            "2026-04-03 08:00:15 - 시스템 시작"
        ]
        
        # 리스트에 있는 이벤트를 하나씩 꺼내서 텍스트 상자(QLabel)로 만듬
        for evt in events:
            lbl = QLabel(evt)
            lbl.setStyleSheet("color: white; font-size: 18px; padding: 15px; background-color: #222; border-radius: 8px; border: 1px solid #444;")
            # 레이아웃에 추가
            self.list_layout.addWidget(lbl)