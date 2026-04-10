from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QPushButton, QSizePolicy, QApplication
from PyQt5.QtCore import Qt

class MenuScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 화면 전체를 덮는 기본 세로 방향 레이아웃을 생성
        layout = QVBoxLayout(self)
        # 화면 테두리와 내부 버튼들 사이의 여백
        layout.setContentsMargins(20, 20, 20, 20)

        # 버튼들을 바둑판(그리드) 모양으로 배치하기 위한 전용 레이아웃
        grid_layout = QGridLayout()
        # 버튼과 버튼 사이의 십자 간격
        grid_layout.setSpacing(15)

        # 일반 메뉴 버튼들의 디자인 스타일을 지정
        btn_style = """
            QPushButton { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #444, stop:1 #333); color: white; font-size: 24px; font-weight: bold; border-radius: 12px; border: 2px solid #555; }
            QPushButton:pressed { background-color: #00FFCC; color: black; border: 2px solid #fff; }
        """

        # 시스템 종료 버튼 전용 디자인 스타일
        exit_btn_style = """
            QPushButton { background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #aa0000, stop:1 #660000); color: white; font-size: 24px; font-weight: bold; border-radius: 12px; border: 2px solid #ff4444; }
            QPushButton:pressed { background-color: #ff4444; color: white; }
        """

        # 텍스트와 스타일만 넘겨주면 완성된 버튼을 만들어주는 도우미 함수
        def create_menu_btn(text, style):
            btn = QPushButton(text)
            btn.setStyleSheet(style)
            # 버튼의 크기가 고정되지 않고, 그리드 레이아웃의 빈 공간에 맞춰 상하좌우로 꽉 차게 늘어나도록 설정
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            return btn

        # 1. 실시간 녹화 버튼
        btn_live = create_menu_btn("실시간 녹화", btn_style)
        btn_live.clicked.connect(lambda: self.main_window.switch_screen(0))

        # 2. 영상 재생 버튼
        btn_playback = create_menu_btn("영상 재생", btn_style)
        btn_playback.clicked.connect(lambda: self.main_window.switch_screen(2))

        # 3. 이벤트 로그 버튼
        btn_event = create_menu_btn("이벤트 로그", btn_style)
        btn_event.clicked.connect(lambda: self.main_window.switch_screen(3))
        
        # 4. 시스템 정보 버튼
        btn_info = create_menu_btn("시스템 정보", btn_style)
        btn_info.clicked.connect(lambda: self.main_window.switch_screen(4))

        # 5. 시스템 설정 버튼
        btn_settings = create_menu_btn("시스템 설정", btn_style)
        btn_settings.clicked.connect(lambda: self.main_window.switch_screen(5))
        
        # 6. 시스템 종료 버튼
        btn_exit = create_menu_btn("시스템 종료", exit_btn_style)
        btn_exit.clicked.connect(QApplication.instance().quit)

        # 만들어진 6개의 버튼을 바둑판 레이아웃에 알맞은 위치에 배치
        # addWidget(버튼객체, 행 번호, 열 번호) 형태        
        grid_layout.addWidget(btn_live, 0, 0)     # 왼쪽 위
        grid_layout.addWidget(btn_playback, 0, 1) # 가운데 위
        grid_layout.addWidget(btn_event, 0, 2)    # 오른쪽 위
        
        grid_layout.addWidget(btn_info, 1, 0)     # 왼쪽 아래
        grid_layout.addWidget(btn_settings, 1, 1) # 가운데 아래
        grid_layout.addWidget(btn_exit, 1, 2)     # 오른쪽 아래

        # 완성된 바둑판 레이아웃을 전체 화면 레이아웃에 집어넣어 모니터 나타나게 함
        layout.addLayout(grid_layout)