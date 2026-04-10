import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

# 프로그램에서 사용할 각 화면 외부 파이썬 파일
from live_screen import LiveScreen                     # 0번: 실시간 녹화 화면
from menu_screen import MenuScreen                     # 1번: 메인 메뉴 화면
from playback_screen import PlaybackScreen             # 2번: 영상 재생 목록 화면
from event_screen import EventScreen                   # 3번: 이벤트(위험 감지) 로그 화면
from info_screen import InfoScreen                     # 4번: 시스템 정보 화면
from settings_screen import SettingsScreen             # 5번: 전체 시스템 설정 화면
from live_settings_screen import LiveSettingsScreen    # 6번: 실시간 녹화 전용 설정 화면
from playback_settings_screen import PlaybackSettingsScreen # 7번: 영상 재생 전용 설정 화면
from roi_setup_screen import RoiSetupScreen            # 8번: 위험구역(ROI) 선 그리기 화면

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 프로그램 창의 기본 설정
        self.setWindowTitle("TEST UI") # 제목 설정
        self.setGeometry(100, 100, 800, 480) # 창이 열리는 위치(x, y)와 크기(가로, 세로) 설정

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 불러온 각 화면 클래스들을 실제 객체로 생성
        self.live_screen = LiveScreen(self)
        self.menu_screen = MenuScreen(self)
        self.playback_screen = PlaybackScreen(self)
        self.event_screen = EventScreen(self)
        self.info_screen = InfoScreen(self)
        self.settings_screen = SettingsScreen(self)
        
        self.live_settings_screen = LiveSettingsScreen(self)
        self.playback_settings_screen = PlaybackSettingsScreen(self)
        
        self.roi_setup_screen = RoiSetupScreen(self)

        # 생성된 화면 객체들을 스택 위젯에 순서대로 차곡차곡 넣습니다.
        # 이 순서대로 0번부터 고유한 인덱스 번호를 부여받습니다.
        self.stacked_widget.addWidget(self.live_screen)             # 인덱스 0
        self.stacked_widget.addWidget(self.menu_screen)             # 인덱스 1
        self.stacked_widget.addWidget(self.playback_screen)         # 인덱스 2
        self.stacked_widget.addWidget(self.event_screen)            # 인덱스 3
        self.stacked_widget.addWidget(self.info_screen)             # 인덱스 4
        self.stacked_widget.addWidget(self.settings_screen)         # 인덱스 5
        self.stacked_widget.addWidget(self.live_settings_screen)    # 인덱스 6
        self.stacked_widget.addWidget(self.playback_settings_screen)# 인덱스 7
        self.stacked_widget.addWidget(self.roi_setup_screen)        # 인덱스 8

        # 프로그램이 켜지면 가장 먼저 보여줄 화면 지정 (0번: 실시간 녹화 화면)
        self.switch_screen(0)

    # 지정한 인덱스 번호의 화면으로 전환하는 함수
    def switch_screen(self, index):
        # 만약 전환하려는 화면이 영상 재생 목록 화면이라면
        if index == 2:
            # 화면이 넘어가기 직전 폴더를 검사해서 최신 영상 목록을 다시 불러오기
            self.playback_screen.load_files()
            
        # 스택 위젯에서 해당 인덱스의 화면을 최상단으로 옮김
        self.stacked_widget.setCurrentIndex(index)

    # 프로그램이 종료될 때 실행되는 정리 함수
    def closeEvent(self, event):
        # 1. 켜져 있는 카메라 장치들을 안전하게 종료
        for cap in self.live_screen.caps:
            if cap is not None and cap.isOpened(): 
                cap.release()

        # 2. 녹화 중이던 비디오 파일들을 안전하게 저장하고 닫기
        for writer in self.live_screen.writers:
            if writer is not None: 
                writer.release()

        # 3. AI 모듈과 연결된 하드웨어를 초기화하고 정리
        self.live_screen.ai.cleanup()

# 이 파이썬 파일이 직접 실행될 때만 아래의 코드를 작동
if __name__ == "__main__":
    app = QApplication(sys.argv) # PyQt5 프로그램 실행 환경 준비
    window = MainApp()           # 위에서 정의한 메인 프로그램 창 객체 생성
    window.show()                # 생성된 창을 모니터 화면에 띄움
    sys.exit(app.exec_())        # 프로그램이 무한 루프를 돌며 이벤트(클릭 등)를 기다리도록 실행