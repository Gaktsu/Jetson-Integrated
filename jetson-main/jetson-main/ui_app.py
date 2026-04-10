"""
PyQt5 UI 진입점
- main.py에서 shared_states, buzzer를 주입받아 pipeline과 연결
- 단독 실행 시(python3 ui_app.py)에는 shared_states=None으로 동작
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from PyQt5.QtCore import QTimer

from ui.screens.live_screen import LiveScreen
from ui.screens.menu_screen import MenuScreen
from ui.screens.playback_screen import PlaybackScreen
from ui.screens.event_screen import EventScreen
from ui.screens.info_screen import InfoScreen
from ui.screens.settings_screen import SettingsScreen
from ui.screens.live_settings_screen import LiveSettingsScreen
from ui.screens.playback_settings_screen import PlaybackSettingsScreen
from ui.screens.roi_setup_screen import RoiSetupScreen


class MainApp(QMainWindow):
    def __init__(self, shared_states=None, buzzer=None):
        super().__init__()

        self.setWindowTitle("실시간 안전 모니터링")
        self.setGeometry(100, 100, 800, 480)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # shared_states를 LiveScreen에 주입 — 카메라/AI/녹화는 pipeline이 담당
        self.live_screen = LiveScreen(self, shared_states=shared_states)
        self.menu_screen = MenuScreen(self)
        self.playback_screen = PlaybackScreen(self)
        self.event_screen = EventScreen(self)
        self.info_screen = InfoScreen(self)
        self.settings_screen = SettingsScreen(self)
        self.live_settings_screen = LiveSettingsScreen(self)
        self.playback_settings_screen = PlaybackSettingsScreen(self)
        self.roi_setup_screen = RoiSetupScreen(self)

        self.stacked_widget.addWidget(self.live_screen)              # 0
        self.stacked_widget.addWidget(self.menu_screen)              # 1
        self.stacked_widget.addWidget(self.playback_screen)          # 2
        self.stacked_widget.addWidget(self.event_screen)             # 3
        self.stacked_widget.addWidget(self.info_screen)              # 4
        self.stacked_widget.addWidget(self.settings_screen)          # 5
        self.stacked_widget.addWidget(self.live_settings_screen)     # 6
        self.stacked_widget.addWidget(self.playback_settings_screen) # 7
        self.stacked_widget.addWidget(self.roi_setup_screen)         # 8

        self.switch_screen(0)

        # 부저 주입: 100ms마다 침입 여부 확인 후 부저 조작
        self._buzzer = buzzer
        self._shared_states = shared_states
        if buzzer is not None and shared_states is not None:
            self._buzzer_timer = QTimer(self)
            self._buzzer_timer.timeout.connect(self._update_buzzer)
            self._buzzer_timer.start(100)

    def _update_buzzer(self):
        """비점장 침입 여부를 읽어 부저를 제어"""
        for state in self._shared_states:
            if state.is_intruding():
                self._buzzer.activate()
                return
        self._buzzer.deactivate()

    def switch_screen(self, index):
        if index == 2:
            self.playback_screen.load_files()
        self.stacked_widget.setCurrentIndex(index)

    def closeEvent(self, event):
        # 카메라/녹화/AI 정리는 pipeline(main.py)의 _cleanup()이 담당
        if hasattr(self, '_buzzer_timer'):
            self._buzzer_timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
