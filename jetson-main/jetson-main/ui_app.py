"""
PyQt5 UI 독립 실행 진입점 (통합 임시 테스트용)
- 기존 main.py(jetson pipeline)와 무관하게 UI만 단독 실행
- 파일 경로: jetson-main/jetson-main/ui_app.py
- 실행: python3 ui_app.py
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget

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
    def __init__(self):
        super().__init__()

        self.setWindowTitle("통합 테스트 UI")
        self.setGeometry(100, 100, 800, 480)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.live_screen = LiveScreen(self)
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

    def switch_screen(self, index):
        if index == 2:
            self.playback_screen.load_files()
        self.stacked_widget.setCurrentIndex(index)

    def closeEvent(self, event):
        for cap in self.live_screen.caps:
            if cap is not None and cap.isOpened():
                cap.release()
        for writer in self.live_screen.writers:
            if writer is not None:
                writer.release()
        self.live_screen.ai.cleanup()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
