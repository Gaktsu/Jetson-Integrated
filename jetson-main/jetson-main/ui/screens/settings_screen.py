from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QTimer
import threading

from config import settings as settings
from system.storage import format_save_dir
from utils.logger import get_logger, EventType

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

        # 1. 저장소 포맷 (모든 영상 삭제) 버튼 생성 및 스타일 적용
        self.btn_format = QPushButton("저장소 포맷 (모든 영상 삭제)")
        self.btn_format.setStyleSheet(setting_btn_style)
        self.btn_format.clicked.connect(self._format_storage)

        # 1.5 상시 녹화 모드 토글 버튼
        self.btn_full_recording = QPushButton(self._recording_btn_text())
        self.btn_full_recording.setStyleSheet(setting_btn_style)
        self.btn_full_recording.setCheckable(True)
        self.btn_full_recording.setChecked(settings.RECORDING_MODE == "full")
        self.btn_full_recording.clicked.connect(self._toggle_recording_mode)

        # 2. 시스템 종료 버튼 — UI를 닫고 백그라운드(pipeline) 정리 후 완전 종료
        self.btn_reboot = QPushButton("시스템 종료")
        self.btn_reboot.setStyleSheet(setting_btn_style)
        self.btn_reboot.clicked.connect(self._shutdown)

        # 만들어진 버튼들을 세로 정렬 레이아웃에 차례대로 넣음
        self.settings_layout.addWidget(self.btn_format)
        self.settings_layout.addWidget(self.btn_full_recording)
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

    def _shutdown(self):
        """UI 창을 즉시 제거하고 QApplication 이벤트루프를 종료.
        main.py의 finally 블록에서 _cleanup()이 별도 스레드로 실행된다."""
        from PyQt5.QtCore import QTimer
        from PyQt5.QtWidgets import QApplication

        # 1) 실행 중인 QTimer 먼저 정지
        live = getattr(self.main_window, 'live_screen', None)
        if live is not None and hasattr(live, 'timer'):
            live.timer.stop()
        if hasattr(self.main_window, '_buzzer_timer'):
            self.main_window._buzzer_timer.stop()

        # 2) 모든 창을 닫고 원도우 매니저에 파괴 신호 전달—화면에서 즉시 사라짐
        QApplication.closeAllWindows()
        QApplication.processEvents()

        # 3) 이벤트 루프 종료
        QTimer.singleShot(50, QApplication.quit)

    def _recording_btn_text(self) -> str:
        return f"상시 녹화 모드: {'ON' if settings.RECORDING_MODE == 'full' else 'OFF'}"

    def _toggle_recording_mode(self):
        """Toggle between 'event' and 'full' recording modes at runtime."""
        logger = get_logger("ui.settings")
        try:
            if settings.RECORDING_MODE == "full":
                settings.RECORDING_MODE = "event"
            else:
                settings.RECORDING_MODE = "full"
            # Update button text to reflect new state
            self.btn_full_recording.setText(self._recording_btn_text())
            self.btn_full_recording.setChecked(settings.RECORDING_MODE == "full")
            logger.event_info(EventType.USER_INPUT, "RECORDING_MODE 토글", {"mode": settings.RECORDING_MODE})
        except Exception as e:
            logger.event_error(EventType.ERROR_OCCURRED, "녹화 모드 토글 실패", {"error": str(e)})

    def _format_storage(self):
        """Ask for confirmation and erase all files under SAVE_DIR."""
        reply = QMessageBox.question(
            self,
            "저장소 포맷 확인",
            "저장소의 모든 영상 및 폴더를 삭제합니다. 계속 진행하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        logger = get_logger("ui.settings")
        logger.event_info(EventType.USER_INPUT, "저장소 포맷 시작")

        def _worker():
            deleted = format_save_dir()

            def _notify():
                QMessageBox.information(self, "포맷 완료", f"포맷 완료: {deleted} 항목이 삭제되었습니다.")

            QTimer.singleShot(0, _notify)

        threading.Thread(target=_worker, daemon=True).start()

