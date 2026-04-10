import json
import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QScrollArea
from PyQt5.QtCore import Qt
from config.settings import PROJECT_ROOT

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

    def showEvent(self, event):
        """화면이 표시될 때마다 로그를 새로 불러옴"""
        self.load_events()
        super().showEvent(event)

    # 기록된 이벤트(위험 감지 로그 등)를 불러와서 화면에 하나씩 그려주는 함수
    def load_events(self):
        # 기존 항목 모두 제거
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = self._read_log_entries()

        if not entries:
            lbl = QLabel("기록된 이벤트가 없습니다.")
            lbl.setStyleSheet("color: #888; font-size: 18px; padding: 15px; background-color: #222; border-radius: 8px; border: 1px solid #444;")
            lbl.setAlignment(Qt.AlignCenter)
            self.list_layout.addWidget(lbl)
            return

        # 최신 항목이 위에 오도록 역순으로 표시 (최대 100개)
        for text, level in reversed(entries[-100:]):
            lbl = QLabel(text)
            if level == "ERROR":
                style = "color: #ff6666;"
            elif level == "WARNING":
                style = "color: #ffcc00;"
            else:
                style = "color: white;"
            lbl.setStyleSheet(
                f"{style} font-size: 16px; padding: 12px; "
                "background-color: #222; border-radius: 8px; border: 1px solid #444;"
            )
            lbl.setWordWrap(True)
            self.list_layout.addWidget(lbl)

    # UI 이벤트 화면에 표시할 이벤트 타입 목록
    # 여기에 없는 타입은 디버그용으로만 남고 화면에 표시되지 않음
    _DISPLAY_EVENTS = {
        "DETECTION_RESULT",
    }

    def _read_log_entries(self):
        """logs/event_project.log 에서 이벤트 항목 파싱. [(표시 텍스트, 레벨), ...] 반환
        _DISPLAY_EVENTS에 해당하는 항목만 반환 (운영/디버그 로그 제외)
        """
        log_path = os.path.join(PROJECT_ROOT, "logs", "event_project.log")
        if not os.path.exists(log_path):
            return []

        entries = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 형식: "2026-04-10 18:51:56 - LEVEL - {json}"
                    parts = line.split(" - ", 2)
                    if len(parts) < 3:
                        continue
                    timestamp_str, level, raw = parts[0], parts[1], parts[2]
                    try:
                        data = json.loads(raw)
                        event = data.get("event", "")

                        # 표시 대상 이벤트 타입이 아니면 건너뜀
                        if event not in self._DISPLAY_EVENTS:
                            continue

                        msg_data = data.get("data", {})
                        cam_id = msg_data.get("cam", "?")
                        roi_count = msg_data.get("roi_count", "?")
                        text = f"{timestamp_str}에 위험 영역 내 객체 탐지 : {roi_count} (CAM {cam_id})"
                    except (json.JSONDecodeError, KeyError):
                        continue  # 파싱 불가 줄은 UI에 표시하지 않음
                    entries.append((text, level))
        except Exception:
            pass
        return entries