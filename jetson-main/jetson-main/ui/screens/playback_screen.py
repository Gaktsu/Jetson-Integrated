import os
import cv2
import shutil
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QFrame, QDialog, QSlider, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# [1] 비디오 플레이어 팝업창 클래스
class VideoPlayerPopup(QDialog):
    def __init__(self, filepath, filename, parent=None):
        super().__init__(parent)
        # 기본 윈도우 창 테두리를 없애고 팝업 형태로 설정
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #555; border-radius: 10px;")
        self.setFixedSize(680, 580) # 팝업창의 전체 크기

        self.filepath = filepath
        self.is_playing = True # 처음 창이 열릴 때 바로 재생되도록 설정

        # 전체 세로 정렬 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # 상단: 재생 중인 파일명을 보여주는 텍스트
        title = QLabel(f"재생 중: {filename}", self)
        title.setStyleSheet("color: #00FFCC; font-size: 18px; font-weight: bold; border: none;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 중앙: 실제 영상이 출력될 라벨
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # 하단: 조작 버튼들을 가로로 배치하기 위한 레이아웃
        controls_layout = QHBoxLayout()
        
        # 재생/일시정지 버튼
        self.play_btn = QPushButton("일시정지")
        self.play_btn.setFixedSize(100, 40)
        self.play_btn.setStyleSheet("""
            QPushButton { background-color: #0055aa; color: white; font-size: 16px; font-weight: bold; border-radius: 5px; border: none; }
            QPushButton:pressed { background-color: #0077ff; }
        """)
        self.play_btn.clicked.connect(self.toggle_play)

        # 영상 탐색을 위한 슬라이더
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #999; height: 8px; background: #333; margin: 2px 0; border-radius: 4px; }
            QSlider::handle:horizontal { background: #00FFCC; border: 1px solid #5c5c5c; width: 18px; margin: -5px 0; border-radius: 9px; }
        """)
        self.slider.sliderMoved.connect(self.set_position) # 드래그 시 영상 위치 이동

        # 창 닫기 버튼
        self.close_btn = QPushButton("닫기")
        self.close_btn.setFixedSize(80, 40)
        self.close_btn.setStyleSheet("""
            QPushButton { background-color: #aa0000; color: white; font-size: 16px; font-weight: bold; border-radius: 5px; border: none; }
            QPushButton:pressed { background-color: #ff4444; }
        """)
        self.close_btn.clicked.connect(self.close_player)

        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.slider)
        controls_layout.addWidget(self.close_btn)
        layout.addLayout(controls_layout)

        # OpenCV를 이용해 선택한 동영상 파일을 불러옴
        self.cap = cv2.VideoCapture(self.filepath)
        # 영상의 총 프레임 수와 초당 프레임(FPS)을 계산
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: 
            self.fps = 15.0 # FPS를 읽어오지 못할 경우 기본값 15로 설정

        # 슬라이더의 범위를 0부터 영상의 끝(총 프레임 수)까지로 설정
        self.slider.setRange(0, self.total_frames)

        # 타이머를 이용해 영상의 원본 속도(FPS)에 맞춰 프레임을 화면에 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(int(1000 / self.fps))

    # 타이머에 의해 반복 실행되며 영상 프레임을 한 장씩 읽어와 화면에 표시 함수
    def update_frame(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                
                # 영상이 재생됨에 따라 슬라이더(진행 바)의 위치도 자동으로 움직이게 함
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.slider.blockSignals(True) # 슬라이더 이동 이벤트가 중복 발생하지 않도록 잠시 차단
                self.slider.setValue(current_frame)
                self.slider.blockSignals(False)
            else:
                # 영상을 끝까지 다 봤다면 정지 상태로 바꿈
                self.is_playing = False
                self.play_btn.setText("재생")
                self.timer.stop()

    # 읽어온 프레임을 화면 크기에 맞춰 조절하고 UI에 띄워주는 함수
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        h, w, c = frame_resized.shape
        q_img = QImage(frame_resized.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    # 재생 및 일시정지 버튼을 눌렀을 때 실행되는 함수
    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.play_btn.setText("재생")
        else:
            # 영상 끝에서 다시 재생을 누르면 맨 처음으로 되돌아가서 재생
            if self.slider.value() >= self.total_frames - 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing = True
            self.timer.start(int(1000 / self.fps))
            self.play_btn.setText("일시정지")

    # 슬라이더를 마우스로 잡고 끌었을 때 해당 위치로 영상을 탐색하는 함수
    def set_position(self, position):
        """ 합니다. """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
    # 닫기 버튼을 누르면 재생을 멈추고 파일을 메모리에서 해제한 뒤 창을 닫는 함수
    def close_player(self):
        self.timer.stop()
        self.cap.release()
        self.reject()



# [2] 영상 재생 목록 창 클래스
class PlaybackScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")

        # 영상 목록이 많아질 것을 대비해 스크롤이 가능한 영역
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setGeometry(0, 50, 800, 350) 
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.list_widget = QWidget()
        self.list_widget.setStyleSheet("background-color: transparent;")
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setAlignment(Qt.AlignTop)
        self.list_layout.setSpacing(10) 
        self.scroll_area.setWidget(self.list_widget)

        overlay_style = "background-color: rgba(0, 0, 0, 150); color: white; font-weight: bold; border-radius: 5px; padding: 5px;"

        # 좌측 상단: 저장된 영상들의 총 용량을 보여주는 텍스트
        self.capacity_label = QLabel("총 용량: 계산중...", self)
        self.capacity_label.setGeometry(10, 10, 200, 30)
        self.capacity_label.setStyleSheet(overlay_style + "font-size: 14px;")
        self.capacity_label.setAlignment(Qt.AlignCenter)

        # 중앙 상단 타이틀
        self.title_label = QLabel("녹화 영상 목록", self)
        self.title_label.setGeometry(300, 10, 200, 30)
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold; background: transparent;")
        self.title_label.setAlignment(Qt.AlignCenter)

        btn_style = """
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 28px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """

        # 좌측 하단 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 410, 60, 60)
        self.back_btn.setStyleSheet(btn_style)
        self.back_btn.clicked.connect(lambda: self.main_window.switch_screen(1))

        # 우측 하단 영상 재생 전용 설정 버튼
        self.set_btn = QPushButton("⚙", self)
        self.set_btn.setGeometry(730, 410, 60, 60)
        self.set_btn.setStyleSheet(btn_style)
        self.set_btn.clicked.connect(lambda: self.main_window.switch_screen(7))

    # 화면이 열릴 때마다 records 폴더를 읽어서 파일 목록 UI를 최신화하는 함수
    def load_files(self):
        # 목록을 새로 구성하기 위해 기존에 화면에 표시되던 항목들을 모두 지움
        for i in reversed(range(self.list_layout.count())): 
            widget = self.list_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        record_dir = "records"
        total_size_bytes = 0

        # 폴더가 없으면 새로 만듬
        if not os.path.exists(record_dir):
            os.makedirs(record_dir)

        # 폴더 내의 .avi 파일들을 가져와서 최신순(내림차순)으로 정렬
        files = sorted([f for f in os.listdir(record_dir) if f.endswith('.avi')], reverse=True)

        # 파일이 하나도 없을 경우 안내 메시지 출력
        if not files:
            empty_lbl = QLabel("저장된 녹화 영상이 없습니다.")
            empty_lbl.setStyleSheet("color: gray; font-size: 18px;")
            empty_lbl.setAlignment(Qt.AlignCenter)
            self.list_layout.addWidget(empty_lbl)
            self.capacity_label.setText("총 용량: 0 MB")
            return

        # 파일을 하나씩 꺼내어 화면 목록에 추가하고, 파일 용량을 누적
        for file in files:
            filepath = os.path.join(record_dir, file)
            total_size_bytes += os.path.getsize(filepath) 
            
            row = self.create_file_row(file, filepath)
            self.list_layout.addWidget(row)

        # 계산된 총 용량을 MB 또는 GB 단위로 변환해서 화면 상단에 표시
        total_mb = total_size_bytes / (1024 * 1024)
        if total_mb >= 1024:
            total_gb = total_mb / 1024
            self.capacity_label.setText(f"총 용량: {total_gb:.2f} GB")
        else:
            self.capacity_label.setText(f"총 용량: {total_mb:.1f} MB")

    # 동영상 파일 1개당 [파일명 + 백업버튼 + 재생버튼] 으로 구성된 가로 한 줄을 만드는 함수
    def create_file_row(self, filename, filepath):
        row_frame = QFrame()
        row_frame.setStyleSheet("background-color: #222; border-radius: 8px; border: 1px solid #444;")
        row_frame.setFixedHeight(70)

        row_layout = QHBoxLayout(row_frame)
        row_layout.setContentsMargins(15, 10, 15, 10)

        # 파일명을 분석해서 보기 편하게 날짜, 시간 단위로 분리
        try:
            parts = filename.replace(".avi", "").split("_")
            date_str = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}" 
            time_str = f"{parts[1][:2]}:{parts[1][2:4]}:{parts[1][4:]}" 
            cam_str = parts[2]
            display_name = f"일자: {date_str}  |  시간: {time_str}  |  {cam_str}"
        except:
            display_name = filename # 형식이 다를 경우 원본 이름 그대로 표시

        name_lbl = QLabel(display_name)
        name_lbl.setStyleSheet("color: white; font-size: 18px; font-weight: bold; border: none; background: transparent;")

        btn_style = """
            QPushButton { background-color: #555; color: white; font-size: 16px; font-weight: bold; border-radius: 5px; padding: 10px 20px; border: none; }
            QPushButton:pressed { background-color: #00FFCC; color: black; }
        """

        # 백업 버튼 설정
        save_btn = QPushButton("백업")
        save_btn.setStyleSheet(btn_style)
        save_btn.clicked.connect(lambda chk, fp=filepath, fn=filename: self.backup_file(fp, fn))
        
        # 재생 버튼 설정
        play_btn = QPushButton("재생")
        play_btn.setStyleSheet(btn_style.replace("#555", "#0055aa")) # 재생 버튼만 파란색으로 변경
        play_btn.clicked.connect(lambda chk, fp=filepath, fn=filename: self.open_player_popup(fp, fn))

        row_layout.addWidget(name_lbl, 1)
        row_layout.addWidget(save_btn)
        row_layout.addWidget(play_btn)

        return row_frame

    # 영상 재생 팝업창을 화면 중앙에 띄움
    def open_player_popup(self, filepath, filename):
        popup = VideoPlayerPopup(filepath, filename, self)
        popup.exec_()

    # 파일을 USB나 다른 경로로 복사하는 기능
    def backup_file(self, filepath, filename):
        # 사용자가 복사할 위치를 직접 지정할 수 있도록 폴더 선택 창을 띄움
        dest_dir = QFileDialog.getExistingDirectory(self, "백업할 폴더를 선택하세요", "")
        
        if dest_dir:
            try:
                # 선택한 폴더 경로에 파일명 합치기
                dest_path = os.path.join(dest_dir, filename)
                # shutil 라이브러리를 이용해 파일을 복사
                shutil.copy2(filepath, dest_path)
                
                # 성공 메시지 팝업창 띄우기
                msg = QMessageBox(self)
                msg.setWindowTitle("백업 완료")
                msg.setText(f"백업이 완료되었습니다.\n저장 위치: {dest_path}")
                msg.setStyleSheet("QMessageBox { background-color: #222; } QLabel { color: white; font-size: 16px; font-weight: bold; } QPushButton { background-color: #0055aa; color: white; padding: 8px 20px; font-weight: bold; border-radius: 4px; font-size: 14px; }")
                msg.exec_()
            except Exception as e:
                # 에러 발생 시 실패 메시지 팝업창 띄우기
                msg = QMessageBox(self)
                msg.setWindowTitle("백업 실패")
                msg.setText(f"백업 중 오류가 발생했습니다.\n{str(e)}")
                msg.setStyleSheet("QMessageBox { background-color: #222; } QLabel { color: white; font-size: 16px; font-weight: bold; } QPushButton { background-color: #aa0000; color: white; padding: 8px 20px; font-weight: bold; border-radius: 4px; font-size: 14px; }")
                msg.exec_()