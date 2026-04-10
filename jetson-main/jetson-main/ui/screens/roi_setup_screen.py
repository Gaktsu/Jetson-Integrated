import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

class RoiSetupScreen(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        
        # 화면 전체 배경을 검은색으로 강제 지정
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: black;")
        
        # 원본 사진(base_frame)과 화면에 띄울 그림용 사진(display_frame)을 담을 변수
        self.base_frame = None
        self.display_frame = None
        # 현재 설정 중인 카메라 번호를 저장할 변수
        self.current_cam_idx = 0
        
        # 찍은 4개의 점 좌표를 저장할 리스트
        # 화면 해상도가 달라져도 정확한 위치를 잡기 위해 0.0 ~ 1.0 사이의 비율값(정규화)으로 저장
        self.pts_norm = []
        # 마우스를 움직일 때 따라다니는 임시 점(점선용)의 좌표
        self.temp_pt_norm = None

        # 상단 타이틀
        self.title_label = QLabel("위험구역 선 그리기 (ROI 설정)", self)
        self.title_label.setGeometry(0, 10, 800, 30)
        self.title_label.setStyleSheet("color: white; font-size: 20px; font-weight: bold; background: transparent;")
        self.title_label.setAlignment(Qt.AlignCenter)

        # 가운데에 사진이 출력되고 실제로 터치가 이루어질 영역
        self.video_label = QLabel(self)
        self.video_label.setGeometry(80, 50, 640, 360) # 가로 640, 세로 360 크기로 중앙에 배치
        self.video_label.setStyleSheet("background-color: #111; border: 2px solid #555;")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # 마우스를 클릭하지 않고 이동만 해도 좌표를 추적하도록 설정
        self.video_label.setMouseTracking(True)
        # 라벨 영역 안에서 마우스를 누르거나 움직일 때 우리가 만든 함수를 실행하도록 연결
        self.video_label.mousePressEvent = self.roi_mouse_press
        self.video_label.mouseMoveEvent = self.roi_mouse_move

        # 좌측 하단 뒤로가기 버튼
        self.back_btn = QPushButton("←", self)
        self.back_btn.setGeometry(10, 415, 60, 50)
        self.back_btn.setStyleSheet("""
            QPushButton { background-color: rgba(0, 0, 0, 150); color: white; font-size: 24px; font-weight: bold; border-radius: 10px; border: 2px solid #555; }
            QPushButton:pressed { background-color: rgba(255, 255, 255, 100); }
        """)
        self.back_btn.clicked.connect(self.cancel_setup) # 설정 취소 후 복귀

        # 하단 도움말 문구
        self.help_label = QLabel("4개의 점을 터치하여 사다리꼴 모양의 구역을 만들어주세요.", self)
        self.help_label.setGeometry(80, 420, 440, 40)
        self.help_label.setStyleSheet("color: #00FFCC; font-size: 13px; font-weight: bold; background: transparent;")
        self.help_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        btn_style_base = "color: white; font-size: 16px; font-weight: bold; border-radius: 8px; border: none;"

        # 초기화 버튼
        self.reset_btn = QPushButton("초기화", self)
        self.reset_btn.setGeometry(525, 420, 110, 40)
        self.reset_btn.setStyleSheet(f"QPushButton {{ background-color: #555; {btn_style_base} }} QPushButton:pressed {{ background-color: #777; }}")
        self.reset_btn.clicked.connect(self.reset_points)

        # 저장 및 적용 버튼
        self.save_btn = QPushButton("저장 및 적용", self)
        self.save_btn.setGeometry(645, 420, 135, 40)
        self.save_btn.setStyleSheet(f"QPushButton {{ background-color: #28a745; {btn_style_base} }} QPushButton:pressed {{ background-color: #34ce57; }}")
        self.save_btn.clicked.connect(self.save_roi)

    # 실시간 녹화 화면에서 카메라 영상을 넘겨받아 배경으로 세팅하는 함수
    def set_base_frame(self, frame, cam_idx):
        # frame이 None이면 검은 화면으로 대체
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 넘어온 사진을 계산하기 편하게 640x480으로 고정
        self.base_frame = cv2.resize(frame, (640, 480)) 
        self.display_frame = self.base_frame.copy()
        self.current_cam_idx = cam_idx
        
        self.title_label.setText(f"CAM {cam_idx + 1} 위험구역 선 그리기 (ROI 설정)")
        
        # 기존에 AI 모듈에 저장되어 있던 이 카메라의 구역 좌표를 불러옴
        current_pts = self.main_window.live_screen.ai.get_roi_points(cam_idx)
        self.pts_norm = []
        for pt in current_pts:
            # 원본 좌표(640x480 기준)를 화면 비율인 0.0 ~ 1.0 값으로 변환하여 저장
            self.pts_norm.append((pt[0] / 640.0, pt[1] / 480.0))
            
        # 세팅이 끝났으면 화면에 선을 그림
        self.update_display()

    # 배경 사진 위에 점, 선, 면을 실시간으로 덧그려주는 그래픽 핵심 함수
    def update_display(self):
        if self.base_frame is None:
            return

        img = self.base_frame.copy()
        h, w = img.shape[:2]

        # BGR → RGB 변환 후 QImage 생성
        # img_rgb를 self에 보관해 GC로 인한 세그폴트 방지
        self._display_buf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_img = QImage(
            self._display_buf.data,
            w, h, w * 3,
            QImage.Format_RGB888
        )
        # QImage 데이터가 살아있는 동안 pixmap 생성
        pixmap = QPixmap.fromImage(q_img.copy())  # .copy()로 QImage 메모리 독립

        painter = QPainter(pixmap) # 그림을 그리는 붓(Painter) 생성
        
        # 펜(테두리 선) 종류 설정
        pen_line = QPen(QColor(0, 255, 255), 3, Qt.SolidLine) # 완성된 실선 (하늘색)
        pen_line_temp = QPen(QColor(255, 255, 0), 2, Qt.DashLine) # 마우스를 따라다니는 점선 (노란색)
        pen_point = QPen(QColor(255, 255, 0), 8, Qt.SolidLine) # 꼭짓점 동그라미 (노란색)
        
        # 비율(0~1)로 저장된 좌표들을 실제 사진 픽셀 좌표로 다시 계산
        points_q = []
        for p in self.pts_norm:
            points_q.append(QPoint(int(p[0] * w), int(p[1] * h)))
            
        if len(points_q) > 0:
            # 1. 찍힌 좌표들에 동그란 점을 그림
            painter.setPen(pen_point)
            for p in points_q:
                painter.drawPoint(p)
                
            # 2. 점이 2개 이상이면 점과 점 사이를 실선으로 이음
            painter.setPen(pen_line)
            if len(points_q) > 1:
                for i in range(len(points_q) - 1):
                    painter.drawLine(points_q[i], points_q[i+1])
            
            # 3. 아직 4개가 다 안 찍혔다면, 마지막 점부터 마우스 포인터까지 점선을 그음
            if self.temp_pt_norm is not None and len(points_q) < 4:
                temp_p = QPoint(int(self.temp_pt_norm[0] * w), int(self.temp_pt_norm[1] * h))
                painter.setPen(pen_line_temp)
                painter.drawLine(points_q[-1], temp_p)
                
            # 4. 점이 4개가 되면 마지막 점과 첫 번째 점을 잇고, 그릇 모양 안쪽을 반투명하게 칠함
            if len(points_q) == 4:
                painter.setPen(pen_line)
                painter.drawLine(points_q[3], points_q[0]) 
                painter.setBrush(QColor(0, 255, 255, 30)) # 안쪽을 채울 붓 설정 (알파값 30으로 반투명)
                painter.drawPolygon(points_q)
                
        painter.end() # 그림 그리기 종료
        # 완성된 그림을 라벨 크기(640x360)에 맞춰서 모니터에 띄움
        self.video_label.setPixmap(pixmap.scaled(640, 360, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

    # 사진 영역을 터치(클릭)했을 때 점을 추가하는 함수
    def roi_mouse_press(self, event):
        # 사진이 없거나 이미 4개를 다 찍었다면 무시
        if self.base_frame is None or len(self.pts_norm) >= 4:
            return
            
        x = event.x()
        y = event.y()
        
        # 클릭한 위치를 라벨 크기(640x360)로 나누어 0.0 ~ 1.0 사이의 비율로 만듬
        norm_x = x / 640.0
        norm_y = y / 360.0
        
        # 혹시라도 마우스가 삐져나가서 0 이하나 1 이상이 되지 않도록 0~1 사이로 고정
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        # 좌표 목록에 추가
        self.pts_norm.append((norm_x, norm_y))
        
        # 점의 개수에 따라 하단 안내 문구를 수정
        if len(self.pts_norm) == 4:
            self.temp_pt_norm = None
            self.help_label.setText("구역 완성! [저장 및 적용] 버튼을 눌러주세요.")
            self.help_label.setStyleSheet("color: #28a745; font-size: 13px; font-weight: bold; background: transparent;")
        else:
            self.help_label.setText(f"점 {len(self.pts_norm)}개 찍힘. 다음 점을 화면에 찍어주세요.")
            self.help_label.setStyleSheet("color: #00FFCC; font-size: 13px; font-weight: bold; background: transparent;")
            
        self.update_display() # 바뀐 점을 화면에 새로 그림

    # 마우스가 이동할 때마다 점선을 이어주기 위해 임시 좌표를 추적하는 함수
    def roi_mouse_move(self, event):
        if self.base_frame is None or len(self.pts_norm) == 0 or len(self.pts_norm) >= 4:
            return
            
        x = event.x()
        y = event.y()
        
        norm_x = x / 640.0
        norm_y = y / 360.0
        
        self.temp_pt_norm = (norm_x, norm_y)
        self.update_display()

    # 초기화 버튼을 누르면 찍었던 좌표를 모두 초기화
    def reset_points(self):
        self.pts_norm = []
        self.temp_pt_norm = None
        self.help_label.setText("4개의 점을 터치하여 사다리꼴 모양의 구역을 만들어주세요.")
        self.help_label.setStyleSheet("color: #00FFCC; font-size: 13px; font-weight: bold; background: transparent;")
        self.update_display()

    # 뒤로가기 버튼을 누르면 변경사항을 버리고 설정창으로 복귀
    def cancel_setup(self):
        self.reset_points()
        self.main_window.switch_screen(6)

    # 4개의 점이 다 찍히면 그 좌표를 AI 모듈에 저장하는 함수
    def save_roi(self):
        # 점이 4개가 아니면 경고창을 띄웁니다.
        if len(self.pts_norm) != 4:
            msg = QMessageBox(self)
            msg.setWindowTitle("알림")
            msg.setText("점을 4개 찍어서 도형을 완성해야 저장할 수 있습니다.")
            msg.setStyleSheet("QMessageBox { background-color: #222; } QLabel { color: white; font-size: 16px; font-weight: bold; } QPushButton { background-color: #0055aa; color: white; padding: 8px 20px; font-weight: bold; border-radius: 4px; font-size: 14px; }")
            msg.exec_()
            return

        final_pts = []
        # 좌표를 원본 AI 인식 크기인 640x480 크기에 맞춰 픽셀 값으로 변환
        for p in self.pts_norm:
            x_f = int(p[0] * 640)
            y_f = int(p[1] * 480) 
            final_pts.append((x_f, y_f))
            
        # 현재 카메라 번호와 함께 변환된 좌표를 AI 객체로 넘겨 저장
        self.main_window.live_screen.ai.set_roi_points(self.current_cam_idx, final_pts)
        
        # 저장 성공 안내창
        msg = QMessageBox(self)
        msg.setWindowTitle("저장 완료")
        msg.setText(f"CAM {self.current_cam_idx + 1} 위험 구역 설정이 저장되었습니다.")
        msg.setStyleSheet("QMessageBox { background-color: #222; } QLabel { color: white; font-size: 16px; font-weight: bold; } QPushButton { background-color: #28a745; color: white; padding: 8px 20px; font-weight: bold; border-radius: 4px; font-size: 14px; }")
        
        if msg.exec_() == QMessageBox.Ok:
            self.cancel_setup() # 저장이 완료되면 설정 화면으로 복귀