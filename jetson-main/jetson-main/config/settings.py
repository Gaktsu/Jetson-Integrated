"""
프로젝트 전역 설정 및 상수
"""
import os

# 프로젝트 루트 디렉토리
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 모델 설정
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL = "best.pt"
MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL)

# 카메라 설정
# Jetson: [0, 2] (video0, video2)
# Windows: [0, 1] (일반적으로 순차적)
CAMERA_INDICES = [0, 2]  # 사용 가능한 카메라 인덱스 리스트
CAMERA_INDEX = CAMERA_INDICES[0]  # 기본 카메라
CAMERA_BUFFER_SIZE = 1
CAMERA_MAX_RETRIES = 10
CAMERA_RETRY_DELAY = 5.0  # 초
CAMERA_FPS = 27.0         # 카메라 실제 FPS (VideoWriter 기준, 저장 영상 재생 속도에 영향)

# YOLO 추론 설정
INFER_STRIDE = 1         # N프레임마다 1회 추론 (1=매 프레임, 3=3프레임 중 1회)
CONFIDENCE_THRESHOLD = 0.5
INFER_IMGSZ = 320        # 추론 입력 해상도 (TensorRT 전환 시에도 이 값 사용)
INFER_HALF = True        # FP16 추론 활성화 (Jetson GPU 권장)
INFER_DEVICE = "cuda:0"  # 추론 디바이스 ("cuda:0": GPU 강제, "cpu": CPU 전용)
TARGET_CLASS_ID = 0      # person 클래스

# 객체 추적 설정
# True : model.track(persist=True) 사용 — TTC/동적ROI 기능에 필요
# False: model()  단순 탐지 사용 (추적 불필요 시 CPU·메모리 절약)
ENABLE_TRACKING = True

# 동적 ROI 설정 (yolo_test-main 기준)
# 지게차 속도 1 단계당 ROI 상단을 위로 확장할 픽셀 수
# 속도가 빠를수록 ROI가 앞으로(위로) 늘어나 제동거리를 확보함
DYNAMIC_ROI_PX_PER_SPEED = 30  # pixels / speed-level

# FPS 계산 설정
FPS_UPDATE_INTERVAL = 1.0  # 초

# 저장 설정
SAVE_DIR = os.path.join(PROJECT_ROOT, "SaveVideos")
os.makedirs(SAVE_DIR, exist_ok=True)

# 녹화 설정
# "event": 침입 이벤트 기반 녹화, "full": 전체 상시 녹화
RECORDING_MODE = "event"
EVENT_RECORD_BUFFER_SEC = 15.0
EVENT_RECORD_POST_SEC = 15.0

# 서버 업로드 설정
SERVER_URL = "http://3.212.81.201:5000"
UPLOAD_ENABLED = True
UPLOAD_URL = f"{SERVER_URL}/upload"
UPLOAD_DEVICE_ID = "jetson1"
UPLOAD_DEVICE_KEY = "a3f9c7e1d2b4a6f8c0e1d3b5a7c9e2f4"
UPLOAD_REL_DIR = "video"
UPLOAD_TIMEOUT_SEC = 120
UPLOAD_MAX_RETRIES = 3
UPLOAD_RETRY_DELAY_SEC = 2.0

# 이벤트 JSON 로그 전송 설정 (yolo_test-main 기준)
# 영상과 별개로 경고 레벨 메타데이터를 JSON으로 빠르게 전송
EVENT_LOG_ENABLED = True                             # False로 바꾸면 전송 비활성화
EVENT_LOG_URL = f"{SERVER_URL}/stats/hourly"  # 백엔드 팀에서 공유된 API 주소
EVENT_LOG_TIMEOUT_SEC = 1.0                          # 영상 끊김 방지용 짧은 타임아웃
EVENT_LOG_COOLDOWN_SEC = 4.0                         # 동일 카메라 연속 전송 최소 간격 (중복 방지)

# 폴더 관리 설정
MAX_EVENT_FOLDERS = 100  # event 모드 최대 폴더 개수 (0: 무제한)
MAX_FULL_FOLDERS = 50    # full 모드 최대 폴더 개수 (0: 무제한)

# 저장 영상 오버레이 설정
# True: 바운딩박스/FPS/텍스트 오버레이가 포함된 영상 저장
# False: 오버레이 없이 원본 영상 저장
SAVE_WITH_OVERLAY = False

# 테스트 설정 (watchdog 테스트용)
WATCHDOG_TEST_MODE = False  # True로 설정하면 10초 후 프로그램 강제 종료
WATCHDOG_TEST_DELAY = 10    # 테스트 모드 시 몇 초 후 종료할지

# 디스플레이 설정
# "switch": 한 번에 한 카메라 표시, [C] 키로 전환
# "split":  모든 카메라를 분할 화면으로 동시 표시
DISPLAY_MODE = "split"
WINDOW_NAME = "Real-time YOLO"

# 마이크 설정
# ALSA 장치 이름: 'default', 'hw:1,0' 등 (arecord -l 로 확인)
MIC_DEVICE = "default"
MIC_SAMPLE_RATE = 16000   # Hz
MIC_CHANNELS = 1
MIC_CHUNK_SIZE = 1024     # 프레임 단위 읽기 크기

# 터치스크린 설정
# /dev/input/eventX 형식, None이면 자동 탐색
TOUCH_DEVICE_PATH = None
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# 색상 설정 (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLACK = (0, 0, 0)
