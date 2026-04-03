# Jetson Person Detection System - 코드 상세 설명

이 문서는 jetson 프로젝트의 모든 파일과 코드를 이해하기 쉽게 설명합니다.

---

## 📁 프로젝트 구조 개요

```
jetson/
├── main.py                 # 🎯 프로그램의 시작점 (모든 것을 조율하는 중심)
├── config/                 # ⚙️ 설정 파일
│   └── settings.py         # 모든 설정값 (카메라, 모델, 경로 등)
├── errors/                 # ❌ 에러 정의
│   └── enums.py           # 에러 타입 열거형
├── hardware/              # 🔧 하드웨어 제어
│   ├── camera.py          # 카메라 초기화 및 영상 캡처
│   ├── buzzer.py          # 부저 제어 (경고음)
│   ├── gps.py             # GPS 센서 (위치 정보)
│   └── imu.py             # IMU 센서 (가속도/자이로)
├── vision/                # 👁️ 영상 처리 및 AI
│   ├── yolo_infer.py      # YOLO 추론 및 스레드 관리
│   ├── preprocess.py      # 전처리 (크기 조정, 정규화)
│   └── postprocess.py     # 후처리 (침입 감지, 화면 표시)
├── utils/                 # 🛠️ 유틸리티
│   ├── logger.py          # 로깅 시스템
│   ├── time_utils.py      # 시간 관련 유틸
│   └── sensor_sync.py     # 센서 데이터 동기화
└── system/                # 🖥️ 시스템 관리
    ├── watchdog.py        # 프로그램 감시 및 자동 재시작
    ├── storage.py         # 저장소 용량 관리
    └── autostart.py       # 부팅 시 자동 실행 설정
```

---

## 🎯 main.py - 프로그램의 심장

### 역할
프로그램의 **시작점**이자 **조율자**입니다. 모든 하드웨어와 소프트웨어 모듈을 초기화하고 스레드를 관리합니다.

### 주요 흐름

```python
def main():
    # 1단계: YOLO 모델 로드
    yolo_model = YOLOInference(...)
    
    # 2단계: 카메라 초기화 (여러 대 가능)
    cameras = []
    states = []  # 각 카메라마다 공유 상태 객체
    
    # 3단계: 스레드 시작
    # - 캡처 스레드 (각 카메라마다)
    # - 추론 스레드 (1개, 모든 카메라 공유)
    # - 저장 스레드 (1개, 모든 카메라 공유)
    # - 센서 스레드 (GPS, IMU)
    
    # 4단계: 메인 루프
    # - 화면에 프레임 표시
    # - 부저 제어
    # - 키 입력 처리
```

### 핵심 개념 설명

#### 1. 왜 여러 스레드를 사용하나요?
```python
# 각 작업이 동시에 실행되어야 하기 때문입니다.
# 스레드(Thread) = 동시에 여러 작업을 하는 방법

capture_thread  # 카메라에서 계속 프레임을 가져옴
inference_thread  # 가져온 프레임을 AI로 분석
save_thread  # 필요한 프레임을 파일로 저장
display_loop  # 화면에 결과를 표시
```

#### 2. SharedState는 무엇인가요?
```python
# 스레드들이 정보를 공유하는 "게시판" 같은 것
state.latest_frame  # 가장 최근 프레임
state.last_detections  # 가장 최근 탐지 결과
state.last_intrusion  # 침입 발생 여부
state.frame_lock  # 동시 접근 방지 (충돌 방지)
```

#### 3. Queue는 무엇인가요?
```python
# 데이터를 전달하는 "우체통" 같은 것
save_queue.put(frame)  # 캡처 스레드가 프레임을 넣음
frame = save_queue.get()  # 저장 스레드가 프레임을 꺼냄
```

### 메인 루프 상세 설명

```python
while True:
    # 1. 현재 카메라의 최신 프레임 가져오기
    with states[current_camera_idx].frame_lock:
        frame_to_show = states[current_camera_idx].latest_frame.copy()
    
    # 2. 탐지 결과 가져오기
    with states[current_camera_idx].det_lock:
        detections = states[current_camera_idx].last_detections
        intrusion = states[current_camera_idx].last_intrusion
    
    # 3. 침입 시 부저 울리기
    if intrusion:
        buzzer.activate()
    else:
        buzzer.deactivate()
    
    # 4. 화면에 그리기
    frame_drawn = draw_detections(frame_to_show, detections, ...)
    cv2.imshow("Real-time YOLO", frame_drawn)
    
    # 5. 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # q 누르면 종료
        break
    elif key == ord('c'):  # c 누르면 카메라 전환
        current_camera_idx = (current_camera_idx + 1) % len(cameras)
```

### 용어 설명
- **스레드(Thread)**: 프로그램 안에서 동시에 실행되는 작업 흐름
- **Lock**: 여러 스레드가 동시에 같은 데이터를 수정하지 못하게 막는 자물쇠
- **Queue**: 스레드 간 안전하게 데이터를 주고받는 통로
- **Daemon Thread**: 메인 프로그램이 종료되면 자동으로 종료되는 스레드

---

## ⚙️ config/settings.py - 모든 설정값

### 역할
프로그램의 **모든 설정값**을 한 곳에 모아놓은 파일입니다. 여기만 수정하면 프로그램 동작을 바꿀 수 있습니다.

### 주요 설정 그룹

#### 1. 경로 설정
```python
PROJECT_ROOT = "프로젝트의 최상위 폴더"
MODEL_DIR = "YOLO 모델이 있는 폴더"
MODEL_PATH = "사용할 YOLO 모델 파일 경로"
SAVE_DIR = "녹화 영상을 저장할 폴더"
```

#### 2. 카메라 설정
```python
# 카메라 번호 리스트 (Jetson은 [0, 2], Windows는 [0, 1])
CAMERA_INDICES = [0, 2]

# 카메라 연결 실패 시 재시도 횟수
CAMERA_MAX_RETRIES = 10

# 재시도 간격 (초)
CAMERA_RETRY_DELAY = 5.0

# 카메라 백엔드 (Windows=DSHOW, Jetson=V4L2)
CAMERA_BACKEND = "CAP_DSHOW"
```

**왜 Jetson은 [0, 2]인가요?**
- Jetson은 video0, video1, video2 장치가 있는데
- video1은 메타데이터 장치라 실제 영상이 안 나옴
- 그래서 video0(0번)과 video2(2번)를 사용

#### 3. YOLO 추론 설정
```python
# N프레임마다 1번 추론 (예: 3이면 3프레임마다 1번)
INFER_STRIDE = 3  # 성능 향상을 위해

# 탐지 신뢰도 임계값 (0.5 = 50% 이상 확실할 때만 탐지)
CONFIDENCE_THRESHOLD = 0.5

# 입력 이미지 크기 (작을수록 빠르지만 정확도 낮아짐)
IMAGE_SIZE = 640

# 탐지할 클래스 (0 = person, COCO 데이터셋 기준)
TARGET_CLASS_ID = 0
```

**INFER_STRIDE가 뭔가요?**
```python
# 매 프레임마다 AI 추론을 하면 너무 느림
# 예: 30fps 카메라면 초당 30번 추론 = 부담

# INFER_STRIDE = 3 이면
# 프레임: 1, 2, 3, 4, 5, 6, 7, 8, 9...
#   추론:      ✓     ✓     ✓     ✓     # 3번째마다만 추론
```

#### 4. 경고 영역 설정
```python
# 화면 우측 20%를 경고 영역으로 설정
WARNING_ZONE_RATIO = 0.8

# 예: 화면 너비가 1000이면
# 경고 영역 = 화면의 800부터 1000까지 (우측 20%)
```

#### 5. 녹화 설정
```python
# 녹화 모드
RECORDING_MODE = "event"  # "event" 또는 "full"
# "event": 침입 발생 시만 녹화
# "full": 항상 녹화

# 이벤트 발생 전 버퍼 시간 (초)
EVENT_RECORD_BUFFER_SEC = 5.0  # 침입 감지 5초 전부터 저장

# 이벤트 종료 후 추가 녹화 시간 (초)
EVENT_RECORD_POST_SEC = 5.0  # 침입 끝난 후 5초 더 저장
```

#### 6. 화면 표시 설정
```python
WINDOW_NAME = "Real-time YOLO"  # 창 이름
FONT = "cv2.FONT_HERSHEY_SIMPLEX"  # 글꼴
FONT_SCALE = 0.6  # 글자 크기
FONT_THICKNESS = 2  # 글자 두께

# 색상 (BGR 순서 주의!)
COLOR_GREEN = (0, 255, 0)  # 안전 (녹색)
COLOR_RED = (0, 0, 255)  # 위험 (빨간색)
COLOR_YELLOW = (0, 255, 255)  # 경고 (노란색)
```

**왜 RGB가 아니라 BGR인가요?**
- OpenCV는 역사적 이유로 BGR 순서를 사용
- RGB: (빨강, 녹색, 파랑)
- BGR: (파랑, 녹색, 빨강) ← OpenCV
- 햇갈리지 않도록 주의!

---

## 🔧 hardware/camera.py - 카메라 제어

### 역할
카메라를 **초기화**하고 **영상을 캡처**하는 역할입니다.

### 주요 클래스: CameraCapture

```python
class CameraCapture:
    def __init__(self, camera_index: int):
        """
        카메라 객체 생성
        
        Args:
            camera_index: 카메라 번호 (0, 1, 2...)
        """
        self.camera_index = camera_index
        self.cap = None  # VideoCapture 객체
        
    def start(self, max_retries: int, retry_delay: float) -> bool:
        """
        카메라 시작 (재시도 기능 포함)
        
        Returns:
            성공하면 True, 실패하면 False
        """
        
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        프레임 읽기
        
        Returns:
            (성공여부, 프레임)
        """
        
    def release(self):
        """카메라 해제 (종료 시 호출)"""
```

### 주요 함수 설명

#### 1. diagnose_camera_error() - 카메라 오류 진단
```python
def diagnose_camera_error(camera_index: int) -> CameraError:
    """
    카메라가 왜 안 열리는지 원인을 찾아냅니다.
    
    진단 순서:
    1. 요청한 카메라 번호가 열리는가?
    2. 다른 백엔드(방법)로 시도했을 때 열리는가?
    3. 시스템에 다른 카메라라도 있는가?
    4. 다른 프로그램이 사용 중인가?
    
    Returns:
        CameraError.OK: 문제없음
        CameraError.DEVICE_NOT_FOUND: 카메라 없음
        CameraError.DEVICE_BUSY: 다른 프로그램이 사용 중
        CameraError.BACKEND_ERROR: 백엔드 문제
        CameraError.PERMISSION_DENIED: 권한 없음
    """
```

#### 2. open_camera_with_retry() - 재시도 기능
```python
def open_camera_with_retry(
    camera_index: int,
    max_retries: int = 10,
    retry_delay: float = 5.0
) -> Optional[cv2.VideoCapture]:
    """
    카메라 열기를 여러 번 시도합니다.
    
    왜 재시도가 필요한가요?
    - 카메라가 부팅 직후 아직 준비 안 됐을 수 있음
    - USB 연결이 불안정할 수 있음
    - 다른 프로그램이 잠깐 사용하다가 놓을 수 있음
    
    동작:
    1. 카메라 열기 시도
    2. 실패하면 오류 진단
    3. retry_delay초 기다림
    4. 다시 시도
    5. max_retries번까지 반복
    """
```

#### 3. OS별 백엔드 선택
```python
# Windows용
if IS_WINDOWS:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    # DSHOW = DirectShow (Windows 전용 카메라 인터페이스)

# Jetson용 (Linux)
elif IS_JETSON:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    # V4L2 = Video4Linux2 (Linux 표준 카메라 인터페이스)

# 기타
else:
    cap = cv2.VideoCapture(camera_index)
    # 자동 선택
```

**백엔드가 뭔가요?**
- 백엔드 = 카메라와 통신하는 방법
- Windows는 DirectShow 사용
- Linux는 Video4Linux2 사용
- 잘못된 백엔드 사용 시 카메라가 안 열림

### 사용 예시
```python
# 카메라 생성
camera = CameraCapture(0)

# 카메라 시작 (최대 10번 재시도, 5초 간격)
if camera.start(max_retries=10, retry_delay=5.0):
    print("카메라 열기 성공!")
    
    # 프레임 읽기
    while True:
        ok, frame = camera.read_frame()
        if ok:
            cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 카메라 해제
    camera.release()
else:
    print("카메라 열기 실패!")
```

---

## 🔊 hardware/buzzer.py - 부저 제어

### 역할
침입 감지 시 **부저를 울려** 경고합니다.

### 동작 원리
```python
# 능동 부저 = 전류만 주면 소리 나는 부저
# GPIO.HIGH → 부저 ON (삐--)
# GPIO.LOW → 부저 OFF (조용)
```

### 주요 메소드
```python
class Buzzer:
    def start(self) -> bool:
        """
        GPIO 핀 초기화
        - RPi.GPIO가 없으면 (Windows 등) 로그만 출력
        """
        
    def activate(self):
        """부저 ON (HIGH 전압)"""
        GPIO.output(self.pin, GPIO.HIGH)
        
    def deactivate(self):
        """부저 OFF (LOW 전압)"""
        GPIO.output(self.pin, GPIO.LOW)
        
    def stop(self):
        """GPIO 정리 (프로그램 종료 시)"""
        GPIO.cleanup(self.pin)
```

### GPIO 설정 설명
```python
# 핀 번호 방식
GPIO.setmode(GPIO.BOARD)  # 물리적 핀 번호 (1~40)
# 또는
GPIO.setmode(GPIO.BCM)  # GPIO 번호 (BCM 번호)

# 핀 설정
GPIO.setup(32, GPIO.OUT, initial=GPIO.LOW)
# 32번 핀을 출력용으로, 초기값은 LOW(꺼짐)
```

**BOARD vs BCM 차이는?**
- BOARD: 물리적 핀 번호 (보드에 적힌 번호)
- BCM: GPIO 칩셋의 번호
- BOARD 추천 (물리적으로 찾기 쉬움)

### 사용 예시
```python
buzzer = Buzzer(pin=32, use_board=True)
buzzer.start()

# 침입 감지 시
if intrusion:
    buzzer.activate()  # 삐--
else:
    buzzer.deactivate()  # 조용

# 종료 시
buzzer.stop()
```

---

## 📍 hardware/gps.py - GPS 센서

### 역할
GPS 모듈(NEO-6M)에서 **위치 정보**를 읽어옵니다.

### 동작 원리
```python
# GPS 모듈 → 시리얼 통신 → NMEA 문장 전송
# 예: $GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
```

### 주요 메소드
```python
class GPS:
    def start(self, max_retries: int, retry_delay: float) -> bool:
        """
        시리얼 포트 열기
        - Jetson: /dev/ttyTHS0 또는 /dev/ttyUSB0
        - Baudrate: 9600 (GPS 모듈 기본값)
        """
        
    def read_data(self):
        """
        GPS 데이터 한 줄 읽기
        
        Returns:
            NMEA 문장 (문자열)
            예: "$GPRMC,123519,A,4807.038,N,..."
        """
        
    def stop(self):
        """시리얼 포트 닫기"""
```

### NMEA 문장이란?
```
$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
  │     │     │    │       │    │       │   │     │      │      │
  │     │     │    │       │    │       │   │     │      │      └─ 체크섬
  │     │     │    │       │    │       │   │     │      └─ 날짜
  │     │     │    │       │    │       │   │     └─ 방위각
  │     │     │    │       │    │       │   └─ 속도
  │     │     │    │       │    │       └─ 경도 (E=동쪽)
  │     │     │    │       │    └─ 위도 (N=북쪽)
  │     │     │    │       └─ 위도 방향
  │     │     │    └─ 위도 (48도 07.038분)
  │     │     └─ 상태 (A=유효, V=무효)
  │     └─ UTC 시간
  └─ 메시지 타입 (GPRMC = 권장 최소 데이터)
```

### 파싱 예시
```python
# NMEA 문장을 위도/경도로 변환하는 것은
# yolo_infer.py의 _parse_nmea_lat_lon() 함수가 담당
```

---

## 🎯 vision/yolo_infer.py - AI 추론의 핵심

이 파일은 프로그램에서 **가장 복잡하고 중요**한 파일입니다.

### 역할
1. YOLO 모델로 사람 탐지
2. 여러 스레드 관리
3. 영상 저장 로직

### 주요 클래스와 함수

#### 1. YOLOInference - AI 모델 래퍼
```python
class YOLOInference:
    def __init__(self, model_path: str, conf: float, imgsz: int):
        """
        YOLO 모델 로드
        
        Args:
            model_path: 모델 파일 경로 (.pt 파일)
            conf: 신뢰도 임계값 (0.5 = 50%)
            imgsz: 입력 이미지 크기 (320, 640 등)
        """
        self.model = YOLO(model_path)
        
    def predict(self, frame: cv2.Mat) -> List[Tuple]:
        """
        프레임에서 사람 탐지
        
        Returns:
            [(x1, y1, x2, y2, cls_id, score), ...]
            x1,y1: 좌상단 좌표
            x2,y2: 우하단 좌표
            cls_id: 클래스 번호 (0=person)
            score: 신뢰도 (0~1)
        """
```

**예시 결과:**
```python
detections = [
    (100, 150, 300, 450, 0, 0.85),  # 사람 1
    (500, 200, 700, 480, 0, 0.92),  # 사람 2
]
# (100,150)부터 (300,450)까지 박스
# 클래스 0 (person), 신뢰도 85%
```

#### 2. SharedState - 스레드 간 데이터 공유
```python
class SharedState:
    """각 카메라마다 1개씩 생성됩니다"""
    
    def __init__(self):
        # 락 (동시 접근 방지)
        self.frame_lock = threading.Lock()
        self.det_lock = threading.Lock()
        
        # 프레임 정보
        self.latest_frame = None  # 가장 최근 프레임
        self.latest_frame_seq = -1  # 프레임 번호
        
        # 탐지 정보
        self.last_detections = []  # 탐지 결과 리스트
        self.last_intrusion = False  # 침입 여부
        self.last_intrusion_ts = 0.0  # 마지막 침입 시각
        
        # 센서 정보
        self.last_sensor_data = None  # GPS/IMU 데이터
        
        # 큐
        self.frame_queue = queue.Queue(maxsize=1)
        
        # 종료 이벤트
        self.stop_event = threading.Event()
```

**왜 Lock이 필요한가요?**
```python
# 여러 스레드가 동시에 같은 변수를 수정하면 문제 발생

# 스레드 A: state.latest_frame = frame1
# 스레드 B: state.latest_frame = frame2
# → 누가 먼저? 결과가 예측 불가능!

# Lock 사용:
with state.frame_lock:
    state.latest_frame = frame1  # A가 끝날 때까지 B는 대기
```

#### 3. capture_loop() - 캡처 스레드
```python
def capture_loop(cap, state, cam_id, save_queue):
    """
    무한 루프로 프레임을 계속 캡처합니다.
    
    동작:
    1. 카메라에서 프레임 읽기
    2. SharedState에 저장 (최신 프레임 갱신)
    3. 추론 큐에 프레임 넣기
    4. 저장 큐에 프레임 넣기
    5. 반복
    """
    frame_count = 0
    while not state.stop_event.is_set():
        ok, frame = cap.read_frame()
        if not ok:
            continue
            
        timestamp = time.time()
        
        # 1. SharedState 갱신
        with state.frame_lock:
            state.latest_frame = frame
            state.latest_frame_seq += 1
            
        # 2. 추론 큐에 넣기
        try:
            state.frame_queue.put_nowait((seq, timestamp, frame))
        except queue.Full:
            # 큐가 꽉 차면 오래된 것 버리고 새 것 넣기
            state.frame_queue.get_nowait()
            state.frame_queue.put_nowait((seq, timestamp, frame))
            
        # 3. 저장 큐에 넣기
        save_queue.put_nowait((cam_id, timestamp, frame.copy()))
```

**왜 frame.copy()를 하나요?**
```python
# frame은 메모리의 특정 위치를 가리킴
# 복사 안 하면 다음 프레임 읽을 때 덮어써짐

frame1 = cap.read()  # 메모리 위치 A
save_queue.put(frame1)  # 위치 A 참조

frame2 = cap.read()  # 메모리 위치 A를 재사용!
# → save_queue에 있던 frame1도 frame2로 바뀜!

# 해결: 복사본 만들기
save_queue.put(frame1.copy())  # 새 메모리 위치 B에 복사
```

#### 4. inference_loop() - 추론 스레드
```python
def inference_loop(model, states, sensor_getter, warning_zone_ratio):
    """
    모든 카메라의 프레임을 추론합니다 (중앙 집중식)
    
    동작:
    1. 각 카메라의 큐를 확인
    2. 새 프레임이 있으면 YOLO 추론
    3. 경고 영역 침입 여부 확인
    4. 결과를 SharedState에 저장
    """
    while True:
        for idx, state in enumerate(states):
            # 큐에서 프레임 꺼내기
            try:
                seq, timestamp, frame = state.frame_queue.get_nowait()
            except queue.Empty:
                continue  # 다음 카메라 확인
                
            # YOLO 추론
            detections = model.predict(frame)
            
            # 경고 영역 계산
            h, w = frame.shape[:2]
            warning_zone = (int(w * 0.8), 0, w, h)
            
            # 침입 확인
            intrusion = check_intrusion(detections, warning_zone)
            
            # 결과 저장
            with state.det_lock:
                state.last_detections = detections
                state.last_intrusion = intrusion
                if intrusion:
                    state.last_intrusion_ts = timestamp
```

**왜 중앙 집중식인가요?**
```python
# 방법 1: 카메라마다 추론 스레드 (X)
# - 카메라 2개 = 추론 스레드 2개
# - GPU를 2개 스레드가 동시 사용 → 충돌 가능
# - 비효율적

# 방법 2: 중앙 추론 스레드 1개 (O)
# - 모든 카메라의 프레임을 하나의 스레드가 순서대로 처리
# - GPU 안전하게 사용
# - 효율적
```

#### 5. save_loop() - 저장 스레드 (가장 복잡!)
```python
def save_loop(
    save_queue,
    stop_event,
    save_dir,
    fps_map,
    codec,
    sensor_getter,
    state_map,
    recording_mode,
    buffer_seconds,
    post_seconds
):
    """
    영상 저장 로직
    
    동작 모드:
    1. "full": 항상 녹화
    2. "event": 침입 발생 시만 녹화
    """
```

**이벤트 기반 녹화 상세 로직:**

```python
# 각 카메라마다 관리:
writers = {}  # VideoWriter 객체
buffers = {}  # 5초 버퍼
recording_active = {}  # 녹화 중인가?
post_deadline = {}  # 종료 데드라인

while True:
    # 큐에서 프레임 꺼내기
    cam_id, timestamp, frame = save_queue.get()
    
    # 🔥핵심: 모든 카메라 중 하나라도 침입이 있는지 확인
    intrusion_active = False
    for check_cam_id in state_map.keys():
        if state_map[check_cam_id].last_intrusion:
            intrusion_active = True
            break
    
    # 5초 버퍼 관리
    buffer = buffers.setdefault(cam_id, deque())
    buffer.append((timestamp, frame))
    # 5초 이전 프레임은 삭제
    while buffer and buffer[0][0] < timestamp - 5.0:
        buffer.popleft()
    
    # 📹 녹화 로직
    is_recording = recording_active.get(cam_id, False)
    
    if intrusion_active:
        # 침입 발생 중
        if not is_recording:
            # 새로 녹화 시작
            writer = VideoWriter(...)
            writers[cam_id] = writer
            recording_active[cam_id] = True
            
            # 버퍼의 모든 프레임 저장 (5초 전부터)
            for buf_ts, buf_frame in buffer:
                writer.write(buf_frame)
        
        # 현재 프레임 저장
        writers[cam_id].write(frame)
        
    else:
        # 침입 없음
        if is_recording:
            # 아직 녹화 중이면 5초 더 저장
            if post_deadline[cam_id] is None:
                post_deadline[cam_id] = timestamp + 5.0
            
            if timestamp <= post_deadline[cam_id]:
                writers[cam_id].write(frame)
            else:
                # 5초 지남 → 녹화 종료
                writers[cam_id].release()
                recording_active[cam_id] = False
```

**타임라인 예시:**
```
시간: 0s   5s   10s  15s  20s  25s  30s
침입: -    -    -    O    O    O    -    -    -    -
녹화: -    -    -    ▶================◼
          ↑         ↑              ↑
      5초 버퍼   침입 시작   5초 후 종료

설명:
- 10s: 침입 감지
- 5~10s: 버퍼에 있던 프레임 저장 (사전 녹화)
- 10~20s: 침입 중 계속 저장
- 20s: 침입 종료
- 20~25s: 5초 더 저장 (후속 녹화)
- 25s: 녹화 완전 종료
```

**다중 카메라 동시 녹화:**
```python
# 🔥 핵심 변경사항
# 이전: 각 카메라가 자기 침입만 확인
if state_map[cam_id].last_intrusion:
    # 이 카메라만 녹화

# ✅ 현재: 모든 카메라 중 하나라도 침입 있으면 전부 녹화
intrusion_active = False
for check_cam_id in state_map.keys():
    if state_map[check_cam_id].last_intrusion:
        intrusion_active = True
        break

# 예시:
# 카메라0에서 침입 감지
# → 카메라0, 카메라1 둘 다 녹화 시작
# → 다각도 영상 확보!
```

---

## 👁️ vision/postprocess.py - 후처리

### 주요 함수

#### 1. check_intrusion() - 침입 감지
```python
def check_intrusion(detections, warning_zone) -> bool:
    """
    탐지된 박스가 경고 영역과 겹치는지 확인
    
    Args:
        detections: [(x1, y1, x2, y2, cls_id, score), ...]
        warning_zone: (wx1, wy1, wx2, wy2)
    
    Returns:
        True: 침입 발생
        False: 안전
    """
    wx1, wy1, wx2, wy2 = warning_zone
    
    for x1, y1, x2, y2, cls_id, score in detections:
        # 박스가 경고 영역과 겹치는가?
        if not (x2 < wx1 or x1 > wx2 or y2 < wy1 or y1 > wy2):
            return True  # 겹침!
    
    return False  # 안전
```

**겹침 판정 로직:**
```python
# 두 사각형이 겹치지 않는 경우:
# 1. 박스가 경고 영역의 왼쪽에 있음: x2 < wx1
# 2. 박스가 경고 영역의 오른쪽에 있음: x1 > wx2
# 3. 박스가 경고 영역의 위쪽에 있음: y2 < wy1
# 4. 박스가 경고 영역의 아래쪽에 있음: y1 > wy2

# 위 4가지 중 하나라도 참이면 → 겹치지 않음
# 모두 거짓이면 → 겹침!

겹치지 않음 = (x2 < wx1) or (x1 > wx2) or (y2 < wy1) or (y1 > wy2)
겹침 = not (겹치지 않음)
```

#### 2. draw_detections() - 화면에 그리기
```python
def draw_detections(frame, detections, names, fps, ...):
    """
    프레임에 모든 정보를 그립니다
    
    그리는 것들:
    1. 경고 영역 (빨간 반투명)
    2. 탐지 박스 (녹색 또는 빨간색)
    3. 라벨 (person 0.85)
    4. WARNING! 문구
    5. FPS, 시간, 탐지 수
    6. Saving... 표시
    """
    
    # 1. 경고 영역 그리기
    overlay = frame.copy()
    cv2.rectangle(overlay, (wx1, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    # 0.3 = 30% 투명도
    
    # 2. 탐지 박스 그리기
    for x1, y1, x2, y2, cls_id, score in detections:
        # 경고 영역 안에 있으면 빨강, 밖이면 녹색
        is_in_zone = not (x2 < wx1 or ...)
        color = (0, 0, 255) if is_in_zone else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 라벨 표시
        text = f"person {score:.2f}"
        cv2.putText(frame, text, (x1, y1-5), ...)
    
    # 3. WARNING 문구 (침입 시)
    if intrusion_detected:
        cv2.putText(frame, "WARNING!", (x, y), ...)
    
    return frame
```

**반투명 효과 원리:**
```python
# 원본 프레임과 색칠한 프레임을 섞기
overlay = frame.copy()
cv2.rectangle(overlay, ..., (0, 0, 255), -1)  # 빨강으로 채우기

cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
#                       ↑        ↑
#                    30%       70%
# 결과 = overlay * 0.3 + frame * 0.7
# → 빨간색이 30%만 비침 (반투명)
```

---

## 🛠️ utils/logger.py - 로깅 시스템

### 역할
프로그램의 모든 동작을 **기록**합니다.

### EventType - 이벤트 분류
```python
class EventType(Enum):
    # 시스템
    SYSTEM_START = "시스템 시작"
    SYSTEM_STOP = "시스템 종료"
    MODULE_INIT = "모듈 초기화"
    MODULE_START = "모듈 시작"
    MODULE_STOP = "모듈 종료"
    
    # 카메라
    CAMERA_OPEN = "카메라 열림"
    CAMERA_CLOSE = "카메라 닫힘"
    CAMERA_ERROR = "카메라 오류"
    
    # 추론
    DETECTION_RESULT = "탐지 결과"
    
    # 오류
    ERROR_OCCURRED = "오류 발생"
```

### 사용 예시
```python
logger = get_logger("main")

# 정보 로그
logger.event_info(
    EventType.SYSTEM_START,
    "프로그램 시작",
    {"version": "1.0"}
)

# 경고 로그
logger.event_warning(
    EventType.CAMERA_ERROR,
    "카메라 재연결 필요"
)

# 오류 로그
logger.event_error(
    EventType.ERROR_OCCURRED,
    "YOLO 모델 로드 실패",
    {"error": str(e)},
    exc_info=True  # 상세 오류 정보 포함
)

# 디버그 로그
logger.debug("프레임 처리", {"fps": 30.5})
```

**로그 파일 구조:**
```
logs/
├── event_project.log   # 중요 이벤트만
└── debug_project.log   # 모든 디버그 정보
```

---

## ⏰ utils/time_utils.py - 시간 유틸리티

### FPSCounter - FPS 계산기
```python
class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
    
    def update(self) -> float:
        """프레임 하나 처리했을 때 호출"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed >= 1.0:  # 1초마다 FPS 계산
            self.fps = self.frame_count / elapsed
            self.frame_count = 0  # 리셋
            self.start_time = time.time()
        
        return self.fps
```

**동작 원리:**
```
시간: 0.0s  0.5s  1.0s  1.5s  2.0s
프레임: △    △    △    △    △
        1     2     3    4     5

1.0s 도달:
- frame_count = 3
- elapsed = 1.0
- fps = 3 / 1.0 = 3.0 FPS
- 리셋

2.0s 도달:
- frame_count = 2 (4, 5번 프레임)
- elapsed = 1.0
- fps = 2 / 1.0 = 2.0 FPS
```

---

## 📦 utils/sensor_sync.py - 센서 동기화

### SensorBuffer - 센서 데이터 버퍼
```python
class SensorBuffer:
    """센서 데이터를 시간순으로 저장"""
    
    def add(self, timestamp: float, data: Any):
        """데이터 추가"""
        self._buffer.append((timestamp, data))
    
    def get_latest(self) -> Tuple[float, Any]:
        """가장 최근 데이터 반환"""
        return self._buffer[-1]
    
    def get_nearest(self, timestamp: float, max_age: float):
        """주어진 시간과 가장 가까운 데이터 반환"""
```

**참고: 센서 동기화를 활용한다면?**
```python
# 현재는 실시간으로 최신 데이터만 사용하지만,
# 만약 특정 시점의 센서 데이터와 정확히 매칭이 필요하다면:

시간:     10.0s   10.1s   10.2s   10.3s
카메라:   Frame1          Frame2
GPS:             GPS1            GPS2

# Frame1(10.0s)과 정확히 매칭할 GPS 데이터는?
frame_ts = 10.0
nearest_gps = buffer.get_nearest(frame_ts, max_age=0.5)
# → GPS1 (10.1s)을 반환 (가장 가까운 시간, 0.1초 차이)

# 하지만 현재 구현에서는 get_latest()로 가장 최신 데이터만 사용합니다.
```

---

## 🔧 system/watchdog.py - 감시 프로그램

### 역할
메인 프로그램이 **오류로 종료**되면 **자동으로 재시작**합니다.

### 동작 원리
```python
class Watchdog:
    def run(self):
        """main.py를 실행하고 감시"""
        retry_count = 0
        
        while retry_count < max_retries:
            # main.py 실행
            process = subprocess.Popen([python, "main.py"])
            returncode = process.wait()  # 종료 대기
            
            if returncode == 0:
                # 정상 종료
                break
            else:
                # 비정상 종료 → 재시작
                print("프로그램 재시작...")
                retry_count += 1
                time.sleep(10)  # 10초 후 재시도
```

**사용법:**
```bash
# 직접 실행 (오류 시 종료)
python main.py

# Watchdog로 실행 (오류 시 자동 재시작)
python system/watchdog.py
```

---

## 💾 system/storage.py - 저장소 관리

### 주요 함수

#### cleanup_old_files() - 용량 관리
```python
def cleanup_old_files(path, threshold_percent=80.0):
    """
    디스크 용량이 80%를 넘으면 오래된 파일 삭제
    
    동작:
    1. 디스크 사용률 확인
    2. 80% 미만이면 종료
    3. 오래된 파일 찾기
    4. 하나씩 삭제
    5. 80% 미만이 될 때까지 반복
    """
```

**사용 시나리오:**
```python
# 매일 자동 실행 (cron)
# 0 3 * * * python -c "from system.storage import cleanup_old_files; cleanup_old_files()"

# 또는 프로그램 시작 시
if get_disk_usage()['percent'] > 80:
    cleanup_old_files()
```

---

## ❌ errors/enums.py - 에러 타입 정의

### 에러 분류
```python
class CameraError(Enum):
    OK = "문제 없음"
    DEVICE_NOT_FOUND = "카메라 없음"
    DEVICE_BUSY = "다른 프로그램이 사용 중"
    PERMISSION_DENIED = "권한 없음"
    BACKEND_ERROR = "백엔드 오류"

class GPSError(Enum):
    OK = "문제 없음"
    DEVICE_NOT_FOUND = "GPS 장치 없음"
    NO_SIGNAL = "GPS 신호 없음"
    INVALID_DATA = "잘못된 데이터"

class IMUError(Enum):
    OK = "문제 없음"
    DEVICE_NOT_FOUND = "IMU 장치 없음"
    BUS_ERROR = "I2C 버스 오류"
```

**사용 예시:**
```python
error = diagnose_camera_error(0)

if error == CameraError.DEVICE_NOT_FOUND:
    print("카메라가 연결되지 않았습니다")
elif error == CameraError.DEVICE_BUSY:
    print("다른 프로그램을 종료하세요")
```

---

## 🔄 전체 실행 흐름 정리

### 1. 초기화 단계
```
main.py 시작
    ↓
YOLO 모델 로드
    ↓
카메라 초기화 (0번, 2번)
    ↓
SharedState 생성 (카메라당 1개)
    ↓
센서 초기화 (GPS, IMU)
    ↓
부저 초기화
```

### 2. 스레드 시작
```
캡처 스레드 (카메라 0) ─┐
캡처 스레드 (카메라 2) ─┤
                        ├→ 동시 실행
추론 스레드 (중앙 1개) ─┤
저장 스레드 (중앙 1개) ─┤
GPS 스레드 ─────────────┤
IMU 스레드 ─────────────┘
```

### 3. 데이터 흐름
```
[카메라] → 캡처 스레드 → SharedState.latest_frame
                        ↓
                    frame_queue
                        ↓
                    추론 스레드 → YOLO → detections
                        ↓
                SharedState.last_detections
                        ↓
                    메인 루프 → 화면 표시
                    
[카메라] → 캡처 스레드 → save_queue → 저장 스레드 → mp4 파일
```

### 4. 이벤트 처리 흐름
```
사람 탐지
    ↓
경고 영역 침입?
    ↓ Yes
침입 플래그 설정 (SharedState.last_intrusion = True)
    ↓
┌───────────────┬───────────────┐
↓               ↓               ↓
부저 ON      저장 시작      화면에 WARNING
```

---

## 💡 핵심 개념 이해하기

### 1. 스레드는 왜 필요한가?
```python
# 동기식 (느림)
while True:
    frame = camera.read()  # 33ms
    result = yolo(frame)   # 100ms
    save(frame)            # 50ms
    show(frame)            # 10ms
    # 총 193ms = 5 FPS

# 비동기식 (빠름)
# 각 작업이 독립적으로 실행
capture: 30 FPS
inference: 10 FPS (INFER_STRIDE=3)
save: 30 FPS
display: 30 FPS
```

### 2. Queue는 왜 사용하나?
```python
# 문제: 생산자-소비자 속도 차이
producer: 30 FPS (빠름)
consumer: 10 FPS (느림)

# Queue 없이:
# → 프레임 손실 또는 메모리 부족

# Queue 사용:
# → 버퍼 역할 (최대 크기 제한)
# → 꽉 차면 오래된 것 버리고 새 것 넣기
```

### 3. Lock은 왜 필요한가?
```python
# 경쟁 조건 (Race Condition)
# 스레드 A와 B가 동시에 counter 증가

counter = 0

# A: tmp = counter (0)
# B: tmp = counter (0)
# A: counter = tmp + 1 (1)
# B: counter = tmp + 1 (1)
# 결과: 2가 되어야 하는데 1!

# Lock 사용:
counter = 0
lock = threading.Lock()

with lock:
    counter += 1  # 한 번에 하나씩만
# 결과: 정확히 2
```

### 4. 왜 이렇게 복잡하게 설계했나?
```
단순 설계 (느림):
- 카메라 읽기 → AI 분석 → 저장 → 화면 표시 → 반복
- 순차 처리 = 5 FPS

현재 설계 (빠름):
- 모든 작업이 동시 진행
- 다중 카메라 지원
- 실시간 30 FPS
- 오류 자동 복구
- 확장 가능
```

---

## 🎓 학습 팁

### 코드를 읽을 때
1. **main.py부터** 시작 → 전체 흐름 파악
2. **한 가지 기능**만 집중 (예: 카메라 캡처)
3. **로그 메시지** 보며 실행 흐름 추적
4. **디버그 모드**로 한 줄씩 실행해보기

### 수정할 때
1. **settings.py** 먼저 확인 → 설정으로 해결 가능한가?
2. **한 파일만** 수정 → 영향 범위 최소화
3. **로그 추가** → 동작 확인
4. **테스트** → 카메라 1개만 연결해서 테스트

### 디버깅할 때
1. **로그 파일** 확인 (`logs/`)
2. **print() 추가** → 값 확인
3. **스레드 이름** 확인 → 어느 스레드에서 오류?
4. **오류 타입** 확인 → CameraError? GPSError?

---

## 📝 자주 묻는 질문 (FAQ)

### Q: 왜 카메라가 안 열리나요?
```python
# 1. 카메라 번호 확인
CAMERA_INDICES = [0, 1]  # 맞나요?

# 2. 다른 프로그램이 사용 중?
# → Zoom, Skype 등 종료

# 3. 권한 문제?
# Linux: sudo chmod 666 /dev/video0

# 4. 백엔드 문제?
# Windows: CAP_DSHOW
# Jetson: CAP_V4L2
```

### Q: FPS가 너무 낮아요
```python
# 1. INFER_STRIDE 증가
INFER_STRIDE = 5  # 3에서 5로

# 2. 이미지 크기 감소
yolo_model = YOLOInference(..., imgsz=320)  # 640에서 320으로

# 3. 가벼운 모델 사용
MODEL_PATH = "yolov8n.pt"  # n = nano (가장 가벼움)
```

### Q: 저장이 안 돼요
```python
# 1. 디렉토리 권한 확인
ls -la SaveVideos/

# 2. 디스크 공간 확인
df -h

# 3. 코덱 확인
codec = "mp4v"  # 또는 "XVID", "H264"
```

### Q: 센서가 안 읽혀요
```python
# GPS
# 1. 포트 확인
ls /dev/ttyTHS*  # Jetson
ls /dev/ttyUSB*  # USB GPS

# 2. 권한 확인
sudo chmod 666 /dev/ttyTHS0

# IMU
# 1. I2C 활성화 확인
ls /dev/i2c*

# 2. 주소 확인
i2cdetect -y -r 1
```

---

## 🚀 마치며

이 프로젝트는:
- **실시간 멀티스레딩**
- **AI 추론**
- **하드웨어 제어**
- **센서 통합**
- **영상 처리**

등 다양한 기술이 결합된 **통합 시스템**입니다.

한번에 모두 이해하려 하지 말고, **한 부분씩** 천천히 학습하세요!

**추천 학습 순서:**
1. main.py의 전체 흐름
2. camera.py로 하드웨어 제어 이해
3. yolo_infer.py의 추론 로직
4. postprocess.py로 결과 처리
5. 스레드 간 통신 (Queue, Lock)
6. 영상 저장 로직 (가장 복잡!)

**행운을 빕니다! 🎉**
