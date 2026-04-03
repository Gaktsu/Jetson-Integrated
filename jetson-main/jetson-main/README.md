# Person Detection System

실시간 사람 탐지 및 경고 시스템

## 프로젝트 구조

```
project/
 ├─ main.py                 # 전체 흐름 제어
 │
 ├─ config/
 │   └─ settings.py         # 상수, 경로, 파라미터
 │
 ├─ errors/
 │   └─ enums.py            # CameraError, GPSError 등
 │
 ├─ hardware/
 │   ├─ camera.py           # USB / CSI 초기화 + 캡처
 │   ├─ gps.py              # NEO-6M 처리
 │   ├─ buzzer.py           # 부저 제어
 │   └─ imu.py              # IMU 센서
 │
 ├─ vision/
 │   ├─ preprocess.py       # resize / normalize
 │   ├─ yolo_infer.py       # YOLO 추론
 │   └─ postprocess.py      # NMS / bbox 변환
 │
 ├─ system/
 │   ├─ autostart.py        # 부팅 자동 실행
 │   ├─ storage.py          # 용량 관리
 │   └─ watchdog.py         # 재시작 / 오류 감지
 │
 ├─ utils/
 │   ├─ logger.py           # 로그
 │   ├─ time_utils.py       # timestamp
 │   └─ sensor_sync.py      # 센서 데이터 동기화
 │
 ├─ logs/                   # 로그 파일 저장 디렉토리
 ├─ SaveVideos/             # 녹화 영상 저장 디렉토리
 └─ models/                 # YOLO 모델 파일
```

## 설치

### 필수 패키지

```bash
pip install ultralytics opencv-python pyserial coloredlogs
```

또는 requirements.txt 사용:

```bash
pip install -r requirements.txt
```

### 하드웨어 요구사항

- USB 카메라 또는 CSI 카메라 (1개 이상)
- GPS 모듈 (선택사항): NEO-6M 또는 호환 모듈
- IMU 센서 (선택사항)
- 부저 (선택사항): GPIO 제어 가능한 부저

## 실행

### 기본 실행

```bash
cd jetson
python main.py
```

### Watchdog로 실행 (자동 재시작)

프로그램 오류 시 자동으로 재시작합니다:

```bash
python system/watchdog.py
```

### 실행 시 확인사항

1. 카메라가 올바르게 연결되어 있는지 확인
2. YOLO 모델 파일이 `models/` 디렉토리에 있는지 확인
3. GPS/IMU 센서가 연결되어 있는지 확인 (선택사항)
4. `SaveVideos/` 디렉토리에 쓰기 권한이 있는지 확인

## 설정

`config/settings.py`에서 다음 항목을 수정할 수 있습니다:

### 카메라 설정
- `CAMERA_INDICES`: 사용 가능한 카메라 인덱스 리스트 (예: [0, 1])
- `CAMERA_MAX_RETRIES`: 카메라 초기화 재시도 횟수
- `CAMERA_RETRY_DELAY`: 재시도 간 대기 시간 (초)

### YOLO 모델 설정
- `MODEL_PATH`: YOLO 모델 경로
- `CONFIDENCE_THRESHOLD`: 신뢰도 임계값
- `INFER_STRIDE`: 추론 간격 (N프레임마다)
- `WARNING_ZONE_RATIO`: 경고 영역 비율

### 영상 저장 설정
- `RECORDING_MODE`: 녹화 모드 (`"event"`: 이벤트 발생 시만, `"full"`: 상시 녹화)
- `EVENT_RECORD_BUFFER_SEC`: 이벤트 발생 전 저장할 버퍼 시간 (기본: 5초)
- `EVENT_RECORD_POST_SEC`: 이벤트 종료 후 저장할 시간 (기본: 5초)
- `SAVE_DIR`: 영상 저장 디렉토리 경로

## 사용법

### 키보드 단축키

- **Q**: 프로그램 종료
- **C**: 카메라 전환 (여러 카메라가 연결된 경우)

### 다중 카메라 설정

`config/settings.py`에서 카메라 인덱스를 설정하세요:

```python
CAMERA_INDICES = [0, 1]  # 카메라 0번과 1번 사용
```

프로그램 실행 시 기본적으로 0번 카메라가 표시되며, `C` 키를 눌러 다른 카메라로 전환할 수 있습니다.
모든 카메라는 백그라운드에서 동시에 촬영되며, 화면에는 선택된 하나의 카메라만 표시됩니다.

**중요**: 어느 한 카메라에서라도 침입이 감지되면 **모든 카메라가 동시에 녹화**됩니다. 이를 통해 다각도 영상 증거를 확보할 수 있습니다.

### 영상 저장 동작 방식

#### 이벤트 기반 녹화 (`RECORDING_MODE = "event"`)
1. 침입 감지 시 모든 카메라 녹화 시작
2. 이벤트 발생 **5초 전** 프레임부터 저장 (사전 버퍼링)
3. 침입이 계속되는 동안 녹화 지속
4. 마지막 침입 종료 후 **5초 동안** 추가 녹화
5. 5초 이내 재침입 발생 시 녹화 자동 연장 (마지막 침입 + 5초)
6. 저장 파일명: `camera{id}_{timestamp}_{gps좌표}.mp4`

#### 전체 녹화 (`RECORDING_MODE = "full"`)
- 프로그램 실행 중 모든 카메라 상시 녹화

## 지원 플랫폼

- **Windows**: DirectShow (CAP_DSHOW) 백엔드 사용
- **Jetson (NVIDIA)**: Video4Linux2 (CAP_V4L2) 백엔드 자동 사용
- **Linux (일반)**: Video4Linux2 (CAP_V4L2) 백엔드 사용

카메라 백엔드는 운영체제를 자동으로 감지하여 설정됩니다.

## 기능

### 핵심 기능
- ✅ 실시간 사람 탐지 (YOLO)
- ✅ 경고 영역 침입 감지
- ✅ FPS, 시간, 탐지 수 표시
- ✅ 다중 카메라 지원 및 실시간 전환
- ✅ 모든 카메라 백그라운드 동시 촬영
- ✅ OS별 카메라 백엔드 자동 감지 (Windows/Jetson/Linux)
- ✅ 카메라 오류 진단 및 자동 재시도

### 경고 시스템
- ✅ 부저 경고 (침입 감지 시 자동 알림)
- ✅ GPIO 기반 부저 제어

### 센서 통합
- ✅ GPS 연동 (NEO-6M)
- ✅ IMU 센서 연동
- ✅ 센서 데이터 동기화 및 버퍼링
- ✅ NMEA 프로토콜 파싱 (GPRMC, GPGGA, GNRMC, GNGGA)

### 영상 저장
- ✅ 이벤트 기반 녹화 (침입 감지 시 자동 녹화)
- ✅ 전체 녹화 모드 (상시 녹화)
- ✅ 5초 사전 버퍼링 (이벤트 발생 전 5초 포함)
- ✅ 5초 후속 녹화 (이벤트 종료 후 5초 추가)
- ✅ 이벤트 재발생 시 녹화 자동 연장
- ✅ **다중 카메라 동시 녹화** (한 카메라에서 이벤트 발생 시 모든 카메라 동시 저장)
- ✅ GPS 좌표 기반 파일명 자동 생성
- ✅ 타임스탬프 기반 영상 파일 관리

### 로깅 시스템
- ✅ 구조화된 로깅 (JSON 형식)
- ✅ 이벤트 타입별 로그 분류
- ✅ 콘솔 및 파일 로그 동시 출력

## 로그 확인

로그 파일은 `logs/` 디렉토리에 저장됩니다:

```bash
logs/
 ├─ app.log              # 일반 로그
 ├─ app.error.log        # 오류 로그
 └─ YYYY-MM-DD.log       # 날짜별 로그 (선택사항)
```

로그 레벨:
- `DEBUG`: 상세한 디버깅 정보
- `INFO`: 일반 정보 메시지
- `WARNING`: 경고 메시지
- `ERROR`: 오류 메시지
- `CRITICAL`: 치명적인 오류

## 문제 해결 (Troubleshooting)

### 카메라 연결 실패

```
오류: 카메라 초기화 실패
```

**해결방법:**
1. 카메라가 물리적으로 올바르게 연결되어 있는지 확인
2. 다른 프로그램에서 카메라를 사용 중인지 확인
3. `CAMERA_INDICES` 설정이 올바른지 확인
4. USB 포트를 변경해보거나 카메라 재연결

### YOLO 모델 로드 실패

```
오류: YOLO 모델 로드 실패
```

**해결방법:**
1. `MODEL_PATH`가 올바른지 확인
2. 모델 파일이 `models/` 디렉토리에 있는지 확인
3. 모델 파일이 손상되지 않았는지 확인

### GPS/IMU 센서 연결 실패

```
경고: GPS 초기화 실패 / IMU 초기화 실패
```

**해결방법:**
1. 센서가 올바른 포트에 연결되어 있는지 확인
2. 시리얼 포트 권한 확인 (Linux: `/dev/ttyUSB0` 등)
3. 센서는 선택사항이므로 시스템은 센서 없이도 작동 가능

### 영상 저장 실패

```
오류: 영상 저장기 생성 실패
```

**해결방법:**
1. `SaveVideos/` 디렉토리가 존재하는지 확인
2. 디렉토리 쓰기 권한 확인
3. 디스크 공간이 충분한지 확인
4. 코덱이 시스템에서 지원되는지 확인

### 낮은 FPS

**해결방법:**
1. `INFER_STRIDE` 값을 증가 (예: 3 → 5)
2. YOLO 모델을 더 가벼운 버전으로 변경 (yolov8n 권장)
3. `imgsz` 값을 낮춤 (예: 640 → 320)
4. 카메라 해상도 조정

## 성능 최적화

### 권장 설정

- **단일 카메라**: `INFER_STRIDE = 1-2`, `imgsz = 640`
- **다중 카메라 (2개)**: `INFER_STRIDE = 3-5`, `imgsz = 320`
- **저사양 PC**: YOLO 모델 `yolov8n.pt` 사용, `imgsz = 320`

### 멀티스레딩 구조

- **캡처 스레드**: 각 카메라마다 1개
- **추론 스레드**: 중앙 1개 (모든 카메라 공유)
- **저장 스레드**: 1개 (모든 카메라 공유)
- **센서 스레드**: GPS 1개, IMU 1개


## Jetson Orin Nano 이벤트 녹화 프레임 누락 대응

YOLO 추론으로 루프 FPS가 흔들릴 때 `pre/post` 길이가 짧아지는 문제를 줄이기 위해,
캡처/추론/저장을 분리한 레퍼런스 코드를 추가했습니다.

- 참고 파일: `vision/async_event_recorder.py`
- 핵심 아이디어:
  - 캡처 스레드는 절대 블로킹하지 않고 큐가 가득 차면 가장 오래된 프레임만 폐기
  - 추론 스레드는 최신 프레임만 사용해 지연 누적 방지
  - 저장 스레드는 타임스탬프 기반 pre/post 윈도우를 유지해 시간 길이 정확도 확보

실서비스에 연결할 때는 `detect_fn` 자리에 YOLO 추론 함수를 주입하세요.
