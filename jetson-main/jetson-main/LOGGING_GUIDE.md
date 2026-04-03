# 구조화된 로깅 시스템 가이드

## 개요
이 프로젝트는 중앙 집중식 구조화 로깅 시스템을 사용합니다.
- 모든 로그는 `StructuredLogger` 클래스를 통해 관리됩니다
- 이벤트 로그와 디버그 로그가 분리되어 저장됩니다
- 모든 로그에는 `module_name`, `event_type`, `timestamp`가 포함됩니다

## 로그 구조

### 로그 레벨
- **INFO**: 일반적인 정보성 이벤트
- **WARNING**: 경고가 필요한 이벤트 (시스템은 계속 동작)
- **ERROR**: 오류 발생 이벤트

### 로그 유형
1. **이벤트 로그** (`logs/event_*.log`)
   - 시스템의 주요 이벤트 기록
   - 콘솔과 파일에 모두 출력
   - 상태 전이, 모듈 시작/종료, 데이터 수신, 오류 발생

2. **디버그 로그** (`logs/debug_*.log`)
   - 내부 변수, 상세 흐름 기록
   - 파일에만 저장 (콘솔 출력 없음)
   - 디버깅 및 문제 추적용

## 사용 방법

### 1. 로거 가져오기
```python
from utils.logger import get_logger, EventType

# 모듈별 로거 생성
logger = get_logger("module_name")
```

### 2. 이벤트 로그 기록
```python
# INFO 레벨 이벤트
logger.event_info(
    EventType.MODULE_START,
    "모듈 시작",
    {"config": "value"}  # 선택적 데이터
)

# WARNING 레벨 이벤트
logger.event_warning(
    EventType.RETRY_ATTEMPT,
    "재시도 중",
    {"retry_count": 3}
)

# ERROR 레벨 이벤트
logger.event_error(
    EventType.ERROR_OCCURRED,
    "오류 발생",
    {"error": str(e)},
    exc_info=True  # 스택 트레이스 포함
)
```

### 3. 디버그 로그 기록
```python
# 내부 변수 추적
logger.debug(
    "프레임 처리 중",
    {"frame_count": 100, "fps": 30.5}
)
```

### 4. 간단한 로그 (이벤트 타입 없이)
```python
logger.info("간단한 정보 메시지")
logger.warning("경고 메시지")
logger.error("오류 메시지", exc_info=True)
```

## 이벤트 타입

### 시스템 라이프사이클
- `SYSTEM_START`: 시스템 전체 시작
- `SYSTEM_STOP`: 시스템 전체 종료
- `MODULE_INIT`: 모듈 초기화
- `MODULE_START`: 모듈 실행 시작
- `MODULE_STOP`: 모듈 실행 종료

### 상태 전이
- `STATE_CHANGE`: 상태 변경

### 데이터 처리
- `DATA_RECEIVED`: 데이터 수신
- `DATA_PROCESSED`: 데이터 처리 완료

### 하드웨어
- `CAMERA_OPEN`: 카메라 열기
- `CAMERA_CLOSE`: 카메라 닫기
- `CAMERA_ERROR`: 카메라 오류
- `FRAME_CAPTURE`: 프레임 캡처

### 추론
- `INFERENCE_START`: 추론 시작
- `INFERENCE_COMPLETE`: 추론 완료
- `DETECTION_RESULT`: 탐지 결과

### 오류 처리
- `ERROR_OCCURRED`: 오류 발생
- `RETRY_ATTEMPT`: 재시도

### 사용자 입력
- `USER_INPUT`: 사용자 입력

### 기타
- `OTHER`: 기타 이벤트

## 로그 파일 구조

```
project/
└── logs/
    ├── event_project.log          # main orchestrator 이벤트 로그
    ├── debug_project.log          # main orchestrator 디버그 로그
    ├── event_project.log          # hardware.camera 이벤트 로그
    ├── debug_project.log          # hardware.camera 디버그 로그
    ├── event_project.log          # vision.yolo_infer 이벤트 로그
    ├── debug_project.log          # vision.yolo_infer 디버그 로그
    ├── event_project.log          # system.watchdog 이벤트 로그
    └── debug_project.log          # system.watchdog 디버그 로그
```

## 로그 포맷

### 이벤트 로그 포맷
```
2026-02-06 10:30:45 - INFO - {"module": "main_orchestrator", "event": "SYSTEM_START", "timestamp": "2026-02-06T10:30:45.123456", "message": "Person Detection System 시작"}
```

### 디버그 로그 포맷
```
2026-02-06 10:30:45 - DEBUG - {"module": "hardware.camera", "timestamp": "2026-02-06T10:30:45.123456", "message": "프레임 처리", "data": {"frame_count": 100}}
```

## 모범 사례

### 언제 이벤트 로그를 사용하나요?
- ✅ 모듈 시작/종료
- ✅ 상태 전이
- ✅ 중요한 데이터 수신
- ✅ 오류 발생
- ✅ 사용자 입력
- ✅ 시스템의 주요 동작

### 언제 디버그 로그를 사용하나요?
- ✅ 내부 변수 값 추적
- ✅ 루프 내부의 상세한 진행 상황
- ✅ 성능 메트릭 (FPS, 처리 시간 등)
- ✅ 개발 및 디버깅 목적

### 주의사항
- ❌ 반복되는 루프 내에서 이벤트 로그를 과도하게 사용하지 마세요
- ❌ 민감한 정보를 로그에 기록하지 마세요
- ✅ 구조화된 데이터를 `data` 파라미터로 전달하세요
- ✅ 적절한 이벤트 타입을 선택하세요

## 예제

### main.py (Orchestrator)
```python
from utils.logger import get_logger, EventType

logger = get_logger("main_orchestrator")

# 시스템 시작
logger.event_info(EventType.SYSTEM_START, "시스템 시작")

# 모듈 초기화
logger.event_info(
    EventType.MODULE_INIT,
    "YOLO 모델 로드 중",
    {"model_path": MODEL_PATH}
)

# 디버그 로그
logger.debug("공유 상태 객체 생성 완료")

# 오류 처리
try:
    # ... 코드 ...
except Exception as e:
    logger.event_error(
        EventType.ERROR_OCCURRED,
        "메인 루프 오류",
        {"error": str(e)},
        exc_info=True
    )
```

### hardware/camera.py
```python
from utils.logger import get_logger, EventType

logger = get_logger("hardware.camera")

# 카메라 열기
logger.event_info(
    EventType.CAMERA_OPEN,
    "카메라 연결 성공",
    {"camera_index": camera_index}
)

# 재시도
logger.event_warning(
    EventType.RETRY_ATTEMPT,
    "카메라 연결 재시도",
    {"retry_count": retry_count, "max_retries": max_retries}
)

# 디버그 로그
logger.debug("카메라 객체 생성", {"buffer_size": 1})
```

### vision/yolo_infer.py
```python
from utils.logger import get_logger, EventType

logger = get_logger("vision.yolo_infer")

# 추론 시작
logger.event_info(
    EventType.MODULE_START,
    "추론 루프 시작",
    {"infer_stride": infer_stride}
)

# 탐지 결과
logger.event_info(
    EventType.DETECTION_RESULT,
    "객체 탐지 완료",
    {"num_detections": len(detections)}
)

# 디버그 로그
logger.debug("추론 시작", {"frame_seq": last_processed_seq})
```

## 로그 분석

### 이벤트 로그 분석 (중요 이벤트 추적)
```bash
# 시스템 시작/종료 이벤트 확인
grep "SYSTEM_START\|SYSTEM_STOP" logs/event_*.log

# 오류 발생 확인
grep "ERROR_OCCURRED" logs/event_*.log

# 특정 모듈의 이벤트 확인
grep "hardware.camera" logs/event_*.log
```

### 디버그 로그 분석 (상세 디버깅)
```bash
# 특정 모듈의 디버그 로그
cat logs/debug_project.log | grep "hardware.camera"

# FPS 추적
grep "fps" logs/debug_*.log
```
