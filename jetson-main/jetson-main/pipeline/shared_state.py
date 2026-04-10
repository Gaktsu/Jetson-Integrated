"""
Shared state used by capture/inference/save pipeline.
"""
from __future__ import annotations

import cv2
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from ai.detector import Detection, WarningLevel

logger = get_logger("pipeline.shared_state")


@dataclass
class CameraStateSnapshot:
    """한 카메라의 렌더링·저장 판단에 필요한 상태 스냅샷 (불변 읽기 전용)."""
    frame: Optional[Any]            # cv2.Mat copy (없으면 None)
    detections: List[Detection]
    intrusion: bool
    warning_level: WarningLevel
    last_intrusion_ts: float
    capture_ms: float
    inference_ms: float
    postprocess_ms: float
    forklift_speed: int


class SharedState:
    """스레드 간 공유 상태"""
    
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        self.latest_frame: Optional[cv2.Mat] = None
        self.latest_frame_seq: int = -1
        self.latest_frame_ts: float = 0.0
        self.last_detections: List[Detection] = []
        self.last_detection_ts: float = 0.0
        self.last_sensor_data: Optional[Dict[str, Any]] = None
        self.last_intrusion: bool = False
        self.last_intrusion_ts: float = 0.0

        self.frame_queue: queue.Queue[Tuple[int, float, cv2.Mat]] = queue.Queue(maxsize=1)
        
        # detection 히스토리 (최대 30개 프레임)
        self.detection_history: Dict[int, List[Detection]] = {}
        self.detection_history_lock = threading.Lock()
        
        # 마지막 유효한 detection (바운딩 박스 유지용)
        self.last_valid_detections: List[Detection] = []
        self.last_valid_detections_lock = threading.Lock()
        
        # smoothed detection (부드러운 전환용)
        self.smoothed_detections: List[Detection] = []
        self.smoothed_detections_lock = threading.Lock()
        
        self.stop_event = threading.Event()

        # ── TTC / 동적 ROI 상태 ──
        # track_history: {track_id: [box_area, ...]} — 최대 15프레임 면적 이력
        # forklift_speed: 0(정지)~5(최고속) — 동적 ROI 팽창량 및 TTC 임계값 조정에 사용
        self.track_history: Dict[int, List[float]] = defaultdict(list)
        self.track_history_lock = threading.Lock()
        self.forklift_speed: int = 0          # inference 스레드에서 기록, main 루프에서 읽기
        self.last_warning_level: WarningLevel = WarningLevel.SAFE  # 최신 경고 레벨

        # ── 성능 측정 (ms, Lock-free 단순 할당) ──
        # 각 스레드에서 작성 / 메인 루프에서 읽기용
        self.capture_ms: float = 0.0      # cap.read_frame() 소요 시간
        self.inference_ms: float = 0.0    # model.run_inference() 소요 시간
        self.postprocess_ms: float = 0.0  # model.postprocess_results() 소요 시간

        # ── 이벤트 업로드 큐 ──
        # inference 스레드가 (event_type, cam_id, speed_level) 튜플을 넣으면
        # uploader 워커 스레드가 꺼내서 서버로 전송 (추론 루프와 업로드 책임 분리)
        self.event_queue: queue.Queue = queue.Queue(maxsize=64)

        logger.debug("SharedState 객체 생성 완료")

    # ──────────────────────────────────────────────
    # 캡처 스레드 인터페이스
    # ──────────────────────────────────────────────

    def put_captured_frame(self, frame, timestamp: float, elapsed_ms: float) -> None:
        """캡처 스레드가 새 프레임을 기록할 때 호출합니다."""
        self.capture_ms = elapsed_ms
        with self.frame_lock:
            self.latest_frame = frame
            self.latest_frame_seq += 1
            self.latest_frame_ts = timestamp

    # ──────────────────────────────────────────────
    # 추론 스레드 인터페이스
    # ──────────────────────────────────────────────

    def update_detection_result(
        self,
        detections: List[Detection],
        warning_level: WarningLevel,
        intrusion: bool,
        timestamp: float,
        sensor_data,
    ) -> None:
        """추론 스레드가 탐지 결과를 기록할 때 호출합니다."""
        with self.detection_lock:
            self.last_detections = detections
            self.last_detection_ts = timestamp
            self.last_sensor_data = sensor_data
            self.last_intrusion = intrusion
            self.last_warning_level = warning_level
            if intrusion:
                self.last_intrusion_ts = timestamp

    def push_smoothed_detections(self, seq: int, smoothed: List[Detection]) -> None:
        """스무딩된 탐지 결과를 히스토리·현재·유효 버퍼에 기록합니다."""
        with self.detection_history_lock:
            self.detection_history[seq] = smoothed
            if len(self.detection_history) > 500:
                del self.detection_history[min(self.detection_history.keys())]

        with self.smoothed_detections_lock:
            self.smoothed_detections = smoothed

        if smoothed:
            with self.last_valid_detections_lock:
                self.last_valid_detections = smoothed

    # ──────────────────────────────────────────────
    # UI / 저장 스레드 인터페이스
    # ──────────────────────────────────────────────

    def snapshot(self) -> CameraStateSnapshot:
        """UI 렌더링·저장 판단용 상태를 스레드 안전하게 일괄 읽습니다."""
        with self.frame_lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        with self.detection_lock:
            return CameraStateSnapshot(
                frame=frame,
                detections=list(self.last_detections),
                intrusion=self.last_intrusion,
                warning_level=self.last_warning_level,
                last_intrusion_ts=self.last_intrusion_ts,
                capture_ms=self.capture_ms,
                inference_ms=self.inference_ms,
                postprocess_ms=self.postprocess_ms,
                forklift_speed=self.forklift_speed,
            )

    def is_intruding(self) -> bool:
        """현재 침입 상태를 스레드 안전하게 반환합니다."""
        with self.detection_lock:
            return self.last_intrusion

