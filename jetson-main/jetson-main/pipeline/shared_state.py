"""
Shared state used by capture/inference/save pipeline.
"""
from __future__ import annotations

import cv2
import queue
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from ai.detector import Detection

logger = get_logger("pipeline.shared_state")

class SharedState:
    """스레드 간 공유 상태"""
    
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.det_lock = threading.Lock()
        
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
        self.forklift_speed: int = 0  # inference 스레드에서 기록, main 루프에서 읽기

        # ── 성능 측정 (ms, Lock-free 단순 할당) ──
        # 각 스레드에서 작성 / 메인 루프에서 읽기용
        self.capture_ms: float = 0.0      # cap.read_frame() 소요 시간
        self.inference_ms: float = 0.0    # model.run_inference() 소요 시간
        self.postprocess_ms: float = 0.0  # model.postprocess_results() 소요 시간

        logger.debug("SharedState 객체 생성 완료")

