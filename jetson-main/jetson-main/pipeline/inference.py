"""
Inference loop pipeline module.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger, EventType
import numpy as np

from ai.detector import (
    Detection, WarningLevel,
    analyze_ttc, check_intrusion_polygon, cleanup_track_history, load_roi_polygon,
)
from ai.model import YOLOInference
from pipeline.shared_state import SharedState
from pipeline.uploader import upload_event_log

logger = get_logger("pipeline.inference")

# 두 카메라 추론 스레드가 동시에 GPU를 사용할 때 발생하는
# CUDA 메모리 경합 / ByteTrack 내부 상태 충돌을 방지하기 위한 Lock.
# 한 번에 한 스레드만 run_inference()를 실행한다.
_gpu_lock = threading.Lock()


def _agg_avg(lst: list) -> float:
    """리스트 평균. 빈 경우 0.0 반환."""
    return sum(lst) / len(lst) if lst else 0.0


def _agg_max(lst: list) -> float:
    """리스트 최댓값. 빈 경우 0.0 반환."""
    return max(lst) if lst else 0.0


def start_inference_thread(
    models: List[YOLOInference],
    states: List[SharedState],
    get_sensor_snapshot,
    stop_event: threading.Event,
) -> List[threading.Thread]:
    """카메라별 독립 추론 스레드 시작. 스레드 리스트 반환."""
    import os
    from config.settings import CAMERA_INDICES
    threads = []
    for idx, (model, state) in enumerate(zip(models, states)):
        cam_id = CAMERA_INDICES[idx] if idx < len(CAMERA_INDICES) else idx
        t = threading.Thread(
            target=_single_cam_inference_loop,
            args=(model, state, cam_id, get_sensor_snapshot, stop_event),
            daemon=True,
            name=f"inference_cam{cam_id}",
        )
        t.start()
        logger.debug(f"추론 스레드 시작 (cam={cam_id})")
        threads.append(t)
    return threads


def _single_cam_inference_loop(
    model: YOLOInference,
    state: SharedState,
    cam_id: int,
    sensor_getter: Optional[Callable[[float], Dict[str, Any]]],
    stop_event: Optional[threading.Event],
) -> None:
    """카메라 1개 전담 추론 루프. 카메라별 스레드에서 실행."""
    import os
    from config.settings import PROJECT_ROOT, ENABLE_TRACKING, DYNAMIC_ROI_PX_PER_SPEED, INFER_STRIDE

    logger.event_info(EventType.MODULE_START, "추론 루프 시작", {"cam": cam_id})

    # ROI 폴리곤 로드 (파일 없으면 매 프레임 재시도)
    roi_path = os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_id}.json")
    roi_polygon: Optional[Any] = load_roi_polygon(roi_path)
    if roi_polygon:
        logger.event_info(EventType.MODULE_INIT, "ROI 폴리곤 로드 완료", {"cam": cam_id})
    else:
        logger.event_info(EventType.MODULE_INIT, "ROI 폴리곤 없음 — 파일 생성 시 자동 로드됨", {"cam": cam_id})

    # 집계 로그 변수 (5초마다 출력)
    _AGG_INTERVAL = 5.0
    _agg_start = time.perf_counter()
    _agg_infer: list = []
    _agg_post:  list = []
    _agg_det:   list = []

    inference_count = 0
    stride_counter = 0

    while not state.stop_event.is_set():
        if stop_event is not None and stop_event.is_set():
            break

        try:
            seq, timestamp, frame = state.frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        # ── INFER_STRIDE: N프레임마다 1회만 추론 ──
        stride_counter = (stride_counter + 1) % INFER_STRIDE
        if stride_counter != 0:
            continue

        # ── GPU 추론 (Lock으로 직렬화 — CUDA 경합 방지) ──
        t0 = time.perf_counter()
        with _gpu_lock:
            results = model.run_inference(frame, tracking=ENABLE_TRACKING)
        state.inference_ms = (time.perf_counter() - t0) * 1000

        # ── CPU 후처리 ──
        t1 = time.perf_counter()
        detections = model.postprocess_results(results)
        state.postprocess_ms = (time.perf_counter() - t1) * 1000

        inference_count += 1
        _agg_infer.append(state.inference_ms)
        _agg_post.append(state.postprocess_ms)
        _agg_det.append(len(detections))

        # 5초마다 집계 로그
        now = time.perf_counter()
        if now - _agg_start >= _AGG_INTERVAL:
            logger.event_info(
                EventType.DATA_PROCESSED,
                f"[집계] cam{cam_id} 추론 성능 (최근 5초)",
                {
                    "infer_avg_ms": round(_agg_avg(_agg_infer), 2),
                    "infer_max_ms": round(_agg_max(_agg_infer), 2),
                    "post_avg_ms":  round(_agg_avg(_agg_post),  2),
                    "det_avg":      round(_agg_avg(_agg_det),   2),
                    "frames":       len(_agg_infer),
                },
            )
            _agg_start = now
            _agg_infer.clear()
            _agg_post.clear()
            _agg_det.clear()

        sensor_data = sensor_getter(timestamp) if sensor_getter else None

        # ROI 지연 로드
        if roi_polygon is None:
            roi_polygon = load_roi_polygon(roi_path)
            if roi_polygon:
                logger.event_info(EventType.MODULE_INIT, "ROI 폴리곤 지연 로드 완료", {"cam": cam_id})

        base_poly = roi_polygon

        # ── 동적 ROI 계산 ──
        dynamic_poly = base_poly
        speed = state.forklift_speed
        if base_poly is not None and speed > 0:
            arr = np.array(base_poly, dtype=np.int32)
            top_idx = np.argsort(arr[:, 1])[:2]
            arr[top_idx, 1] -= speed * DYNAMIC_ROI_PX_PER_SPEED
            arr[top_idx, 1] = np.maximum(arr[top_idx, 1], 0)
            dynamic_poly = arr.tolist()

        # ── TTC 분석 → 경고 레벨 산출 ──
        with state.track_history_lock:
            warning_level = analyze_ttc(detections, dynamic_poly, state.track_history)
            active_ids: List[int] = [
                det["track_id"] for det in detections  # type: ignore[typeddict-item]
                if det.get("track_id") is not None
            ]
            cleanup_track_history(state.track_history, active_ids)

        intrusion = warning_level != WarningLevel.SAFE

        with state.det_lock:
            prev_warning = state.last_warning_level
            state.last_detections = detections
            state.last_detection_ts = timestamp
            state.last_sensor_data = sensor_data
            state.last_intrusion = intrusion
            state.last_warning_level = warning_level
            if intrusion:
                state.last_intrusion_ts = timestamp

        if intrusion and warning_level != prev_warning:
            upload_event_log(
                event_type=warning_level.value,
                cam_id=cam_id,
                speed_level=state.forklift_speed,
            )

        # smoothing + 히스토리
        smoothed = _smooth_detections(state.smoothed_detections, detections)

        with state.detection_history_lock:
            state.detection_history[seq] = smoothed
            if len(state.detection_history) > 500:
                oldest_seq = min(state.detection_history.keys())
                del state.detection_history[oldest_seq]

        with state.smoothed_detections_lock:
            state.smoothed_detections = smoothed

        if smoothed:
            with state.last_valid_detections_lock:
                state.last_valid_detections = smoothed

    logger.event_info(
        EventType.MODULE_STOP,
        "추론 루프 종료",
        {"cam": cam_id, "total_inferences": inference_count},
    )

def _smooth_detections(
    prev_detections: List[Detection],
    curr_detections: List[Detection],
    alpha: float = 0.7
) -> List[Detection]:
    """
    바운딩 박스 smoothing (부드러운 전환)
    
    Args:
        prev_detections: 이전 프레임 detection
        curr_detections: 현재 프레임 detection
        alpha: smoothing 계수 (0.7 = 이전 70%, 현재 30%)
    
    Returns:
        smoothed detections
    """
    if not curr_detections:
        return prev_detections

    if not prev_detections:
        return curr_detections

    smoothed: List[Detection] = []

    for curr in curr_detections:
        cx1, cy1, cx2, cy2 = curr["bbox"]
        curr_cx = (cx1 + cx2) / 2
        curr_cy = (cy1 + cy2) / 2

        min_dist = float('inf')
        closest_prev = None

        for prev in prev_detections:
            px1, py1, px2, py2 = prev["bbox"]
            if curr["class_id"] == prev["class_id"]:
                prev_cx = (px1 + px2) / 2
                prev_cy = (py1 + py2) / 2
                dist = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_prev = prev

        if closest_prev and min_dist < 200:  # 200픽셀 이내
            px1, py1, px2, py2 = closest_prev["bbox"]
            smoothed.append(Detection(
                bbox=(
                    int(alpha * px1 + (1 - alpha) * cx1),
                    int(alpha * py1 + (1 - alpha) * cy1),
                    int(alpha * px2 + (1 - alpha) * cx2),
                    int(alpha * py2 + (1 - alpha) * cy2),
                ),
                confidence=alpha * closest_prev["confidence"] + (1 - alpha) * curr["confidence"],
                class_id=curr["class_id"],
                class_name=curr["class_name"],
            ))
        else:
            smoothed.append(curr)

    return smoothed
