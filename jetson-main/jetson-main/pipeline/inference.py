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


def _agg_avg(lst: list) -> float:
    """리스트 평균. 빈 경우 0.0 반환."""
    return sum(lst) / len(lst) if lst else 0.0


def _agg_max(lst: list) -> float:
    """리스트 최댓값. 빈 경우 0.0 반환."""
    return max(lst) if lst else 0.0


def start_inference_thread(
    model: YOLOInference,
    states: List[SharedState],
    get_sensor_snapshot,
    stop_event: threading.Event,
) -> List[threading.Thread]:
    """단일 추론 스레드 시작. 모든 카메라를 1개 스레드에서 순차 처리."""
    from config.settings import CAMERA_INDICES
    cam_ids = list(CAMERA_INDICES[:len(states)])
    t = threading.Thread(
        target=inference_loop,
        args=(model, states, cam_ids, get_sensor_snapshot, stop_event),
        daemon=True,
        name="inference",
    )
    t.start()
    logger.debug(f"추론 스레드 시작 (cameras={cam_ids})")
    return [t]


def inference_loop(
    model: YOLOInference,
    states: List[SharedState],
    cam_ids: List[int],
    sensor_getter: Optional[Callable[[float], Dict[str, Any]]],
    stop_event: Optional[threading.Event],
) -> None:
    """단일 추론 루프 — 모든 카메라를 순서대로 처리 (GPU 직렬화 보장)."""
    import os
    from config.settings import PROJECT_ROOT, ENABLE_TRACKING, DYNAMIC_ROI_PX_PER_SPEED, INFER_STRIDE

    n = len(states)
    roi_paths = [
        os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_id}.json")
        for cam_id in cam_ids
    ]
    roi_polygons: List[Optional[Any]] = []
    for idx, cam_id in enumerate(cam_ids):
        poly = load_roi_polygon(roi_paths[idx])
        roi_polygons.append(poly)
        if poly:
            logger.event_info(EventType.MODULE_INIT, "ROI 폴리곤 로드 완료", {"cam": cam_id})
        else:
            logger.event_info(EventType.MODULE_INIT, "ROI 폴리곤 없음 — 파일 생성 시 자동 로드됨", {"cam": cam_id})

    _AGG_INTERVAL = 5.0
    agg_starts   = [time.perf_counter()] * n
    agg_infer:  List[list] = [[] for _ in range(n)]
    agg_post:   List[list] = [[] for _ in range(n)]
    agg_det:    List[list] = [[] for _ in range(n)]
    inference_counts = [0] * n
    stride_counters  = [0] * n

    logger.event_info(EventType.MODULE_START, "추론 루프 시작", {"cameras": cam_ids})

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        if all(s.stop_event.is_set() for s in states):
            break

        processed_any = False

        for idx, (state, cam_id) in enumerate(zip(states, cam_ids)):
            if state.stop_event.is_set():
                continue

            try:
                seq, timestamp, frame = state.frame_queue.get_nowait()
            except queue.Empty:
                continue

            processed_any = True

            # ── INFER_STRIDE ──
            stride_counters[idx] = (stride_counters[idx] + 1) % INFER_STRIDE
            if stride_counters[idx] != 0:
                continue

            # ── GPU 추론 (단일 스레드이므로 직렬화 보장) ──
            t0 = time.perf_counter()
            results = model.run_inference(frame, tracking=ENABLE_TRACKING)
            state.inference_ms = (time.perf_counter() - t0) * 1000

            # ── CPU 후처리 ──
            t1 = time.perf_counter()
            detections = model.postprocess_results(results)
            state.postprocess_ms = (time.perf_counter() - t1) * 1000

            inference_counts[idx] += 1
            agg_infer[idx].append(state.inference_ms)
            agg_post[idx].append(state.postprocess_ms)
            agg_det[idx].append(len(detections))

            # 5초마다 집계 로그
            now = time.perf_counter()
            if now - agg_starts[idx] >= _AGG_INTERVAL:
                logger.event_info(
                    EventType.DATA_PROCESSED,
                    f"[집계] cam{cam_id} 추론 성능 (최근 5초)",
                    {
                        "infer_avg_ms": round(_agg_avg(agg_infer[idx]), 2),
                        "infer_max_ms": round(_agg_max(agg_infer[idx]), 2),
                        "post_avg_ms":  round(_agg_avg(agg_post[idx]),  2),
                        "det_avg":      round(_agg_avg(agg_det[idx]),   2),
                        "frames":       len(agg_infer[idx]),
                    },
                )
                agg_starts[idx] = now
                agg_infer[idx].clear()
                agg_post[idx].clear()
                agg_det[idx].clear()

            sensor_data = sensor_getter(timestamp) if sensor_getter else None

            # ROI 지연 로드
            if roi_polygons[idx] is None:
                roi_polygons[idx] = load_roi_polygon(roi_paths[idx])
                if roi_polygons[idx]:
                    logger.event_info(EventType.MODULE_INIT, "ROI 폴리곤 지연 로드 완료", {"cam": cam_id})

            base_poly = roi_polygons[idx]

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

        if not processed_any:
            time.sleep(0.001)

    logger.event_info(
        EventType.MODULE_STOP,
        "추론 루프 종료",
        {"total_inferences": inference_counts},
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
