"""
Inference loop pipeline module.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger, EventType
from ai.detector import Detection, check_intrusion_polygon, load_roi_polygon
from ai.model import YOLOInference
from pipeline.shared_state import SharedState

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
) -> threading.Thread:
    """전체 카메라 공용 추론 스레드 시작"""
    t = threading.Thread(
        target=inference_loop,
        args=(model, states, get_sensor_snapshot, stop_event),
        daemon=True, name="inference_central"
    )
    t.start()
    logger.debug("추론 스레드 시작")
    return t


def inference_loop(
    model: YOLOInference,
    states: List[SharedState],
    sensor_getter: Optional[Callable[[float], Dict[str, Any]]] = None,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """중앙 추론 스레드"""
    logger.event_info(
        EventType.MODULE_START,
        "추론 루프 시작",
        {"num_cameras": len(states)}
    )

    import os
    from config.settings import PROJECT_ROOT, CAMERA_INDICES

    inference_count = 0
    loop_count = 0
    empty_count = 0

    # 카메라별 ROI 폴리곤 로드 (roi_config_cam{N}.json)
    roi_polygons: Dict[int, Any] = {}
    for cam_idx in CAMERA_INDICES:
        path = os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_idx}.json")
        poly = load_roi_polygon(path)
        roi_polygons[cam_idx] = poly
        if poly:
            logger.event_info(EventType.MODULE_INIT, f"ROI 폴리곤 로드 완료", {"cam": cam_idx, "path": path})
        else:
            logger.event_info(EventType.MODULE_INIT, f"ROI 폴리곤 없음 — 침입 판별 비활성", {"cam": cam_idx})

    # 집계 로그용 누적 변수 (5초마다 평균/최대 출력)
    _AGG_INTERVAL = 5.0
    _agg_start = time.perf_counter()
    _agg_infer: list = []
    _agg_post:  list = []
    _agg_det:   list = []

    while not all(state.stop_event.is_set() for state in states):
        if stop_event is not None and stop_event.is_set():
            break

        did_work = False
        loop_count += 1
        
        # 1000번마다 루프 상태 로깅
        if loop_count % 1000 == 0:
            logger.event_info(
                EventType.DATA_PROCESSED,
                "inference_loop 진행 중",
                {"loop_count": loop_count, "inference_count": inference_count, "empty_count": empty_count}
            )

        for idx, state in enumerate(states):
            if stop_event is not None and stop_event.is_set():
                break

            try:
                seq, timestamp, frame = state.frame_queue.get_nowait()
            except queue.Empty:
                empty_count += 1
                if empty_count % 1000 == 0:
                    logger.debug(
                        "frame_queue가 비어있음",
                        {"camera_index": idx, "empty_count": empty_count}
                    )
                continue

            did_work = True
            logger.debug("추론 시작", {"camera_index": idx, "frame_seq": seq, "ts": timestamp})

            # ── GPU 추론 시간 단독 측정 ──
            t0 = time.perf_counter()
            results = model.run_inference(frame)
            state.inference_ms = (time.perf_counter() - t0) * 1000

            # ── CPU 후처리 시간 단독 측정 ──
            t1 = time.perf_counter()
            detections = model.postprocess_results(results)
            state.postprocess_ms = (time.perf_counter() - t1) * 1000

            inference_count += 1

            # ── 집계 누적 ──
            _agg_infer.append(state.inference_ms)
            _agg_post.append(state.postprocess_ms)
            _agg_det.append(len(detections))

            # 5초마다 평균/최대 로그 출력
            now = time.perf_counter()
            if now - _agg_start >= _AGG_INTERVAL:
                logger.event_info(
                    EventType.DATA_PROCESSED,
                    "[집계] 추론 성능 요약 (최근 5초)",
                    {
                        "infer_avg_ms":  round(_agg_avg(_agg_infer), 2),
                        "infer_max_ms":  round(_agg_max(_agg_infer), 2),
                        "post_avg_ms":   round(_agg_avg(_agg_post), 2),
                        "post_max_ms":   round(_agg_max(_agg_post), 2),
                        "det_avg":        round(_agg_avg(_agg_det), 2),
                        "det_max":        int(_agg_max(_agg_det)),
                        "frames":         len(_agg_infer),
                    },
                )
                _agg_start = now
                _agg_infer.clear()
                _agg_post.clear()
                _agg_det.clear()

            sensor_data = sensor_getter(timestamp) if sensor_getter else None

            # 해당 카메라의 ROI 폴리곤으로 foot-point 기반 침입 판별
            cam_id = CAMERA_INDICES[idx] if idx < len(CAMERA_INDICES) else idx
            intrusion = check_intrusion_polygon(detections, roi_polygons.get(cam_id))

            with state.det_lock:
                state.last_detections = detections
                state.last_detection_ts = timestamp
                state.last_sensor_data = sensor_data
                state.last_intrusion = intrusion
                if intrusion:
                    state.last_intrusion_ts = timestamp
            
            # detection 히스토리에 저장 전에 smoothing 적용
            smoothed = _smooth_detections(state.smoothed_detections, detections)
            
            with state.detection_history_lock:
                state.detection_history[seq] = smoothed
                # 히스토리 크기 제한 (500개 = 30fps 기준 약 16초)
                if len(state.detection_history) > 500:
                    oldest_seq = min(state.detection_history.keys())
                    del state.detection_history[oldest_seq]
            
            # smoothed detection 업데이트
            with state.smoothed_detections_lock:
                state.smoothed_detections = smoothed
            
            # 마지막 유효한 detection 업데이트 (비어있지 않으면)
            if smoothed:
                with state.last_valid_detections_lock:
                    state.last_valid_detections = smoothed

            # 저장은 capture_loop → save_queue 경로로 직접 처리됨

        if not did_work:
            time.sleep(0.001)

    logger.event_info(
        EventType.MODULE_STOP,
        "추론 루프 종료",
        {"total_inferences": inference_count}
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
