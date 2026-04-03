"""
Capture loop pipeline module.
"""
from __future__ import annotations

import cv2
import queue
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from config.settings import CAMERA_FPS, CAMERA_INDICES
from utils.logger import get_logger, EventType
from pipeline.shared_state import SharedState

if TYPE_CHECKING:
    from hardware.camera import CameraCapture


logger = get_logger("pipeline.capture")


def _put_dropping_oldest(q: queue.Queue, item) -> None:
    """큐가 가득 찼을 때 가장 오래된 항목을 버리고 새 항목을 넣는다."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def start_capture_threads(
    cameras: List[CameraCapture],
    states: List[SharedState],
    fps_map: Dict[int, float],
    save_queue: queue.Queue,
) -> List[threading.Thread]:
    """카메라당 캡처 스레드를 시작하고 fps_map을 채운 뒤 스레드 리스트 반환"""
    threads = []
    for i, (camera, state) in enumerate(zip(cameras, states)):
        fps_value = CAMERA_FPS
        if camera.cap is not None:
            reported = camera.cap.get(cv2.CAP_PROP_FPS)
            logger.event_info(EventType.MODULE_INIT, f"카메라 {CAMERA_INDICES[i]} FPS 정보",
                              {"reported_fps": reported, "will_use_fps": fps_value})
        fps_map[CAMERA_INDICES[i]] = fps_value
        t = threading.Thread(
            target=capture_loop,
            args=(camera, state, CAMERA_INDICES[i], save_queue),
            daemon=True, name=f"capture_{i}"
        )
        t.start()
        threads.append(t)
        logger.debug(f"캡처 스레드 시작 (cam={CAMERA_INDICES[i]}, fps={fps_value})")
    return threads


def capture_loop(cap, state: SharedState, cam_id: int, save_queue: Optional[queue.Queue] = None) -> None:
    """캡처 스레드"""
    logger.event_info(
        EventType.MODULE_START,
        "캡처 루프 시작"
    )
    
    frame_count = 0
    while not state.stop_event.is_set():
        t0 = time.perf_counter()
        ok, frame = cap.read_frame()
        capture_elapsed = (time.perf_counter() - t0) * 1000  # ms
        if not ok:
            time.sleep(0.001)
            continue

        # capture_ms: 이번 프레임의 측정값으로 갱신 (Lock 없이 float 단순 할당)
        state.capture_ms = capture_elapsed

        timestamp = time.time()

        with state.frame_lock:
            state.latest_frame = frame
            state.latest_frame_seq += 1
            state.latest_frame_ts = timestamp
            frame_count += 1

        _put_dropping_oldest(state.frame_queue, (state.latest_frame_seq, timestamp, frame))

        # 저장 스레드에 raw frame 직접 전달 (추론 스레드와 독립)
        if save_queue is not None:
            _put_dropping_oldest(save_queue, (cam_id, timestamp, frame, state.latest_frame_seq))

        # 100 프레임마다 디버그 로깅
        if frame_count % 100 == 0:
            logger.debug("프레임 캡처 진행", {"frame_count": frame_count})
    
    logger.event_info(
        EventType.MODULE_STOP,
        "캡처 루프 종료",
        {"total_frames": frame_count}
    )
