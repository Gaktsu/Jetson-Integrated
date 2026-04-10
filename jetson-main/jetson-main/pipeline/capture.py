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


def _read_frame_with_timing(cap) -> tuple:
    """캡처 장치에서 프레임을 읽고 소요 시간(ms)을 함께 반환합니다."""
    t0 = time.perf_counter()
    read_success, frame = cap.read_frame()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return read_success, frame, elapsed_ms


def _update_shared_state(state: SharedState, frame, timestamp: float, elapsed_ms: float) -> None:
    """캡처된 프레임과 타이밍 정보를 SharedState에 기록합니다."""
    state.put_captured_frame(frame, timestamp, elapsed_ms)


def _enqueue_frame_for_inference(state: SharedState, timestamp: float, frame) -> None:
    """추론 스레드용 프레임 큐에 프레임을 넣습니다."""
    _put_dropping_oldest(state.frame_queue, (state.latest_frame_seq, timestamp, frame))


def _enqueue_frame_for_saving(
    save_queue: Optional[queue.Queue],
    cam_id: int,
    timestamp: float,
    frame,
    seq: int,
) -> None:
    """저장 스레드용 큐에 raw 프레임을 넣습니다."""
    if save_queue is not None:
        _put_dropping_oldest(save_queue, (cam_id, timestamp, frame, seq))


def _log_capture_progress(frame_count: int) -> None:
    """100프레임마다 캡처 진행 상황을 디버그 로그로 출력합니다."""
    if frame_count % 100 == 0:
        logger.debug("프레임 캡처 진행", {"frame_count": frame_count})


def capture_loop(cap, state: SharedState, cam_id: int, save_queue: Optional[queue.Queue] = None) -> None:
    """캡처 스레드"""
    logger.event_info(EventType.MODULE_START, "캡처 루프 시작")

    frame_count = 0
    while not state.stop_event.is_set():
        read_success, frame, elapsed_ms = _read_frame_with_timing(cap)
        if not read_success:
            time.sleep(0.001)
            continue

        timestamp = time.time()
        _update_shared_state(state, frame, timestamp, elapsed_ms)
        frame_count += 1

        _enqueue_frame_for_inference(state, timestamp, frame)
        _enqueue_frame_for_saving(save_queue, cam_id, timestamp, frame, state.latest_frame_seq)
        _log_capture_progress(frame_count)

    logger.event_info(
        EventType.MODULE_STOP,
        "캡처 루프 종료",
        {"total_frames": frame_count}
    )
