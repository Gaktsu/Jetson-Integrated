"""
Recorder loop pipeline module.
"""
from __future__ import annotations

import cv2
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from config.settings import EVENT_RECORD_BUFFER_SEC, EVENT_RECORD_POST_SEC, MAX_EVENT_FOLDERS, MAX_FULL_FOLDERS, RECORDING_MODE, SAVE_DIR
from utils.logger import get_logger, EventType
from pipeline.shared_state import SharedState
from pipeline.recorder_utils import (
    _cleanup_old_folders,
    _create_event_folder,
    _create_writer,
    _transcode_and_upload,
)

logger = get_logger("pipeline.recorder")

def start_save_thread(
    save_queue: queue.Queue,
    save_stop_event: threading.Event,
    fps_map: Dict[int, float],
    get_sensor_snapshot,
    state_map: Dict[int, SharedState],
) -> threading.Thread:
    """영상 저장 스레드 시작"""
    t = threading.Thread(
        target=save_loop,
        args=(
            save_queue, save_stop_event, SAVE_DIR, fps_map,
            "mp4v", get_sensor_snapshot, state_map,
            RECORDING_MODE, EVENT_RECORD_BUFFER_SEC, EVENT_RECORD_POST_SEC,
        ),
        daemon=True, name="save_worker"
    )
    t.start()
    logger.debug("저장 스레드 시작")
    return t

def save_loop(
    save_queue: queue.Queue,
    stop_event: threading.Event,
    save_dir: str,
    fps_map: dict[int, float],
    codec: str = "mp4v",
    sensor_getter: Optional[Callable[[float], Dict[str, Any]]] = None,
    state_map: Optional[Dict[int, SharedState]] = None,
    recording_mode: str = "event",
    buffer_seconds: float = 5.0,
    post_seconds: float = 5.0,
) -> None:
    """영상 저장 스레드"""
    logger.event_info(
        EventType.MODULE_START,
        "저장 루프 시작",
        {"queue_maxsize": save_queue.maxsize, "codec": codec}
    )

    os.makedirs(save_dir, exist_ok=True)

    if recording_mode not in {"event", "full"}:
        recording_mode = "event"
    if recording_mode == "event" and not state_map:
        recording_mode = "full"
    
    # full 모드일 때 전용 폴더 생성
    full_folder = None
    if recording_mode == "full":
        full_folder = os.path.join(save_dir, "full_recording")
        os.makedirs(full_folder, exist_ok=True)
        logger.event_info(
            EventType.MODULE_START,
            "Full 녹화 폴더 생성",
            {"path": full_folder}
        )

    writers: dict[int, cv2.VideoWriter] = {}
    writer_paths: dict[int, str] = {}
    buffers: dict[int, Deque[Tuple[float, cv2.Mat]]] = {}
    recording_active: dict[int, bool] = {}
    # post_deadline: inference timestamp 가 아닌 wall clock(time.time()) 기준
    # YOLO FPS, save_queue 지연, flush 시간 등에 영향받지 않아 항상 정확히 5초 보장
    post_deadline: dict[int, Optional[float]] = {}
    # 이전 프레임을 보관해 다음 프레임 도착 시 간격 계산 후 반복 write
    pending_frames: dict[int, Optional[Tuple[float, cv2.Mat]]] = {}
    # 누산기: round() 오차 누적 방지 - 소수 부분을 다음 프레임으로 이월
    frame_oweds: dict[int, float] = {}
    
    # 이벤트별 폴더 관리
    current_event_folder: Optional[str] = None
    event_start_ts: Optional[float] = None
    full_event_folder: Optional[str] = None  # full 모드용 이벤트 폴더
    transcode_threads: List[threading.Thread] = []  # 종료 시 join() 대기용
    shutdown_files: List[str] = []  # q 종료 시 순차 변환/업로드 대상
    shutdown_deadline: Optional[float] = None
    
    frame_process_count = 0  # 디버깅용

    try:
        while True:
            if stop_event.is_set():
                if shutdown_deadline is None:
                    shutdown_deadline = time.time() + post_seconds
                    logger.event_info(
                        EventType.MODULE_START,
                        "q 종료: 고정 post 녹화 시작",
                        {"shutdown_deadline": shutdown_deadline, "post_seconds": post_seconds}
                    )
                has_active_recording = any(
                    recording_active.get(cid)
                    for cid in list(recording_active.keys())
                )
                if not has_active_recording:
                    break
            try:
                cam_id, timestamp, frame, frame_seq = save_queue.get_nowait()  # seq 추가
            except queue.Empty:
                time.sleep(0.001)
                continue

            if frame is None:
                continue
            
            frame_process_count += 1
            
            # 100 프레임마다 진행 상황 로깅
            if frame_process_count % 100 == 0:
                logger.event_info(
                    EventType.DATA_PROCESSED,
                    "save_loop 프레임 처리 진행 중",
                    {"processed_frames": frame_process_count, "cam_id": cam_id}
                )

            sensor_data = None
            intrusion_active = False

            # 모든 카메라 중 하나라도 침입 중이거나 post-buffer 구간이면 True
            if state_map:
                now = time.time()
                for check_cam_id in state_map.keys():
                    with state_map[check_cam_id].det_lock:
                        _intr = state_map[check_cam_id].last_intrusion
                        _ts   = state_map[check_cam_id].last_intrusion_ts
                    if _intr:
                        intrusion_active = True
                        break
                    if _ts > 0 and now - _ts <= post_seconds:
                        intrusion_active = True
                        break

                if cam_id in state_map:
                    with state_map[cam_id].det_lock:
                        sensor_data = state_map[cam_id].last_sensor_data

            if sensor_data is None and sensor_getter:
                sensor_data = sensor_getter(timestamp)

            if recording_mode == "full":
                full_dir = full_folder if full_folder is not None else save_dir
                # full 모드: 시작 시 이벤트 폴더 생성
                if full_event_folder is None:
                    full_event_folder = _create_event_folder(full_dir, timestamp, sensor_data)
                    # full 모드 폴더 정리
                    _cleanup_old_folders(full_dir, MAX_FULL_FOLDERS, is_full_mode=True)
                    logger.event_info(
                        EventType.STATE_CHANGE,
                        "Full 모드 이벤트 폴더 생성",
                        {"folder": full_event_folder}
                    )
                
                if cam_id not in writers:
                    # full 모드: full_recording/event_xxx/ 폴더에 저장
                    result = _create_writer(save_dir, cam_id, timestamp, frame, fps_map, codec, sensor_data, event_folder=full_event_folder)
                    if result is None:
                        continue
                    writers[cam_id], writer_paths[cam_id] = result
                
                # 타임스탬프 기반 반복 write (누산기 방식 - 오차 이월로 버벅임/느림 방지)
                fps = fps_map.get(cam_id, 30.0) or 30.0
                target_dt = 1.0 / fps
                max_gap = target_dt * 4  # 최대 4프레임 공백까지 허용 (이상치 클리핑)
                prev = pending_frames.get(cam_id)
                if prev is not None:
                    prev_ts, prev_frame = prev
                    gap = min(timestamp - prev_ts, max_gap)
                    frame_oweds[cam_id] = frame_oweds.get(cam_id, 0.0) + gap / target_dt
                    repeat = int(frame_oweds[cam_id])
                    frame_oweds[cam_id] -= repeat
                    for _ in range(repeat):
                        writers[cam_id].write(prev_frame)
                pending_frames[cam_id] = (timestamp, frame)
                continue

            buffer = buffers.setdefault(cam_id, deque())
            # 오버레이가 적용된 프레임을 버퍼에 저장
            buffer.append((timestamp, frame))
            cutoff = timestamp - buffer_seconds
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()

            logger.debug(
                "버퍼 상태",
                {"cam_id": cam_id, "buffer_size": len(buffer)}
            )

            is_recording = recording_active.get(cam_id, False)
            shutdown_mode = stop_event.is_set() and shutdown_deadline is not None
            active_shutdown_deadline = shutdown_deadline
            if shutdown_mode and active_shutdown_deadline is not None:
                # q 이후에는 모든 카메라를 동일하게 post 구간에 포함
                intrusion_active = time.time() <= active_shutdown_deadline

            if intrusion_active:
                if not is_recording:
                    # 새 이벤트 시작: 이벤트 폴더 생성
                    if current_event_folder is None:
                        event_start_ts = timestamp
                        current_event_folder = _create_event_folder(save_dir, timestamp, sensor_data)
                        # event 모드 폴더 정리
                        _cleanup_old_folders(save_dir, MAX_EVENT_FOLDERS, is_full_mode=False)
                        logger.event_info(
                            EventType.STATE_CHANGE,
                            "이벤트 폴더 생성",
                            {"folder": current_event_folder, "timestamp": timestamp}
                        )
                    
                    logger.event_info(
                        EventType.DATA_PROCESSED,
                        "VideoWriter 생성 시도 (침입 감지)",
                        {"camera": cam_id, "buffer_size": len(buffer)}
                    )
                    
                    result = _create_writer(save_dir, cam_id, timestamp, frame, fps_map, codec, sensor_data, event_folder=current_event_folder)
                    
                    logger.event_info(
                        EventType.DATA_PROCESSED,
                        "VideoWriter 생성 결과",
                        {"camera": cam_id, "writer_created": result is not None}
                    )
                    
                    if result is None:
                        logger.event_error(
                            EventType.ERROR_OCCURRED,
                            "VideoWriter 생성 실패 (침입 감지 시)",
                            {"camera": cam_id}
                        )
                        recording_active[cam_id] = False
                        post_deadline[cam_id] = None
                        continue
                    writers[cam_id], writer_paths[cam_id] = result
                    recording_active[cam_id] = True
                    post_deadline[cam_id] = None
                    frame_oweds[cam_id] = 0.0  # 새 이벤트 시작 시 누산기 초기화

                    # 버퍼 프레임 저장 - 누산기 방식 (오차 이월로 버벅임/느림 방지)
                    fps = fps_map.get(cam_id, 30.0) or 30.0
                    target_dt = 1.0 / fps
                    max_gap = target_dt * 4  # 최대 4프레임 공백까지 허용 (이상치 클리핑)
                    buf_list = list(buffer)
                    buffer_count = 0
                    buf_owed = 0.0
                    try:
                        for i, (buf_ts, buf_frame) in enumerate(buf_list):
                            if i + 1 < len(buf_list):
                                gap = min(buf_list[i + 1][0] - buf_ts, max_gap)
                            else:
                                gap = target_dt  # 마지막 프레임은 1회만
                            buf_owed += gap / target_dt
                            repeat = int(buf_owed)
                            buf_owed -= repeat
                            for _ in range(repeat):
                                writers[cam_id].write(buf_frame)
                            buffer_count += repeat
                    except Exception as e:
                        logger.event_error(
                            EventType.ERROR_OCCURRED,
                            "버퍼 프레임 저장 중 예외",
                            {"camera": cam_id, "error": str(e), "error_type": type(e).__name__, "buffer_count": buffer_count}
                        )
                    # 버퍼 누산기 잔량을 실시간 구간으로 이월
                    frame_oweds[cam_id] = buf_owed

                    logger.event_info(
                        EventType.MODULE_START,
                        "버퍼 프레임 저장 완료",
                        {"camera": cam_id, "buffer_frames": buffer_count}
                    )

                    # 마지막 버퍼 프레임을 pending에 등록 (실시간 구간과 연결)
                    if buf_list:
                        pending_frames[cam_id] = buf_list[-1]
                    else:
                        pending_frames[cam_id] = None

                    # 녹화 시작 시 버퍼만 저장하고 끝냄 (현재 프레임 중복 방지)
                    # 다음 프레임부터는 intrusion_active 상태에서 실시간 저장
                    continue

                # 침입이 다시 활성화됐으므로 post_deadline 초기화
                # (탐지 누락으로 잠깐 False가 됐다가 True로 돌아온 경우 방지)
                if (not shutdown_mode) and post_deadline.get(cam_id) is not None:
                    logger.event_info(
                        EventType.MODULE_START,
                        "침입 재감지: post_deadline 초기화 (파일 분리 방지)",
                        {"camera": cam_id}
                    )
                    post_deadline[cam_id] = None

                writer = writers.get(cam_id)
                if writer:
                    fps = fps_map.get(cam_id, 30.0) or 30.0
                    target_dt = 1.0 / fps
                    max_gap = target_dt * 4  # 최대 4프레임 공백까지 허용 (이상치 클리핑)
                    prev = pending_frames.get(cam_id)
                    if prev is not None:
                        prev_ts, prev_frame = prev
                        gap = min(timestamp - prev_ts, max_gap)
                        frame_oweds[cam_id] = frame_oweds.get(cam_id, 0.0) + gap / target_dt
                        repeat = int(frame_oweds[cam_id])
                        frame_oweds[cam_id] -= repeat
                        for _ in range(repeat):
                            writer.write(prev_frame)
                    pending_frames[cam_id] = (timestamp, frame)
                    logger.debug(
                        "이벤트 프레임 저장",
                        {"camera": cam_id, "timestamp": timestamp}
                    )
                else:
                    logger.event_warning(
                        EventType.ERROR_OCCURRED,
                        "writer가 없음 (침입 중)",
                        {"camera": cam_id}
                    )
                continue

            if is_recording:
                deadline = shutdown_deadline if shutdown_mode else post_deadline.get(cam_id)
                if deadline is None and not shutdown_mode:
                    # wall clock 기준으로 deadline 설정
                    # inference timestamp, save_queue 지연, pre-buffer flush 시간 등과 무관하게
                    # 항상 실제 시계 기준 정확히 post_seconds 를 보장
                    deadline = time.time() + post_seconds
                    post_deadline[cam_id] = deadline
                    logger.event_info(
                        EventType.MODULE_START,
                        "post-event deadline 설정 (wall clock)",
                        {"camera": cam_id, "deadline": deadline, "post_seconds": post_seconds}
                    )

                if deadline is not None and time.time() <= deadline:
                    writer = writers.get(cam_id)
                    if writer:
                        fps = fps_map.get(cam_id, 30.0) or 30.0
                        target_dt = 1.0 / fps
                        max_gap = target_dt * 4  # 최대 4프레임 공백까지 허용 (이상치 클리핑)
                        prev = pending_frames.get(cam_id)
                        if prev is not None:
                            prev_ts, prev_frame = prev
                            gap = min(timestamp - prev_ts, max_gap)
                            frame_oweds[cam_id] = frame_oweds.get(cam_id, 0.0) + gap / target_dt
                            repeat = int(frame_oweds[cam_id])
                            frame_oweds[cam_id] -= repeat
                            for _ in range(repeat):
                                writer.write(prev_frame)
                        pending_frames[cam_id] = (timestamp, frame)
                else:
                    # 녹화 종료 전 pending 프레임 마지막 1회 flush
                    writer = writers.get(cam_id)
                    if writer:
                        prev = pending_frames.pop(cam_id, None)
                        if prev is not None:
                            writer.write(prev[1])
                    writer = writers.pop(cam_id, None)
                    if writer:
                        writer.release()
                        logger.event_info(
                            EventType.MODULE_STOP,
                            "영상 저장기 해제",
                            {"camera": cam_id, "event_folder": current_event_folder}
                        )
                        # H.264 변환 (비동기, 종료 시 join 대기)
                        file_path = writer_paths.pop(cam_id, None)
                        if file_path:
                            if shutdown_mode:
                                shutdown_files.append(file_path)
                            else:
                                t = threading.Thread(
                                    target=_transcode_and_upload,
                                    args=(file_path,),
                                    daemon=False,
                                    name=f"h264_transcode_cam{cam_id}"
                                )
                                t.start()
                                transcode_threads.append(t)
                    recording_active[cam_id] = False
                    post_deadline[cam_id] = None
                    frame_oweds.pop(cam_id, None)  # 녹화 종료 시 누산기 초기화
                    
                    # 모든 카메라 녹화 종료되면 이벤트 폴더 초기화
                    if not any(recording_active.values()):
                        logger.event_info(
                            EventType.MODULE_STOP,
                            "이벤트 종료",
                            {"folder": current_event_folder}
                        )
                        current_event_folder = None
                        event_start_ts = None

    finally:
        for cam_id, writer in writers.items():
            try:
                # pending 프레임 마지막 1회 flush
                prev = pending_frames.pop(cam_id, None)
                if prev is not None:
                    writer.write(prev[1])
                writer.release()
                logger.event_info(
                    EventType.MODULE_STOP,
                    "영상 저장기 해제",
                    {"camera": cam_id}
                )
                # H.264 변환 (비동기, 종료 시 join 대기)
                file_path = writer_paths.pop(cam_id, None)
                if file_path:
                    if stop_event.is_set():
                        shutdown_files.append(file_path)
                    else:
                        t = threading.Thread(
                            target=_transcode_and_upload,
                            args=(file_path,),
                            daemon=False,
                            name=f"h264_transcode_cam{cam_id}"
                        )
                        t.start()
                        transcode_threads.append(t)
            except Exception as e:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "영상 저장기 해제 실패",
                    {"camera": cam_id, "error": str(e)}
                )

        # q 종료 경로: 파일을 카메라 순으로 순차 변환/업로드
        if shutdown_files:
            def _cam_sort_key(path: str) -> int:
                base = os.path.basename(path)
                if base.startswith("camera"):
                    num = "".join(ch for ch in base[6:] if ch.isdigit())
                    if num:
                        return int(num)
                return 10_000

            for file_path in sorted(dict.fromkeys(shutdown_files), key=_cam_sort_key):
                logger.event_info(
                    EventType.MODULE_START,
                    "q 종료: 순차 변환/업로드 시작",
                    {"file": file_path}
                )
                _transcode_and_upload(file_path)

        # 미완료 트랜스코딩 작업이 있으면 완료될 때까지 대기
        for t in transcode_threads:
            if t.is_alive():
                logger.event_info(
                    EventType.MODULE_STOP,
                    "H.264 변환 완료 대기 중",
                    {"thread": t.name}
                )
                t.join()
