"""
메인 실행 파일 - 전체 흐름 제어
"""
import cv2
import threading
import time
import queue
import numpy as np
from typing import Dict, List, Tuple
from config.settings import (
    CAMERA_INDICES,
    DISPLAY_MODE,
    EVENT_RECORD_POST_SEC,
    RECORDING_MODE,
    WATCHDOG_TEST_DELAY,
    WATCHDOG_TEST_MODE,
    WINDOW_NAME,
)
from ai.model import load_model
from hardware.camera import CameraCapture, init_cameras
from hardware.buzzer import Buzzer
from hardware.gps import GPS
from hardware.imu import IMU
from pipeline.shared_state import SharedState
from pipeline.capture import start_capture_threads
from pipeline.inference import start_inference_thread
from pipeline.recorder import start_save_thread
from pipeline.sensors import start_sensor_threads
from ui.renderer import draw_detections
from utils.time_utils import FPSCounter
from utils.sensor_sync import SensorBuffer
from utils.logger import get_logger, EventType
from config import roi_setup as _roi_setup_mod
# 중앙 Orchestrator 로거
logger = get_logger("main_orchestrator")


# ──────────────────────────────────────────────
# 초기화 헬퍼
# ──────────────────────────────────────────────

def _start_watchdog_timer() -> None:
    """Watchdog 테스트 모드: 설정된 시간 후 강제 오류 발생"""
    def _crash():
        time.sleep(WATCHDOG_TEST_DELAY)
        logger.event_error(EventType.ERROR_OCCURRED, "⚠️ Watchdog 테스트: 강제 오류 발생")
        raise RuntimeError("Watchdog 테스트를 위해 일부러 발생시킨 오류입니다!")
    threading.Thread(target=_crash, daemon=False, name="watchdog_test").start()




def _build_sensor_getter(gps_buffer: SensorBuffer, imu_buffer: SensorBuffer):
    """GPS/IMU 최신 데이터를 시각 기준으로 묶어 반환하는 getter 클로저 생성"""
    def get_sensor_snapshot(timestamp: float) -> dict:
        gps_sample = gps_buffer.get_latest()
        imu_sample = imu_buffer.get_latest()
        return {
            "timestamp": timestamp,
            "gps": {"ts": gps_sample[0], "data": gps_sample[1]} if gps_sample else None,
            "imu": {"ts": imu_sample[0], "data": imu_sample[1]} if imu_sample else None,
        }
    return get_sensor_snapshot




# ──────────────────────────────────────────────
# 메인 루프 헬퍼
# ──────────────────────────────────────────────

def _get_current_frame(states: List[SharedState], cam_idx: int):
    """현재 표시 카메라의 최신 프레임과 탐지 결과를 SharedState에서 읽어 반환"""
    state = states[cam_idx]
    with state.frame_lock:
        seq = state.latest_frame_seq
        frame = state.latest_frame.copy() if state.latest_frame is not None else None
    with state.det_lock:
        detections = list(state.last_detections)
        intrusion = state.last_intrusion
        last_intrusion_ts = state.last_intrusion_ts
    return frame, seq, detections, intrusion, last_intrusion_ts


def _determine_saving(intrusion: bool, last_intrusion_ts: float) -> bool:
    """현재 saving 상태 결정 (full 모드는 항상 True, event 모드는 침입 + post 구간)"""
    if RECORDING_MODE == "full":
        return True
    if last_intrusion_ts > 0:
        return intrusion or (time.time() - last_intrusion_ts <= EVENT_RECORD_POST_SEC)
    return intrusion


def _determine_saving_global(states: List[SharedState]) -> bool:
    """모든 카메라 중 하나라도 저장 중이면 True (침입 + post 구간 포함)"""
    if RECORDING_MODE == "full":
        return True
    now = time.time()
    for state in states:
        with state.det_lock:
            intrusion = state.last_intrusion
            last_ts   = state.last_intrusion_ts
        if intrusion:
            return True
        if last_ts > 0 and (now - last_ts) <= EVENT_RECORD_POST_SEC:
            return True
    return False


def _open_roi_setup(states: List[SharedState]) -> None:
    """ROI 캘리브레이션 도구를 메인 스레드에서 실행.
    백그라운드 스레드(캡처/추론/저장/센서)는 계속 실행된다.
    카메라 충돌 방지를 위해 SharedState의 최신 프레임을 frame_getter로 전달한다.
    """
    logger.event_info(EventType.USER_INPUT, "ROI 캘리브레이션 도구 열기", {"key": "w"})

    # 창을 닫기 전에 현재 창 모드와 크기를 읽어 roi_setup에 그대로 전달
    is_fullscreen = (
        cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
    )
    rect = cv2.getWindowImageRect(WINDOW_NAME)  # (x, y, w, h)
    window_config = {
        "fullscreen": is_fullscreen,
        "width": rect[2] if rect[2] > 0 else 1280,
        "height": rect[3] if rect[3] > 0 else 720,
    }

    cv2.destroyWindow(WINDOW_NAME)
    cv2.waitKey(1)

    def _make_getter(state):
        def getter():
            with state.frame_lock:
                if state.latest_frame is not None:
                    return state.latest_frame.copy()
            return None
        return getter

    frame_getters = {
        CAMERA_INDICES[i]: _make_getter(states[i])
        for i in range(len(states))
    }

    _roi_setup_mod.main(frame_getters=frame_getters, window_config=window_config)
    # 캘리브레이션 종료 후 메인 전체화면 창 복원
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    logger.event_info(EventType.STATE_CHANGE, "ROI 캘리브레이션 완료, 메인 창 복원")


def _handle_keypress(key: int, cam_idx: int, num_cameras: int) -> Tuple[int, bool]:
    """키 입력 처리. (새 카메라 인덱스, 종료 여부) 반환"""
    if key == ord('c'):
        new_idx = (cam_idx + 1) % num_cameras
        logger.event_info(EventType.USER_INPUT, "카메라 전환",
                          {"key": "c", "camera_index": CAMERA_INDICES[new_idx]})
        print(f"카메라 {CAMERA_INDICES[new_idx]}번으로 전환")
        return new_idx, False
    if key == ord('q'):
        logger.event_info(EventType.USER_INPUT, "종료 신호 감지", {"key": "q"})
        return cam_idx, True
    return cam_idx, False


# ──────────────────────────────────────────────
# 분할 화면 헬퍼 (DISPLAY_MODE="split" 전용)
# ──────────────────────────────────────────────

def _stack_panels(panels: List[cv2.Mat]) -> cv2.Mat:
    """N개 패널을 2열 그리드로 합성"""
    if not panels:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if len(panels) == 1:
        return panels[0]
    h, w = panels[0].shape[:2]
    resized = [cv2.resize(p, (w, h)) if p.shape[:2] != (h, w) else p for p in panels]
    if len(resized) == 2:
        return np.hstack(resized)
    # 3개 이상: 2열 그리드
    cols = 2
    rows = (len(resized) + cols - 1) // cols
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    while len(resized) < rows * cols:
        resized.append(blank)
    row_imgs = [np.hstack(resized[r * cols:(r + 1) * cols]) for r in range(rows)]
    return np.vstack(row_imgs)


def _build_split_frame(
    states: List[SharedState],
    fps_counters: Dict[int, "FPSCounter"],
    draw_ms_list: List[float],
) -> Tuple[cv2.Mat, bool, List[float]]:
    """모든 카메라 프레임을 분할 합성. (combined_frame, any_intrusion, new_draw_ms_list) 반환"""
    panels = []
    any_intrusion = False
    new_draw_ms: List[float] = []
    global_saving = _determine_saving_global(states)  # 전체 카메라 기준 saving

    for i, state in enumerate(states):
        cam_id = CAMERA_INDICES[i]

        with state.frame_lock:
            frame = state.latest_frame.copy() if state.latest_frame is not None else None
        with state.det_lock:
            detections = list(state.last_detections)
            intrusion  = state.last_intrusion

        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        any_intrusion = any_intrusion or intrusion
        fps = fps_counters[cam_id].update()

        t0 = time.perf_counter()
        panel = draw_detections(
            frame, detections, fps, global_saving, cam_id,
            intrusion=intrusion,
            capture_ms=state.capture_ms,
            inference_ms=state.inference_ms,
            postprocess_ms=state.postprocess_ms,
            draw_ms=draw_ms_list[i],
        )
        new_draw_ms.append((time.perf_counter() - t0) * 1000)
        panels.append(panel)

    return _stack_panels(panels), any_intrusion, new_draw_ms


# ──────────────────────────────────────────────
# 정리
# ──────────────────────────────────────────────

def _cleanup(
    cameras: List[CameraCapture],
    states: List[SharedState],
    threads: List[threading.Thread],
    save_stop_event: threading.Event,
    inference_stop_event: threading.Event,
    sensor_stop_event: threading.Event,
    gps: GPS,
    imu: IMU,
    buzzer: Buzzer,
) -> None:
    """모든 스레드 종료 및 리소스 해제"""
    logger.event_info(EventType.MODULE_STOP, "시스템 종료 프로세스 시작")

    # 1단계: 추론은 즉시 중단, 저장은 종료 신호만 먼저 보냄
    #   캡처 스레드는 유지 → save_worker가 q 시점부터 post 구간 프레임을 계속 받을 수 있음
    inference_stop_event.set()
    save_stop_event.set()
    sensor_stop_event.set()
    cv2.destroyAllWindows()
    logger.debug("OpenCV 윈도우 종료")
    buzzer.stop()

    # 2단계: save_worker가 q 시점 기준 post 녹화 + 변환 + 업로드 완료할 때까지 대기
    logger.debug("save_worker 종료 대기 중 (고정 post 녹화 + 변환 + 업로드)")
    for t in threads:
        if t.name == "save_worker":
            t.join()
            break

    # 3단계: 저장이 끝난 뒤 캡처/추론/센서 스레드 종료
    logger.debug("캡처/추론 스레드 종료 신호 전송")
    for state in states:
        state.stop_event.set()
    for t in threads:
        if t.name != "save_worker":
            t.join(timeout=1.0)

    for i, camera in enumerate(cameras):
        camera.release()
        logger.event_info(EventType.CAMERA_CLOSE, f"카메라 {i} 리소스 해제")
    gps.stop()
    imu.stop()
    logger.event_info(EventType.SYSTEM_STOP, "Person Detection System 종료 완료")
    try:
        import subprocess as _sp
        _sp.run(["stty", "sane"], check=False)
    except Exception:
        pass


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    """메인 함수 - 전체 시스템 흐름 제어

    흐름:
        1. 모델 로드
        2. 카메라 초기화
        3. 공유 자원 준비
        4. 스레드 시작 (캡처 / 추론 / GPS / IMU / 저장)
        5. 부저 초기화
        6. 메인 표시 루프 (프레임 획득 → 시각화 → 키 입력)
        7. 정리 (스레드 종료, 리소스 해제)
    """
    logger.event_info(EventType.SYSTEM_START, "Person Detection System 시작")

    # Watchdog 테스트 모드 확인
    if WATCHDOG_TEST_MODE:
        logger.event_warning(EventType.MODULE_INIT,
                             f"⚠️ WATCHDOG 테스트 모드 활성화: {WATCHDOG_TEST_DELAY}초 후 강제 종료됩니다",
                             {"test_delay": WATCHDOG_TEST_DELAY})
        _start_watchdog_timer()

    # 1. 모델 로드
    model = load_model()
    if model is None:
        return

    # 2. 카메라 초기화
    cameras, states = init_cameras()
    if cameras is None:
        return

    # 3. 공유 자원 준비
    save_queue       = queue.Queue(maxsize=512)
    save_stop_event  = threading.Event()
    inference_stop_event = threading.Event()
    sensor_stop_event = threading.Event()
    fps_map: Dict[int, float] = {}
    state_map    = {cam_id: state for cam_id, state in zip(CAMERA_INDICES, states)}
    fps_counters = {cam_id: FPSCounter() for cam_id in CAMERA_INDICES}

    gps_buffer = SensorBuffer(maxlen=1)
    imu_buffer = SensorBuffer(maxlen=1)
    get_sensor_snapshot = _build_sensor_getter(gps_buffer, imu_buffer)

    # 4. 스레드 시작
    gps, imu = GPS(), IMU()
    threads: List[threading.Thread] = (
        start_capture_threads(cameras, states, fps_map, save_queue)
        + [start_inference_thread(model, states, get_sensor_snapshot, inference_stop_event)]
        + start_sensor_threads(gps, imu, sensor_stop_event, gps_buffer, imu_buffer)
        + [start_save_thread(save_queue, save_stop_event, fps_map, get_sensor_snapshot, state_map)]
    )
    logger.event_info(EventType.MODULE_START, f"{len(cameras)}개 카메라 스레드 시작 완료")

    # 5. 부저 초기화
    buzzer = Buzzer(pin=32, use_board=True)
    buzzer.start()

    # 6. 디스플레이 창 설정
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 7. 메인 표시 루프
    cam_idx = 0
    last_frame_seq = -1
    draw_ms = 0.0
    draw_ms_list = [0.0] * len(cameras)   # split 모드용 카메라별 draw_ms
    last_seqs = [-1] * len(cameras)        # split 모드용 프레임 시퀀스 추적
    try:
        logger.event_info(EventType.STATE_CHANGE, "메인 루프 시작",
                          {"exit_key": "q",
                           "switch_key": "c" if DISPLAY_MODE == "switch" else "없음",
                           "display_mode": DISPLAY_MODE,
                           "num_cameras": len(cameras)})
        while True:
            if DISPLAY_MODE == "split":
                # ── 분할 모드: 모든 카메라 동시 표시 ──
                new_seqs = [s.latest_frame_seq for s in states]
                if new_seqs == last_seqs:
                    time.sleep(0.001)
                    continue
                last_seqs = new_seqs[:]

                combined, any_intrusion, draw_ms_list = _build_split_frame(
                    states, fps_counters, draw_ms_list
                )
                buzzer.activate() if any_intrusion else buzzer.deactivate()
                cv2.imshow(WINDOW_NAME, combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.event_info(EventType.USER_INPUT, "종료 신호 감지", {"key": "q"})
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break
                elif key == ord('w'):
                    _open_roi_setup(states)

            else:
                # ── 전환 모드: 한 카메라씩, [C] 키로 전환 ──
                frame, seq, detections, intrusion, last_intrusion_ts = _get_current_frame(states, cam_idx)
                if frame is None or seq == last_frame_seq:
                    time.sleep(0.001)
                    continue
                last_frame_seq = seq

                fps = fps_counters[CAMERA_INDICES[cam_idx]].update()
                buzzer.activate() if intrusion else buzzer.deactivate()
                saving = _determine_saving_global(states)  # 모든 카메라 기준

                state = states[cam_idx]
                t_draw = time.perf_counter()
                frame_drawn = draw_detections(
                    frame, detections, fps, saving, CAMERA_INDICES[cam_idx],
                    intrusion=intrusion,
                    capture_ms=state.capture_ms,
                    inference_ms=state.inference_ms,
                    postprocess_ms=state.postprocess_ms,
                    draw_ms=draw_ms,
                )
                draw_ms = (time.perf_counter() - t_draw) * 1000

                cv2.imshow(WINDOW_NAME, frame_drawn)
                key = cv2.waitKey(1) & 0xFF
                cam_idx, should_quit = _handle_keypress(key, cam_idx, len(cameras))
                if should_quit:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break
                if key == ord('c'):
                    last_frame_seq = -1  # 카메라 전환 시 이전 시퀀스 초기화
                elif key == ord('w'):
                    _open_roi_setup(states)
                    last_frame_seq = -1  # 창 복원 후 이전 시퀀스 초기화

    except KeyboardInterrupt:
        logger.event_warning(EventType.USER_INPUT, "키보드 인터럽트로 종료")
    except Exception as e:
        logger.event_error(EventType.ERROR_OCCURRED, "메인 루프 오류 발생",
                           {"error": str(e)}, exc_info=True)
    finally:
        _cleanup(cameras, states, threads, save_stop_event, inference_stop_event, sensor_stop_event, gps, imu, buzzer)


if __name__ == "__main__":
    main()
