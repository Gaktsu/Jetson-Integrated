"""
터치스크린 입력 처리 (Linux evdev)

Jetson에서 터치스크린은 /dev/input/eventX 로 노출됩니다.
설치: pip install evdev
장치 확인: ls /dev/input/event*  또는  python3 -m evdev

지원 이벤트:
    - 탭 (단일 터치 press/release)
    - 스와이프 (방향: left/right/up/down)
    - 멀티터치 (슬롯 기반 ABS_MT_POSITION)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, List, Optional
from collections import deque

from errors.enums import TouchscreenError
from utils.logger import get_logger, EventType

try:
    import evdev
    from evdev import ecodes
except Exception:
    evdev = None
    ecodes = None

logger = get_logger("hardware.touchscreen")


# ──────────────────────────────────────────────
# 이벤트 타입
# ──────────────────────────────────────────────

class TouchEventType(Enum):
    TAP      = auto()   # 짧은 단일 터치
    SWIPE_LEFT  = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP    = auto()
    SWIPE_DOWN  = auto()


@dataclass
class TouchEvent:
    """파싱된 터치 이벤트"""
    type: TouchEventType
    x: int = 0          # 터치 시작 x 좌표
    y: int = 0          # 터치 시작 y 좌표
    dx: int = 0         # 스와이프 x 변위
    dy: int = 0         # 스와이프 y 변위
    timestamp: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# 장치 탐색 유틸
# ──────────────────────────────────────────────

def find_touch_device() -> Optional[str]:
    """
    /dev/input/event* 에서 터치스크린 장치를 자동 탐색합니다.

    Returns:
        장치 경로 문자열 (예: '/dev/input/event3') 또는 None
    """
    if evdev is None:
        return None
    try:
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            # ABS_MT_POSITION_X(53) 이벤트를 지원하면 멀티터치 장치로 판단
            if ecodes.EV_ABS in caps:
                abs_codes = [code for code, _ in caps[ecodes.EV_ABS]]
                if ecodes.ABS_MT_POSITION_X in abs_codes or ecodes.ABS_X in abs_codes:
                    logger.event_info(
                        EventType.MODULE_INIT,
                        "터치스크린 장치 자동 탐색 성공",
                        {"path": path, "name": dev.name},
                    )
                    return path
    except Exception as e:
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "터치스크린 탐색 중 예외",
            {"error": str(e)},
        )
    return None


# ──────────────────────────────────────────────
# 메인 클래스
# ──────────────────────────────────────────────

class Touchscreen:
    """
    터치스크린 입력 처리 클래스

    사용법:
        ts = Touchscreen()
        if ts.start():
            evt = ts.read_event()       # 폴링 방식
        ts.stop()

    콜백 방식:
        ts = Touchscreen(on_event=my_handler)
        ts.start()
        # my_handler(TouchEvent) 형식으로 호출됨

    설정:
        TOUCH_DEVICE_PATH = None  → 자동 탐색
        TOUCH_DEVICE_PATH = "/dev/input/event3"  → 지정
    """

    # 스와이프 판정 최소 이동 거리 (픽셀)
    SWIPE_THRESHOLD = 30

    def __init__(
        self,
        device_path: Optional[str] = None,
        on_event: Optional[Callable[[TouchEvent], None]] = None,
        buffer_maxlen: int = 20,
    ):
        self.device_path = device_path
        self.on_event    = on_event

        self._device = None
        self._running    = False
        self._initialized = False
        self._thread: Optional[threading.Thread] = None

        self._buffer: Deque[TouchEvent] = deque(maxlen=buffer_maxlen)
        self._buffer_lock = threading.Lock()

        # 현재 진행 중인 터치 상태
        self._touch_x: Optional[int] = None
        self._touch_y: Optional[int] = None
        self._start_x: Optional[int] = None
        self._start_y: Optional[int] = None
        self._touching = False

    # ──────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────

    def start(self) -> bool:
        """터치스크린 장치 열기 및 읽기 스레드 시작."""
        if evdev is None:
            logger.event_warning(
                EventType.MODULE_INIT,
                "evdev 패키지 없음 — 터치스크린 비활성화 (더미 모드)",
            )
            self._initialized = False
            return False

        path = self.device_path or find_touch_device()
        if path is None:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "터치스크린 장치를 찾을 수 없음",
                {"diagnosis": TouchscreenError.DEVICE_NOT_FOUND.name},
            )
            return False

        try:
            self._device = evdev.InputDevice(path)
            self._device.grab()   # 다른 프로세스가 이벤트 가져가지 않도록 독점
            self.device_path = path
            self._running = True
            self._initialized = True

            self._thread = threading.Thread(
                target=self._read_loop,
                daemon=True,
                name="touchscreen_reader",
            )
            self._thread.start()

            logger.event_info(
                EventType.MODULE_START,
                "터치스크린 시작",
                {"path": path, "name": self._device.name},
            )
            return True

        except PermissionError:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "터치스크린 접근 권한 없음 (sudo 또는 input 그룹 필요)",
                {"path": path, "diagnosis": TouchscreenError.PERMISSION_DENIED.name},
            )
            return False
        except Exception as e:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "터치스크린 초기화 예외",
                {"path": path, "error": str(e)},
                exc_info=True,
            )
            return False

    def stop(self) -> None:
        """터치스크린 종료."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            if self._device is not None:
                self._device.ungrab()
                self._device.close()
                self._device = None
        except Exception:
            pass
        self._initialized = False
        logger.event_info(EventType.MODULE_STOP, "터치스크린 종료")

    def read_event(self) -> Optional[TouchEvent]:
        """
        버퍼에서 가장 오래된 미처리 이벤트를 꺼내 반환합니다 (FIFO).

        Returns:
            TouchEvent 또는 버퍼가 비어있으면 None
        """
        with self._buffer_lock:
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def is_running(self) -> bool:
        return self._running and self._initialized

    # ──────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────

    def _read_loop(self) -> None:
        """evdev 이벤트 읽기 루프 (별도 스레드)."""
        logger.event_info(EventType.MODULE_START, "터치스크린 읽기 루프 시작")
        try:
            for ev in self._device.read_loop():
                if not self._running:
                    break
                self._process_event(ev)
        except OSError as e:
            if self._running:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "터치스크린 읽기 오류",
                    {"error": str(e)},
                )
        except Exception as e:
            if self._running:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "터치스크린 읽기 루프 예외",
                    {"error": str(e)},
                    exc_info=True,
                )
        logger.event_info(EventType.MODULE_STOP, "터치스크린 읽기 루프 종료")

    def _process_event(self, ev) -> None:
        """단일 evdev 이벤트를 파싱해 TouchEvent로 변환."""
        if ev.type == ecodes.EV_ABS:
            if ev.code in (ecodes.ABS_X, ecodes.ABS_MT_POSITION_X):
                self._touch_x = ev.value
                if not self._touching:
                    self._start_x = ev.value
            elif ev.code in (ecodes.ABS_Y, ecodes.ABS_MT_POSITION_Y):
                self._touch_y = ev.value
                if not self._touching:
                    self._start_y = ev.value

        elif ev.type == ecodes.EV_KEY:
            if ev.code == ecodes.BTN_TOUCH:
                if ev.value == 1:   # 손가락 닿음
                    self._touching = True
                    self._start_x = self._touch_x
                    self._start_y = self._touch_y
                elif ev.value == 0:  # 손가락 뗌
                    self._on_release()

        elif ev.type == ecodes.EV_SYN:
            # SYN_REPORT: 멀티터치 시작 위치 갱신
            if not self._touching and self._touch_x is not None:
                self._start_x = self._touch_x
                self._start_y = self._touch_y

    def _on_release(self) -> None:
        """터치 릴리즈 시 제스처 판정."""
        self._touching = False

        if self._start_x is None or self._touch_x is None:
            return

        dx = self._touch_x - self._start_x
        dy = self._touch_y - self._start_y if (self._touch_y and self._start_y) else 0

        abs_dx, abs_dy = abs(dx), abs(dy)

        if abs_dx < self.SWIPE_THRESHOLD and abs_dy < self.SWIPE_THRESHOLD:
            evt_type = TouchEventType.TAP
        elif abs_dx >= abs_dy:
            evt_type = TouchEventType.SWIPE_RIGHT if dx > 0 else TouchEventType.SWIPE_LEFT
        else:
            evt_type = TouchEventType.SWIPE_DOWN if dy > 0 else TouchEventType.SWIPE_UP

        evt = TouchEvent(
            type=evt_type,
            x=self._start_x or 0,
            y=self._start_y or 0,
            dx=dx,
            dy=dy,
        )

        with self._buffer_lock:
            self._buffer.append(evt)

        logger.event_info(
            EventType.USER_INPUT,
            "터치 이벤트",
            {"type": evt_type.name, "x": evt.x, "y": evt.y, "dx": dx, "dy": dy},
        )

        if self.on_event is not None:
            self.on_event(evt)

        # 상태 초기화
        self._start_x = None
        self._start_y = None
