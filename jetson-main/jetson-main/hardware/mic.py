"""
마이크 입력 처리 (ALSA / PyAudio)

Jetson에서 PyAudio는 ALSA 백엔드를 사용합니다.
설치: sudo apt install python3-pyaudio portaudio19-dev
      pip install pyaudio
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from errors.enums import MicError
from utils.logger import get_logger, EventType

try:
    import pyaudio
except Exception:
    pyaudio = None

logger = get_logger("hardware.mic")


def diagnose_mic_error(device_name: str) -> MicError:
    """마이크 연결 실패 시 에러 타입 진단"""
    if pyaudio is None:
        return MicError.DEVICE_NOT_FOUND
    try:
        pa = pyaudio.PyAudio()
        count = pa.get_device_count()
        pa.terminate()
        if count == 0:
            return MicError.DEVICE_NOT_FOUND
        return MicError.DEVICE_BUSY
    except OSError:
        return MicError.PERMISSION_DENIED
    except Exception:
        return MicError.UNKNOWN


class Microphone:
    """
    마이크 캡처 클래스

    사용법:
        mic = Microphone()
        if mic.start():
            chunk, ts = mic.read_chunk()   # 최신 오디오 청크 읽기
        mic.stop()

    콜백 방식:
        mic = Microphone(on_chunk=my_callback)
        mic.start()
        # my_callback(chunk_bytes, timestamp) 형식으로 호출됨
    """

    def __init__(
        self,
        device: str = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        on_chunk: Optional[Callable[[bytes, float], None]] = None,
        buffer_maxlen: int = 50,
    ):
        self.device      = device
        self.sample_rate = sample_rate
        self.channels    = channels
        self.chunk_size  = chunk_size
        self.on_chunk    = on_chunk

        self._pa: Optional["pyaudio.PyAudio"] = None
        self._stream = None
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._initialized = False

        # 최신 청크를 저장하는 링버퍼 (read_chunk() 폴링용)
        self._buffer: Deque[Tuple[bytes, float]] = deque(maxlen=buffer_maxlen)
        self._buffer_lock = threading.Lock()

    # ──────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────

    def start(self) -> bool:
        """마이크 스트림 시작. 성공 시 True 반환."""
        if pyaudio is None:
            logger.event_warning(
                EventType.MODULE_INIT,
                "pyaudio 패키지 없음 — 마이크 비활성화 (더미 모드)",
            )
            self._initialized = False
            return False

        try:
            self._pa = pyaudio.PyAudio()
            device_index = self._resolve_device_index()

            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
            )
            self._running = True
            self._initialized = True

            self._thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name="mic_capture",
            )
            self._thread.start()

            logger.event_info(
                EventType.MODULE_START,
                "마이크 시작",
                {
                    "device": self.device,
                    "device_index": device_index,
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "chunk_size": self.chunk_size,
                },
            )
            return True

        except OSError as e:
            err = diagnose_mic_error(self.device)
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "마이크 초기화 실패",
                {"device": self.device, "error": str(e), "diagnosis": err.name},
            )
            self._cleanup_pa()
            return False
        except Exception as e:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "마이크 초기화 중 예외",
                {"error": str(e)},
                exc_info=True,
            )
            self._cleanup_pa()
            return False

    def stop(self) -> None:
        """마이크 스트림 종료."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cleanup_pa()
        logger.event_info(EventType.MODULE_STOP, "마이크 종료")

    def read_chunk(self) -> Optional[Tuple[bytes, float]]:
        """
        버퍼에서 가장 최신 오디오 청크를 반환합니다.

        Returns:
            (chunk_bytes, timestamp) 또는 버퍼가 비어있으면 None
        """
        with self._buffer_lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def is_running(self) -> bool:
        return self._running and self._initialized

    # ──────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────

    def _resolve_device_index(self) -> Optional[int]:
        """device 문자열을 PyAudio 장치 인덱스로 변환. 'default'이면 None 반환."""
        if self.device in ("default", ""):
            return None
        if self.device.startswith("hw:") or self.device.startswith("plughw:"):
            # ALSA 장치 이름으로 인덱스 탐색
            count = self._pa.get_device_count()
            for i in range(count):
                info = self._pa.get_device_info_by_index(i)
                if self.device in info.get("name", ""):
                    return i
            logger.event_warning(
                EventType.MODULE_INIT,
                f"마이크 장치 '{self.device}' 를 찾을 수 없음 — 기본 장치 사용",
            )
            return None
        # 숫자 인덱스
        try:
            return int(self.device)
        except ValueError:
            return None

    def _capture_loop(self) -> None:
        """마이크 읽기 루프 (별도 스레드)."""
        logger.event_info(EventType.MODULE_START, "마이크 캡처 루프 시작")
        while self._running:
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                ts = time.time()
                with self._buffer_lock:
                    self._buffer.append((data, ts))
                if self.on_chunk is not None:
                    self.on_chunk(data, ts)
            except OSError as e:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "마이크 읽기 오류",
                    {"error": str(e)},
                )
                time.sleep(0.1)
            except Exception as e:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "마이크 캡처 루프 예외",
                    {"error": str(e)},
                    exc_info=True,
                )
                break
        logger.event_info(EventType.MODULE_STOP, "마이크 캡처 루프 종료")

    def _cleanup_pa(self) -> None:
        """PyAudio 스트림/인스턴스 정리."""
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception:
            pass
        try:
            if self._pa is not None:
                self._pa.terminate()
                self._pa = None
        except Exception:
            pass
        self._initialized = False
