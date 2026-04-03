"""
GPS 처리 (NEO-6M)
"""
from __future__ import annotations

import os
import time
from typing import Optional
from errors.enums import GPSError
from utils.logger import get_logger, EventType

try:
    import serial  # type: ignore
except Exception:
    serial = None

logger = get_logger("hardware.gps")


def diagnose_gps_error(port: str) -> GPSError:
    """GPS 연결 실패 시 에러 타입 진단"""
    if not os.path.exists(port):
        return GPSError.DEVICE_NOT_FOUND

    try:
        with open(port, "rb"):
            pass
    except PermissionError:
        return GPSError.PERMISSION_DENIED
    except OSError:
        return GPSError.DEVICE_BUSY

    if serial is None:
        return GPSError.UNKNOWN

    return GPSError.OK


class GPS:
    """GPS 장치 제어 클래스"""
    
    def __init__(self, port: str = "/dev/ttyTHS0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self._serial: Optional["serial.Serial"] = None
        
    def start(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """GPS 시작"""
        logger.event_info(
            EventType.MODULE_START,
            "GPS 시작",
            {"port": self.port, "max_retries": max_retries}
        )

        error_messages = {
            GPSError.DEVICE_NOT_FOUND: "GPS 장치를 찾을 수 없습니다.",
            GPSError.DEVICE_BUSY: "GPS 장치가 다른 프로그램에서 사용 중입니다.",
            GPSError.PERMISSION_DENIED: "GPS 접근 권한이 거부되었습니다.",
            GPSError.NO_SIGNAL: "GPS 신호를 찾을 수 없습니다.",
            GPSError.TIMEOUT: "GPS 응답이 없습니다.",
            GPSError.INVALID_DATA: "GPS 데이터 형식이 올바르지 않습니다.",
            GPSError.UNKNOWN: "알 수 없는 GPS 오류가 발생했습니다.",
        }

        retry_count = 0
        while retry_count < max_retries:
            error_type = diagnose_gps_error(self.port)
            if error_type == GPSError.OK and serial is not None:
                try:
                    self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
                    self.running = True
                    return True
                except Exception:
                    error_type = GPSError.UNKNOWN
            elif error_type == GPSError.OK and serial is None:
                error_type = GPSError.UNKNOWN

            retry_count += 1
            error_msg = error_messages.get(error_type, "알 수 없는 오류")
            print(f"[에러 유형: {error_type.name}] {error_msg}")
            print(f"GPS 연결 재시도 중... ({retry_count}/{max_retries})")
            time.sleep(retry_delay)

        return False
    
    def read_data(self):
        """GPS 데이터 읽기"""
        if not self.running or self._serial is None:
            return None

        try:
            line = self._serial.readline()
            if not line:
                return None
            return line.decode(errors="ignore").strip()
        except Exception:
            return None
    
    def stop(self):
        """GPS 중지"""
        self.running = False
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        logger.event_info(
            EventType.MODULE_STOP,
            "GPS 종료",
            {"port": self.port}
        )
