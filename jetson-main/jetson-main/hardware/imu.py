"""
IMU 센서 (선택 사항)
"""
from __future__ import annotations

import os
import time
from typing import Optional
from errors.enums import IMUError
from utils.logger import get_logger, EventType

try:
    import smbus2  # type: ignore
except Exception:
    smbus2 = None

logger = get_logger("hardware.imu")


def diagnose_imu_error(bus_path: str) -> IMUError:
    """IMU 연결 실패 시 에러 타입 진단"""
    if not os.path.exists(bus_path):
        return IMUError.DEVICE_NOT_FOUND

    try:
        with open(bus_path, "rb"):
            pass
    except PermissionError:
        return IMUError.PERMISSION_DENIED
    except OSError:
        return IMUError.DEVICE_BUSY

    if smbus2 is None:
        return IMUError.UNKNOWN

    return IMUError.OK


class IMU:
    """IMU 센서 제어 클래스"""
    
    def __init__(self, bus_num: int = 1, address: int = 0x68, bus_path: str = "/dev/i2c-1"):
        self.bus_num = bus_num
        self.address = address
        self.bus_path = bus_path
        self.running = False
        self._bus: Optional["smbus2.SMBus"] = None
        
    def start(self, max_retries: int = 5, retry_delay: float = 2.0) -> bool:
        """IMU 초기화"""
        logger.event_info(
            EventType.MODULE_START,
            "IMU 시작",
            {"bus_num": self.bus_num, "address": hex(self.address), "max_retries": max_retries}
        )

        error_messages = {
            IMUError.DEVICE_NOT_FOUND: "IMU 장치를 찾을 수 없습니다.",
            IMUError.DEVICE_BUSY: "IMU 장치가 다른 프로그램에서 사용 중입니다.",
            IMUError.PERMISSION_DENIED: "IMU 접근 권한이 거부되었습니다.",
            IMUError.BUS_ERROR: "I2C 버스 오류가 발생했습니다.",
            IMUError.TIMEOUT: "IMU 응답이 없습니다.",
            IMUError.INVALID_DATA: "IMU 데이터 형식이 올바르지 않습니다.",
            IMUError.UNKNOWN: "알 수 없는 IMU 오류가 발생했습니다.",
        }

        retry_count = 0
        while retry_count < max_retries:
            error_type = diagnose_imu_error(self.bus_path)
            if error_type == IMUError.OK and smbus2 is not None:
                try:
                    self._bus = smbus2.SMBus(self.bus_num)
                    self.running = True
                    return True
                except Exception:
                    error_type = IMUError.BUS_ERROR
            elif error_type == IMUError.OK and smbus2 is None:
                error_type = IMUError.UNKNOWN

            retry_count += 1
            error_msg = error_messages.get(error_type, "알 수 없는 오류")
            print(f"[에러 유형: {error_type.name}] {error_msg}")
            print(f"IMU 연결 재시도 중... ({retry_count}/{max_retries})")
            time.sleep(retry_delay)

        return False
    
    def read_data(self):
        """IMU 데이터 읽기"""
        if not self.running or self._bus is None:
            return None

        try:
            return None
        except Exception:
            return None
    
    def stop(self):
        """IMU 중지"""
        self.running = False
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        logger.event_info(
            EventType.MODULE_STOP,
            "IMU 종료",
            {"bus_num": self.bus_num, "address": hex(self.address)}
        )
