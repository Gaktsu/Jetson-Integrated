"""
에러 타입 열거형
"""
from enum import Enum, auto


class CameraError(Enum):
    """카메라 관련 에러"""
    OK = auto()
    DEVICE_NOT_FOUND = auto()
    DEVICE_BUSY = auto()
    PERMISSION_DENIED = auto()
    BACKEND_ERROR = auto()
    UNKNOWN = auto()


class GPSError(Enum):
    """GPS 관련 에러"""
    OK = auto()
    DEVICE_NOT_FOUND = auto()
    DEVICE_BUSY = auto()
    PERMISSION_DENIED = auto()
    NO_SIGNAL = auto()
    INVALID_DATA = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()


class IMUError(Enum):
    """IMU 관련 에러"""
    OK = auto()
    DEVICE_NOT_FOUND = auto()
    DEVICE_BUSY = auto()
    PERMISSION_DENIED = auto()
    BUS_ERROR = auto()
    INVALID_DATA = auto()
    TIMEOUT = auto()
    UNKNOWN = auto()


class MicError(Enum):
    """마이크 관련 에러"""
    OK = auto()
    DEVICE_NOT_FOUND = auto()
    DEVICE_BUSY = auto()
    PERMISSION_DENIED = auto()
    UNKNOWN = auto()


class TouchscreenError(Enum):
    """터치스크린 관련 에러"""
    OK = auto()
    DEVICE_NOT_FOUND = auto()
    PERMISSION_DENIED = auto()
    UNKNOWN = auto()
