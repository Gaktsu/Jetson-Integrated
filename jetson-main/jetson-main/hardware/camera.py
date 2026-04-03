"""
카메라 초기화 및 캡처 기능
"""
from __future__ import annotations

import cv2
import time
import threading
import platform
from typing import List, Optional, Tuple
from config.settings import CAMERA_INDICES, CAMERA_MAX_RETRIES, CAMERA_RETRY_DELAY
from errors.enums import CameraError
from pipeline.shared_state import SharedState
from utils.logger import get_logger, EventType

logger = get_logger("hardware.camera")

# OS 감지
IS_WINDOWS = platform.system() == "Windows"
IS_JETSON = platform.system() == "Linux" and platform.machine() in ["aarch64", "arm64"]


def diagnose_camera_error(camera_index: int) -> CameraError:
    """
    카메라 연결 실패 시 에러 타입을 진단합니다.
    """
    # 1. 기본 테스트 - 요청한 카메라가 열리는지 확인
    test_cap = cv2.VideoCapture(camera_index)
    if test_cap.isOpened():
        test_cap.release()
        return CameraError.OK
    test_cap.release()
    
    # 2. 다른 백엔드로 시도
    if IS_WINDOWS:
        # Windows용: CAP_DSHOW 대신 CAP_MSMF 등
        backends = [cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        # Jetson용 (Linux): CAP_V4L2, CAP_GSTREAMER 등
        backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
    
    for backend in backends:
        test_cap = cv2.VideoCapture(camera_index, backend)
        if test_cap.isOpened():
            test_cap.release()
            return CameraError.BACKEND_ERROR
        test_cap.release()
    
    # 3. 시스템에 카메라가 전혀 없는지 확인 (여러 인덱스 시도)
    any_camera_found = False
    for idx in range(3):  # 0, 1, 2 인덱스 확인
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            any_camera_found = True
            test_cap.release()
            break
        test_cap.release()
    
    # 다른 카메라는 있지만 요청한 인덱스의 카메라가 없는 경우
    if any_camera_found and camera_index >= 1:
        return CameraError.DEVICE_NOT_FOUND
    
    # 시스템에 카메라가 전혀 없는 경우
    if not any_camera_found:
        return CameraError.DEVICE_NOT_FOUND
    
    # 4. 그 외의 경우 (장치 사용 중 또는 권한 문제)
    return CameraError.DEVICE_BUSY


def _open_cap(camera_index: int) -> cv2.VideoCapture:
    """OS에 맞는 백엔드로 VideoCapture를 생성하고 버퍼 크기를 1로 설정한다."""
    if IS_WINDOWS:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    elif IS_JETSON:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_camera_with_retry(
    camera_index: int,
    max_retries: int = 10,
    retry_delay: float = 5.0
) -> Optional[cv2.VideoCapture]:
    """
    카메라를 열고, 실패 시 재시도합니다.
    
    Args:
        camera_index: 카메라 인덱스
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 간 대기 시간 (초)
    
    Returns:
        성공 시 VideoCapture 객체, 실패 시 None
    """
    error_messages = {
        CameraError.DEVICE_NOT_FOUND: "카메라 장치를 찾을 수 없습니다.",
        CameraError.DEVICE_BUSY: "카메라가 다른 프로그램에서 사용 중입니다.",
        CameraError.PERMISSION_DENIED: "카메라 접근 권한이 거부되었습니다.",
        CameraError.BACKEND_ERROR: "카메라 백엔드 오류가 발생했습니다.",
        CameraError.UNKNOWN: "알 수 없는 카메라 오류가 발생했습니다.",
    }
    
    cap = _open_cap(camera_index)

    retry_count = 0
    while not cap.isOpened() and retry_count < max_retries:
        # 에러 진단
        error_type = diagnose_camera_error(camera_index)
        error_msg = error_messages.get(error_type, "알 수 없는 오류")
        
        retry_count += 1
        
        print(f"[에러 유형: {error_type.name}] {error_msg}")
        print(f"카메라 연결 재시도 중... ({retry_count}/{max_retries})")
        
        time.sleep(retry_delay)
        cap.release()
        cap = _open_cap(camera_index)

    if not cap.isOpened():
        print(f"\n카메라를 열 수 없습니다. {max_retries}번 시도 후 실패했습니다.")
        cap.release()
        return None
    print("카메라 연결 성공!")
    return cap


class CameraCapture:
    """카메라 캡처를 담당하는 클래스"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        
    def start(self, max_retries: int = 10, retry_delay: float = 5.0) -> bool:
        """카메라 시작"""
        logger.event_info(
            EventType.MODULE_START,
            "CameraCapture 시작",
            {"camera_index": self.camera_index}
        )
        self.cap = open_camera_with_retry(self.camera_index, max_retries, retry_delay)
        if self.cap is None:
            return False
        self.running = True
        return True
    
    def read_frame(self) -> tuple[bool, Optional[cv2.Mat]]:
        """프레임 읽기"""
        if self.cap is None or not self.running:
            return False, None
        return self.cap.read()
    
    def release(self):
        """카메라 리소스 해제"""
        logger.event_info(
            EventType.MODULE_STOP,
            "CameraCapture 종료",
            {"camera_index": self.camera_index}
        )
        self.running = False
        if self.cap is not None:
            self.cap.release()


def init_cameras() -> Tuple[Optional[List[CameraCapture]], Optional[List[SharedState]]]:
    """설정된 카메라 인덱스로 카메라를 초기화. 하나라도 실패 시 None 반환."""
    cameras: List[CameraCapture] = []
    states: List[SharedState] = []
    for idx in CAMERA_INDICES:
        logger.event_info(EventType.MODULE_INIT, f"카메라 {idx} 초기화 중")
        camera = CameraCapture(idx)
        if not camera.start(CAMERA_MAX_RETRIES, CAMERA_RETRY_DELAY):
            logger.event_error(EventType.CAMERA_ERROR, f"카메라 {idx} 초기화 실패")
            for cam in cameras:
                cam.release()
            return None, None
        cameras.append(camera)
        states.append(SharedState())
        logger.event_info(EventType.CAMERA_OPEN, f"카메라 {idx} 초기화 완료")
    logger.debug(f"{len(cameras)}개 카메라 초기화 완료")
    return cameras, states
