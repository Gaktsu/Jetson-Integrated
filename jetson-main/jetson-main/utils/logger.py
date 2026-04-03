"""
구조화된 로깅 시스템
- 로그 레벨: INFO, WARNING, ERROR
- 이벤트 로그와 디버그 로그 분리
- 모든 로그에 module_name, event_type, timestamp 포함
"""
import logging
import os
import json
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from config.settings import PROJECT_ROOT


class LogLevel(Enum):
    """로그 레벨"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class EventType(Enum):
    """이벤트 타입"""
    # 시스템 라이프사이클
    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"
    MODULE_INIT = "MODULE_INIT"
    MODULE_START = "MODULE_START"
    MODULE_STOP = "MODULE_STOP"
    
    # 상태 전이
    STATE_CHANGE = "STATE_CHANGE"
    
    # 데이터 수신/처리
    DATA_RECEIVED = "DATA_RECEIVED"
    DATA_PROCESSED = "DATA_PROCESSED"
    
    # 하드웨어
    CAMERA_OPEN = "CAMERA_OPEN"
    CAMERA_CLOSE = "CAMERA_CLOSE"
    CAMERA_ERROR = "CAMERA_ERROR"
    FRAME_CAPTURE = "FRAME_CAPTURE"
    
    # 추론
    INFERENCE_START = "INFERENCE_START"
    INFERENCE_COMPLETE = "INFERENCE_COMPLETE"
    DETECTION_RESULT = "DETECTION_RESULT"
    
    # 오류
    ERROR_OCCURRED = "ERROR_OCCURRED"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    
    # 사용자 입력
    USER_INPUT = "USER_INPUT"
    
    # 기타
    OTHER = "OTHER"


class StructuredLogger:
    """구조화된 로깅을 위한 중앙 Logger 클래스"""
    
    def __init__(self, module_name: str, log_file: str = "project.log"):
        """
        Args:
            module_name: 모듈 이름
            log_file: 로그 파일 이름
        """
        self.module_name = module_name
        
        # 로그 디렉토리 생성
        self.log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 이벤트 로그 핸들러 (파일 + 콘솔)
        self.event_logger = self._setup_logger(
            f"{module_name}_event",
            os.path.join(self.log_dir, f"event_{log_file}"),
            logging.INFO
        )
        
        # 디버그 로그 핸들러 (파일만)
        self.debug_logger = self._setup_logger(
            f"{module_name}_debug",
            os.path.join(self.log_dir, f"debug_{log_file}"),
            logging.DEBUG,
            console=False
        )
    
    def _setup_logger(
        self,
        name: str,
        file_path: str,
        level: int,
        console: bool = True
    ) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers.clear()  # 기존 핸들러 제거
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 핸들러
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러 (이벤트 로그만)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _format_log_message(
        self,
        event_type: EventType,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """로그 메시지 포맷팅"""
        log_entry = {
            "module": self.module_name,
            "event": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        
        if data:
            log_entry["data"] = data
        
        return json.dumps(log_entry, ensure_ascii=False)
    
    # ===== 이벤트 로그 메서드 =====
    
    def event_info(
        self,
        event_type: EventType,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """이벤트 INFO 로그"""
        formatted = self._format_log_message(event_type, message, data)
        self.event_logger.info(formatted)
    
    def event_warning(
        self,
        event_type: EventType,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """이벤트 WARNING 로그"""
        formatted = self._format_log_message(event_type, message, data)
        self.event_logger.warning(formatted)
    
    def event_error(
        self,
        event_type: EventType,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """이벤트 ERROR 로그"""
        formatted = self._format_log_message(event_type, message, data)
        self.event_logger.error(formatted, exc_info=exc_info)
    
    # ===== 디버그 로그 메서드 =====
    
    def debug(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """디버그 로그 (내부 변수, 상세 흐름)"""
        log_entry = {
            "module": self.module_name,
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        
        if data:
            log_entry["data"] = data
        
        formatted = json.dumps(log_entry, ensure_ascii=False)
        self.debug_logger.debug(formatted)
    
    # ===== 편의 메서드 (간단한 로깅용) =====
    
    def info(self, message: str):
        """간단한 INFO 로그 (이벤트 타입 없이)"""
        self.event_info(EventType.OTHER, message)
    
    def warning(self, message: str):
        """간단한 WARNING 로그"""
        self.event_warning(EventType.OTHER, message)
    
    def error(self, message: str, exc_info: bool = False):
        """간단한 ERROR 로그"""
        self.event_error(EventType.ERROR_OCCURRED, message, exc_info=exc_info)


# ===== 로거 팩토리 함수 =====

_loggers: Dict[str, StructuredLogger] = {}


def get_logger(module_name: str) -> StructuredLogger:
    """
    모듈별 로거 인스턴스 가져오기 (싱글톤)
    
    Args:
        module_name: 모듈 이름
    
    Returns:
        StructuredLogger 인스턴스
    """
    if module_name not in _loggers:
        _loggers[module_name] = StructuredLogger(module_name)
    return _loggers[module_name]


# 하위 호환성을 위한 기본 로거
default_logger = get_logger("project")


# ===== 레거시 함수 (하위 호환성) =====

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    레거시 setup_logger 함수 (하위 호환성)
    새 코드에서는 get_logger() 사용 권장
    """
    return logging.getLogger(name)
