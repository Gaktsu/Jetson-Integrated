"""
Watchdog - 시스템 감시 및 자동 재시작
"""
import time
import subprocess
import sys
from datetime import datetime
from utils.logger import get_logger, EventType

logger = get_logger("system.watchdog")


class Watchdog:
    """시스템 감시 및 재시작 관리"""
    
    def __init__(self, script_path: str, max_retries: int = 5, retry_delay: int = 10):
        self.script_path = script_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_count = 0
        
        logger.debug(
            "Watchdog 객체 생성",
            {
                "script_path": script_path,
                "max_retries": max_retries,
                "retry_delay": retry_delay
            }
        )
        
    def run(self):
        """메인 프로그램을 감시하며 실행"""
        logger.event_info(
            EventType.SYSTEM_START,
            "Watchdog 시작",
            {"script_path": self.script_path}
        )
        
        while self.retry_count < self.max_retries:
            try:
                logger.event_info(
                    EventType.MODULE_START,
                    "프로그램 실행 시도",
                    {
                        "retry_count": self.retry_count + 1,
                        "max_retries": self.max_retries
                    }
                )
                
                # 프로그램 실행
                process = subprocess.Popen([sys.executable, self.script_path])
                returncode = process.wait()
                
                if returncode == 0:
                    logger.event_info(
                        EventType.MODULE_STOP,
                        "프로그램이 정상 종료되었습니다",
                        {"return_code": returncode}
                    )
                    break
                else:
                    logger.event_error(
                        EventType.ERROR_OCCURRED,
                        "프로그램이 비정상 종료되었습니다",
                        {"return_code": returncode}
                    )
                    self.retry_count += 1
                    
                    if self.retry_count < self.max_retries:
                        logger.event_warning(
                            EventType.RETRY_ATTEMPT,
                            "프로그램 재시작 대기 중",
                            {
                                "retry_delay": self.retry_delay,
                                "next_retry": self.retry_count + 1
                            }
                        )
                        time.sleep(self.retry_delay)
                    
            except KeyboardInterrupt:
                logger.event_warning(
                    EventType.USER_INPUT,
                    "사용자에 의해 중단되었습니다"
                )
                break
            except Exception as e:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "Watchdog 실행 중 오류 발생",
                    {"error": str(e)},
                    exc_info=True
                )
                self.retry_count += 1
                
                if self.retry_count < self.max_retries:
                    logger.event_warning(
                        EventType.RETRY_ATTEMPT,
                        "프로그램 재시작 대기 중",
                        {"retry_delay": self.retry_delay}
                    )
                    time.sleep(self.retry_delay)
        
        if self.retry_count >= self.max_retries:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "최대 재시도 횟수 초과",
                {"max_retries": self.max_retries}
            )
        
        logger.event_info(EventType.SYSTEM_STOP, "Watchdog 종료")


if __name__ == "__main__":
    import os
    main_script = os.path.join(os.path.dirname(__file__), "..", "main.py")
    watchdog = Watchdog(main_script, max_retries=5, retry_delay=10)
    watchdog.run()
