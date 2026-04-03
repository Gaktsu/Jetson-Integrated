"""
시간 관련 유틸리티
"""
from collections import deque
from datetime import datetime
import time


def get_timestamp() -> str:
    """현재 타임스탬프 반환 (문자열)"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_formatted_time() -> str:
    """포맷된 현재 시간 반환"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class FPSCounter:
    """FPS 계산 클래스 (롤링 윈도우 방식)"""

    def __init__(self, window: int = 30):
        """
        Args:
            window: FPS 평균을 이 프레임 수만큼 유지 (default 30)
        """
        self._ts: deque = deque(maxlen=window)
        self.fps = 0.0

    def update(self) -> float:
        """프레임 타임스탬프 기록 및 FPS 계산

        매 호출마다 갱신되므로 화면에 수치가 부드럽게 변합니다.
        """
        now = time.time()
        self._ts.append(now)
        if len(self._ts) >= 2:
            elapsed = self._ts[-1] - self._ts[0]
            if elapsed > 0:
                self.fps = (len(self._ts) - 1) / elapsed
        return self.fps

    def get_fps(self) -> float:
        """현재 FPS 반환"""
        return self.fps
