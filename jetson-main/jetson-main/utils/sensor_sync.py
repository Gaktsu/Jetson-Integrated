"""
Sensor sample buffering.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional, Tuple, Any


class SensorBuffer:
    """Thread-safe buffer for latest samples."""

    def __init__(self, maxlen: int = 256):
        self._buffer: Deque[Tuple[float, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, timestamp: float, data: Any) -> None:
        with self._lock:
            self._buffer.append((timestamp, data))

    def get_nearest(self, timestamp: float, max_age: float) -> Optional[Tuple[float, Any]]:
        """Return the sample with the closest timestamp within max_age seconds."""
        with self._lock:
            if not self._buffer:
                return None

            best: Optional[Tuple[float, Any]] = None
            best_dt = None

            for ts, data in self._buffer:
                dt = abs(ts - timestamp)
                if dt <= max_age and (best_dt is None or dt < best_dt):
                    best = (ts, data)
                    best_dt = dt

            return best

    def get_latest(self) -> Optional[Tuple[float, Any]]:
        """Return the most recent sample, if any."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]
