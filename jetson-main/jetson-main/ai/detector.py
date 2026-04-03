"""
AI detection schema and intrusion check.
"""
from __future__ import annotations

from typing import List, Tuple, TypedDict

class Detection(TypedDict):
    """탐지 결과 단일 항목"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

def check_intrusion(
    detections: List[Detection],
    warning_zone: Tuple[int, int, int, int]
) -> bool:
    """
    경고 영역 침입 확인
    
    Args:
        detections: 탐지 결과
        warning_zone: (x1, y1, x2, y2) 경고 영역
    
    Returns:
        침입 여부
    """
    wx1, wy1, wx2, wy2 = warning_zone

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if not (x2 < wx1 or x1 > wx2 or y2 < wy1 or y1 > wy2):
            return True
    return False
