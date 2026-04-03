"""
AI detection schema and intrusion check.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import List, Optional, Sequence, Tuple, TypedDict

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
    직사각형 경고 영역 침입 확인 (레거시 — bbox 전체 겹침 기준).

    Args:
        detections:   탐지 결과 리스트
        warning_zone: (x1, y1, x2, y2) 직사각형 경고 영역

    Returns:
        침입 여부
    """
    wx1, wy1, wx2, wy2 = warning_zone
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if not (x2 < wx1 or x1 > wx2 or y2 < wy1 or y1 > wy2):
            return True
    return False


def check_intrusion_polygon(
    detections: List[Detection],
    roi_polygon: Optional[Sequence[Sequence[int]]],
) -> bool:
    """
    폴리곤 ROI 침입 확인 (foot-point 기반).

    yolo_test-main의 dectect_roi_J.py 방식을 이식:
    - 판별 기준: 바운딩 박스 하단 중앙(발끝) 좌표가 폴리곤 내부에 있는지 검사
    - 발끝 좌표: foot_x = (x1+x2)//2,  foot_y = y2
    - cv2.pointPolygonTest 반환값 >= 0 이면 내부(경계 포함)

    Args:
        detections:  탐지 결과 리스트
        roi_polygon: [[x,y], ...] 형태의 꼭짓점 목록 (4점 사다리꼴 권장).
                     None 이면 항상 False 반환 (ROI 미설정 상태).

    Returns:
        침입 여부
    """
    if roi_polygon is None or len(roi_polygon) < 3:
        return False

    poly = np.array(roi_polygon, dtype=np.int32)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        foot_x = (x1 + x2) // 2
        foot_y = y2
        # >= 0: 내부 또는 경계 위 / < 0: 외부
        if cv2.pointPolygonTest(poly, (float(foot_x), float(foot_y)), False) >= 0:
            return True
    return False


def load_roi_polygon(config_path: str) -> Optional[List[List[int]]]:
    """
    roi_config_cam{N}.json 파일을 읽어 폴리곤 좌표를 반환.

    Args:
        config_path: roi_config_cam{N}.json 파일 절대 경로

    Returns:
        [[x,y], ...] 또는 파일 없으면 None
    """
    import json
    import os
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        polygon = data.get("roi_polygon")
        if polygon and len(polygon) >= 3:
            return polygon
    except Exception:
        pass
    return None
