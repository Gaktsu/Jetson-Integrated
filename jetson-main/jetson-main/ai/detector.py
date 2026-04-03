"""
AI detection schema, intrusion check, and TTC-based warning level analysis.
"""
from __future__ import annotations

import cv2
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

class _DetectionRequired(TypedDict):
    """탐지 결과 — 필수 필드"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

class Detection(_DetectionRequired, total=False):
    """탐지 결과 단일 항목 (track_id 는 tracking=True 시에만 채워짐)"""
    track_id: Optional[int]


# ──────────────────────────────────────────────
# 경고 레벨
# ──────────────────────────────────────────────

class WarningLevel(Enum):
    """
    3단계 경고 레벨 (yolo_test-main/main_system.py 기준).

    SAFE       : ROI 밖 또는 TTC 이상 없음
    BLIND_SPOT : ROI 내부에 사람이 존재 (정지 시 사각지대)
    APPROACH   : ROI 내부 + 박스 면적 10% 이상 팽창 (보행 속도 접근)
    URGENT     : ROI 내부 + 박스 면적 30% 이상 급팽창 (뛰어오는 속도)
    """
    SAFE       = "SAFE"
    BLIND_SPOT = "BLIND_SPOT"
    APPROACH   = "APPROACH"
    URGENT     = "URGENT"

    def __gt__(self, other: "WarningLevel") -> bool:
        _order = [WarningLevel.SAFE, WarningLevel.BLIND_SPOT,
                  WarningLevel.APPROACH, WarningLevel.URGENT]
        return _order.index(self) > _order.index(other)


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


# ──────────────────────────────────────────────
# TTC (Time-To-Collision) 분석
# ──────────────────────────────────────────────

# yolo_test-main 기준 상수
_TTC_HISTORY_LEN   = 15    # 판별에 사용할 최대 면적 이력 프레임 수
_URGENT_RATIO      = 1.30  # 30% 이상 급팽창 → URGENT  (뛰어오는 속도)
_APPROACH_RATIO    = 1.10  # 10% 이상 팽창   → APPROACH (보행 속도)


def analyze_ttc(
    detections: List[Detection],
    roi_polygon: Optional[Sequence[Sequence[int]]],
    track_history: Dict[int, List[float]],
) -> WarningLevel:
    """
    TTC 알고리즘으로 프레임 전체의 최고 경고 레벨을 반환.

    yolo_test-main/main_system.py 의 로직을 이식:
      1. ROI 내부 여부 → BLIND_SPOT 이상
      2. 박스 면적 이력(15프레임) 팽창 비율:
         - > 1.30 → URGENT
         - > 1.10 → APPROACH

    Args:
        detections:    현재 프레임의 탐지 결과 (track_id 포함 필요)
        roi_polygon:   폴리곤 ROI ([[x,y], ...]), None이면 SAFE 반환
        track_history: {track_id: [box_area, ...]} — SharedState에서 전달.
                       이 함수 내부에서 직접 업데이트(append/trim)함.

    Returns:
        WarningLevel — 해당 프레임의 최고 경고 레벨
    """
    if roi_polygon is None or len(roi_polygon) < 3:
        return WarningLevel.SAFE

    poly = np.array(roi_polygon, dtype=np.int32)
    worst = WarningLevel.SAFE

    for det in detections:
        track_id = det.get("track_id")
        x1, y1, x2, y2 = det["bbox"]
        foot_x = (x1 + x2) // 2
        foot_y = y2

        # ── BLIND_SPOT: ROI 내부 판별 ──
        inside = cv2.pointPolygonTest(poly, (float(foot_x), float(foot_y)), False) >= 0
        if not inside:
            continue

        level = WarningLevel.BLIND_SPOT

        # ── TTC: 박스 면적 팽창 비율 ──
        if track_id is not None:
            box_area = float((x2 - x1) * (y2 - y1))
            history = track_history[track_id]
            history.append(box_area)
            if len(history) > _TTC_HISTORY_LEN:
                history.pop(0)

            if len(history) == _TTC_HISTORY_LEN and history[0] > 0:
                expansion_ratio = history[-1] / history[0]
                if expansion_ratio > _URGENT_RATIO:
                    level = WarningLevel.URGENT
                elif expansion_ratio > _APPROACH_RATIO:
                    level = WarningLevel.APPROACH

        if level > worst:
            worst = level

    return worst


def cleanup_track_history(
    track_history: Dict[int, List[float]],
    active_track_ids: List[int],
    max_inactive: int = 60,
) -> None:
    """
    화면에서 사라진 track_id의 이력을 정리해 메모리 누수를 방지.

    Args:
        track_history:   SharedState.track_history
        active_track_ids: 현재 프레임에 존재하는 track_id 목록
        max_inactive:    이 함수 호출 주기 기준 허용 누적 횟수 (기본 60 ≈ 약 2초)
                         단순 호출 횟수 기반이 아닌 직접 제거 방식으로 구현.
    """
    active_set = set(active_track_ids)
    stale = [tid for tid in list(track_history.keys()) if tid not in active_set]
    for tid in stale:
        del track_history[tid]
