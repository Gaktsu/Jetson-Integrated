"""
UI rendering helpers.
"""
import cv2
import numpy as np
from typing import Any, List, Optional, Sequence
from datetime import datetime

from ai.detector import Detection, WarningLevel

def draw_results(
    frame: cv2.Mat,
    detections: List[Detection],
    fps: Optional[float] = None,
) -> cv2.Mat:
    """
    탐지 결과를 프레임에 직접 그립니다 (재사용 가능한 기본 시각화).

    표시 항목:
        - bounding box
        - label: "class_name confidence"  (예: "person 0.87")
        - FPS: 좌상단 (fps 전달 시)

    Args:
        frame:      입력 프레임 (in-place 수정)
        detections: Detection 리스트
        fps:        FPS 값 — None 이면 표시 안 함

    Returns:
        수정된 프레임 (동일 객체 반환)
    """
    _FONT       = cv2.FONT_HERSHEY_SIMPLEX
    _BOX_COLOR  = (0, 255, 0)      # 바운딩 박스 (BGR 초록)
    _BG_COLOR   = (0, 255, 0)      # 라벨 배경
    _TXT_COLOR  = (0, 0, 0)        # 라벨 텍스트 (검정 — 초록 배경 위 가독성)
    _FONT_SCALE = 0.55
    _THICKNESS  = 2

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, _THICKNESS)

        # 라벨 배경 크기 계산
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, 1)
        pad = 4
        label_h = th + baseline + pad * 2

        # 박스 위에 공간이 있으면 박스 바깥 위쪽, 없으면 박스 안쪽 상단
        if y1 >= label_h:
            bg_y1, bg_y2 = y1 - label_h, y1
        else:
            bg_y1, bg_y2 = y1, y1 + label_h

        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + pad * 2, bg_y2), _BG_COLOR, -1)
        cv2.putText(
            frame, label,
            (x1 + pad, bg_y2 - baseline - pad),
            _FONT, _FONT_SCALE, _TXT_COLOR, 1, cv2.LINE_AA,
        )

    # FPS 좌상단 표시
    if fps is not None:
        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (10, 30), _FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )

    return frame

def draw_detections(
    frame: cv2.Mat,
    detections: List[Detection],
    fps: float = 0.0,
    saving: bool = False,
    camera_index: int = 0,
    intrusion: bool = False,
    capture_ms: float = 0.0,
    inference_ms: float = 0.0,
    postprocess_ms: float = 0.0,
    draw_ms: float = 0.0,
    roi_polygon: Optional[Sequence[Sequence[int]]] = None,
    warning_level: Optional[WarningLevel] = None,
    forklift_speed: int = 0,
) -> cv2.Mat:
    """
    탐지 결과를 프레임에 그리기 (yolo_test-main UI 스타일 적용).

    상단 상태바: 경고 레벨별 색상 + 알람 메시지 + 속도 레벨
    ROI 오버레이: 경고 레벨 색상으로 테두리
    바운딩박스: ROI 내부 = 경고색, 외부 = 초록
    우상단: Time / FPS / Detected / 타이밍 정보
    """
    h, w = frame.shape[:2]

    # ── 경고 레벨 → 색상·메시지 매핑 (yolo_test-main 기준) ──
    # warning_level이 None이면 intrusion 플래그로 하위 호환
    if warning_level is None:
        warning_level = WarningLevel.BLIND_SPOT if intrusion else WarningLevel.SAFE

    _LEVEL_COLOR = {
        WarningLevel.SAFE:       (0, 255, 0),    # 초록
        WarningLevel.BLIND_SPOT: (0, 255, 255),  # 노랑
        WarningLevel.APPROACH:   (0, 165, 255),  # 주황
        WarningLevel.URGENT:     (0, 0, 255),    # 빨강
    }
    _LEVEL_MSG = {
        WarningLevel.SAFE:       "SAFE / NORMAL DRIVING",
        WarningLevel.BLIND_SPOT: "CAUTION: BLIND SPOT OCCUPIED",
        WarningLevel.APPROACH:   "WARNING: PERSON APPROACHING",
        WarningLevel.URGENT:     "URGENT: RAPID APPROACH! STOP!",
    }
    screen_color = _LEVEL_COLOR[warning_level]
    alarm_msg    = _LEVEL_MSG[warning_level]

    # ── 폴리곤 ROI 오버레이 (경고 레벨 색상) ──
    poly_arr = None
    if roi_polygon is not None and len(roi_polygon) >= 3:
        poly_arr = np.array(roi_polygon, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_arr], screen_color)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [poly_arr], True, screen_color, 2)

    # ── 발끝 점 + ID 그리기 (yolo_test-main 스타일 — 바운딩박스 없음) ──
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det["class_id"] == 0:  # person
            foot_x = (x1 + x2) // 2
            foot_y = y2
            is_inside = (
                poly_arr is not None and
                cv2.pointPolygonTest(poly_arr, (float(foot_x), float(foot_y)), False) >= 0
            )
            dot_color = screen_color if is_inside else (255, 0, 0)  # ROI 내부=경고색, 외부=파랑
            cv2.circle(frame, (foot_x, foot_y), 7, dot_color, -1)
            tid = det.get("track_id")
            if tid is not None:
                label = f"ID:{tid}"
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.putText(frame, label, (foot_x - tw // 2, foot_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ── 상단 상태바 (yolo_test-main 스타일) ──
    BAR_H = 55
    cv2.rectangle(frame, (0, 0), (w, BAR_H), (0, 0, 0), -1)

    # 알람 메시지 (좌측)
    cv2.putText(frame, alarm_msg, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, screen_color, 2, cv2.LINE_AA)

    # 속도 레벨 (우측)
    speed_text = f"Speed: {forklift_speed}/5"
    spd_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, speed_text, (w - spd_size[0] - 12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Saving... (속도 레벨 아래)
    # ── 카메라 번호 (상태바 바로 아래 우상단) ──
    cam_text = f"CAM {camera_index}"
    cam_sz = cv2.getTextSize(cam_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(frame, cam_text, (w - cam_sz[0] - 10, BAR_H + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if saving:
        sav_size = cv2.getTextSize("Saving...", cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(frame, "Saving...", (w - sav_size[0] - 12, BAR_H + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    return frame
