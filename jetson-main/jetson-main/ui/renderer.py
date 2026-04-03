"""
UI rendering helpers.
"""
import cv2
import numpy as np
from typing import Any, List, Optional, Sequence
from datetime import datetime

from ai.detector import Detection

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
) -> cv2.Mat:
    """
    탐지 결과를 프레임에 그리기
    """
    h, w = frame.shape[:2]

    # ── 폴리곤 ROI 오버레이 ──
    # roi_setup.py 에서 설정한 영역을 반투명으로 표시
    # 침입 시 빨강, 평상시 초록
    poly_arr = None
    if roi_polygon is not None and len(roi_polygon) >= 3:
        poly_arr = np.array(roi_polygon, dtype=np.int32)
        roi_color = (0, 0, 255) if intrusion else (0, 255, 0)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly_arr], roi_color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
        cv2.polylines(frame, [poly_arr], True, roi_color, 2)

    # ── 바운딩 박스 그리기 ──
    # 발끝(foot-point)이 폴리곤 내부인지 개별 판별해 색상 결정
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det["class_id"] == 0:  # person
            foot_x = (x1 + x2) // 2
            foot_y = y2
            is_inside = (
                poly_arr is not None and
                cv2.pointPolygonTest(poly_arr, (float(foot_x), float(foot_y)), False) >= 0
            )
            color = (0, 0, 255) if is_inside else (0, 255, 0)
            thickness = 3 if is_inside else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            text = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(frame, text, (x1, max(0, y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Warning 문구
    if intrusion and not saving:
        warning_text = "WARNING!"
        font_scale = 2.0
        thickness = 4
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.rectangle(frame, (text_x - 20, text_y - text_size[1] - 20),
                     (text_x + text_size[0] + 20, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, warning_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    # 우상단 정보
    font_scale = 0.6
    thickness = 2
    y_offset = 30
    
    if saving:
        text = "Saving..."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x_pos = w - text_size[0] - 10
        cv2.putText(frame, text, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    else:
        color = (0, 255, 255)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 시간
        text_time = f"Time: {current_time}"
        text_size = cv2.getTextSize(text_time, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x_pos = w - text_size[0] - 10
        cv2.putText(frame, text_time, (x_pos, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        # FPS
        text_fps = f"FPS: {fps:.1f}"
        text_size = cv2.getTextSize(text_fps, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x_pos = w - text_size[0] - 10
        cv2.putText(frame, text_fps, (x_pos, y_offset + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        # 탐지 수
        text_count = f"Detected: {len(detections)}"
        text_size = cv2.getTextSize(text_count, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x_pos = w - text_size[0] - 10
        cv2.putText(frame, text_count, (x_pos, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
        # 카메라 번호
        text_camera = f"Camera: {camera_index}"
        text_size = cv2.getTextSize(text_camera, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x_pos = w - text_size[0] - 10
        cv2.putText(frame, text_camera, (x_pos, y_offset + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        # 타이밍
        timing_color = (180, 255, 180)
        for i, t_text in enumerate([
            f"cap:     {capture_ms:5.1f} ms",
            f"infer:   {inference_ms:5.1f} ms",
            f"post:    {postprocess_ms:5.1f} ms",
            f"draw:    {draw_ms:5.1f} ms",
        ]):
            text_size = cv2.getTextSize(t_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x_pos = w - text_size[0] - 10
            cv2.putText(frame, t_text, (x_pos, y_offset + 120 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, timing_color, thickness, cv2.LINE_AA)
    
    # 좌하단 메뉴 (항상 표시)
    menu_font_scale = 0.5
    menu_thickness = 1
    menu_color = (255, 255, 255)
    menu_y_start = h - 50
    
    cv2.putText(frame, "[Q] Exit", (10, menu_y_start),
               cv2.FONT_HERSHEY_SIMPLEX, menu_font_scale, menu_color, menu_thickness, cv2.LINE_AA)
    cv2.putText(frame, "[C] Switch Camera", (10, menu_y_start + 20),
               cv2.FONT_HERSHEY_SIMPLEX, menu_font_scale, menu_color, menu_thickness, cv2.LINE_AA)
    
    return frame
