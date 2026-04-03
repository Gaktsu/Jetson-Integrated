"""
ROI 캘리브레이션 도구 (일회성 실행 스크립트)

실행 방법:
    python -m config.roi_setup        (프로젝트 루트에서)
    또는
    python config/roi_setup.py

사용법:
    1. 카메라 영상이 뜨면 마우스 왼쪽 클릭으로 꼭짓점 4개를 순서대로 찍습니다.
    2. 4개가 완성되면 자동으로 config/roi_config.json에 저장하고 종료됩니다.
    3. 저장된 좌표는 pipeline/inference.py에서 ROI 폴리곤으로 사용됩니다.

주의:
    - 화면 해상도는 카메라 실제 출력값을 자동으로 읽어 사용합니다 (main.py와 동일 기준).
    - 기존 roi_config.json이 있으면 덮어씁니다.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

# 어느 디렉터리에서 실행하든 프로젝트 루트를 sys.path에 추가
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import CAMERA_INDEX, PROJECT_ROOT

# 저장 경로: 프로젝트 루트 기준 config/roi_config.json
ROI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "roi_config.json")

points: list = []


def _draw_roi_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    """마우스 콜백: 좌클릭 시 꼭짓점 추가"""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"[{len(points)}/4] 좌표 저장됨: ({x}, {y})")


def main(frame_getter=None) -> None:
    """ROI 캘리브레이션 도구 실행.

    Args:
        frame_getter: 호출 시 최신 프레임(numpy array)을 반환하는 콜백.
                      None이면 카메라를 직접 열어 사용 (단독 실행 모드).
                      main.py에서 호출 시 SharedState 프레임을 전달해 카메라 충돌을 방지.
    """
    global points
    points = []  # 재실행 시 초기화

    cap = None
    frame_w: int = 0
    frame_h: int = 0

    if frame_getter is None:
        # ── 단독 실행 모드: 카메라 직접 열기 ──
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
            print("       config/settings.py 의 CAMERA_INDEX 값을 확인하세요.")
            return
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"카메라 해상도: {frame_w}x{frame_h}")
    # frame_getter 모드일 땐 첫 프레임을 받은 뒤 해상도를 결정

    window_name = "ROI Calibration Tool"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _draw_roi_callback)

    print("=== ROI 캘리브레이션 툴 ===")
    print(f"저장 경로: {ROI_CONFIG_PATH}")
    print("마우스 왼쪽 클릭으로 꼭짓점 4개를 순서대로 찍으세요.")
    print("'r' 키: 점 초기화 | 4개 완성 후 'q': 저장 및 종료 | 4개 미만 'q': 저장 없이 종료")

    while True:
        if frame_getter is not None:
            # ── main.py 연동 모드: SharedState에서 프레임 읽기 ──
            frame = frame_getter()
            if frame is None:
                cv2.waitKey(1)
                continue
            # 첫 유효 프레임에서 해상도 결정
            if frame_h == 0:
                frame_h, frame_w = frame.shape[:2]
                print(f"카메라 해상도: {frame_w}x{frame_h} (main.py 표시 기준과 동일)")
        else:
            # ── 단독 실행 모드: 카메라에서 직접 읽기 ──
            ret, frame = cap.read()
            if not ret:
                print("[에러] 프레임을 읽을 수 없습니다.")
                break

        # 저장된 점 표시
        for p in points:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)

        # 점 사이 선 표시
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

        # 4개 완성 시: 닫힌 폴리곤 + 안내 메시지 (저장은 q키로만)
        if len(points) == 4:
            cv2.line(frame, points[3], points[0], (0, 255, 0), 2)

            # 반투명 폴리곤 오버레이
            overlay = frame.copy()
            pts_arr = np.array(points, np.int32)
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            cv2.putText(
                frame, "4 points ready. Press 'Q' to SAVE & quit.",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2,
            )

        # 진행 상태 표시 (하단 안내)
        status_text = f"Points: {len(points)}/4  |  'r'=Reset  'q'=Quit"
        cv2.putText(
            frame, status_text,
            (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if len(points) == 4:
                # 4개 완성 상태에서만 저장
                os.makedirs(os.path.dirname(ROI_CONFIG_PATH), exist_ok=True)
                with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump({"roi_polygon": points}, f, ensure_ascii=False, indent=2)
                print(f"[완료] ROI 저장됨: {ROI_CONFIG_PATH}")
            else:
                print(f"[종료] {len(points)}개 미만으로 종료되었습니다. roi_config.json이 저장되지 않았습니다.")
            break
        elif key == ord("r"):
            points = []
            print("[초기화] 점을 다시 찍으세요.")

        # (자동 종료 없음 — q키로만 종료)

    if cap is not None:
        cap.release()
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
