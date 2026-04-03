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
    - 반드시 640x480 해상도 기준으로 점을 찍어야 합니다 (메인 시스템과 동일 해상도).
    - 기존 roi_config.json이 있으면 덮어씁니다.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from config.settings import CAMERA_INDEX, PROJECT_ROOT

# 저장 경로: 프로젝트 루트 기준 config/roi_config.json
ROI_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "roi_config.json")

# 캘리브레이션 기준 해상도 (메인 시스템 표시 해상도와 반드시 일치해야 함)
CALIB_WIDTH = 640
CALIB_HEIGHT = 480

points: list = []


def _draw_roi_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    """마우스 콜백: 좌클릭 시 꼭짓점 추가"""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"[{len(points)}/4] 좌표 저장됨: ({x}, {y})")


def main() -> None:
    global points
    points = []  # 재실행 시 초기화

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[에러] 카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        print("       config/settings.py 의 CAMERA_INDEX 값을 확인하세요.")
        return

    window_name = "ROI Calibration Tool"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _draw_roi_callback)

    print("=== ROI 캘리브레이션 툴 ===")
    print(f"저장 경로: {ROI_CONFIG_PATH}")
    print("마우스 왼쪽 클릭으로 꼭짓점 4개를 순서대로 찍으세요.")
    print("'r' 키: 점 초기화 | 'q' 키: 저장 없이 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[에러] 프레임을 읽을 수 없습니다.")
            break

        # 메인 시스템과 동일한 해상도로 리사이즈한 상태에서 캘리브레이션
        frame = cv2.resize(frame, (CALIB_WIDTH, CALIB_HEIGHT))

        # 저장된 점 표시
        for p in points:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)

        # 점 사이 선 표시
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)

        # 4개 완성 시: 닫힌 폴리곤 + 안내 메시지
        if len(points) == 4:
            cv2.line(frame, points[3], points[0], (0, 255, 0), 2)

            # 반투명 폴리곤 오버레이
            overlay = frame.copy()
            pts_arr = np.array(points, np.int32)
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

            cv2.putText(
                frame, "SAVED! Press 'Q' to quit.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
            )

            # json 저장
            os.makedirs(os.path.dirname(ROI_CONFIG_PATH), exist_ok=True)
            with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"roi_polygon": points}, f, ensure_ascii=False, indent=2)
            print(f"[완료] ROI 저장됨: {ROI_CONFIG_PATH}")

        # 진행 상태 표시 (상단 안내)
        status_text = f"Points: {len(points)}/4  |  'r'=Reset  'q'=Quit"
        cv2.putText(
            frame, status_text,
            (10, CALIB_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if len(points) < 4:
                print("[종료] 4개 미만으로 종료되었습니다. roi_config.json이 저장되지 않았습니다.")
            break
        elif key == ord("r"):
            points = []
            print("[초기화] 점을 다시 찍으세요.")

        # 4개 완성 후 2초 대기 뒤 자동 종료
        if len(points) == 4:
            cv2.waitKey(2000)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
