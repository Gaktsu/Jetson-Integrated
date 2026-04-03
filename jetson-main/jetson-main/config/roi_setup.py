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

from config.settings import CAMERA_INDEX, CAMERA_INDICES, PROJECT_ROOT

# 카메라별 ROI 설정 파일 경로
def roi_config_path(cam_index: int) -> str:
    """카메라 인덱스별 ROI 저장 경로 반환"""
    return os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_index}.json")

# 현재 활성 카메라의 점 목록 (모듈 레벨 — 마우스 콜백과 공유)
points: list = []


def _draw_roi_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
    """마우스 콜백: 좌클릭 시 현재 카메라의 꼭짓점 추가"""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"[{len(points)}/4] 좌표 저장됨: ({x}, {y})")


def main(frame_getters=None, window_config=None) -> None:
    """ROI 캘리브레이션 도구 실행.

    Args:
        frame_getters: {cam_index: callable} 형태의 dict.
                       callable은 최신 프레임(numpy array)을 반환.
                       None이면 카메라를 직접 열어 사용 (단독 실행 모드).
                       main.py에서 호출 시 SharedState 프레임을 전달해 카메라 충돌 방지.
        window_config: main.py 창 설정 dict. 아래 키를 포함.
                       {'fullscreen': bool, 'width': int, 'height': int}
                       None이면 기본 창 모드로 실행.
    """
    global points

    # ── 사용할 카메라 목록 결정 ──
    if frame_getters is not None:
        cam_list = sorted(frame_getters.keys())
    else:
        cam_list = list(CAMERA_INDICES)

    if not cam_list:
        print("[에러] 사용 가능한 카메라가 없습니다.")
        return

    # 카메라별 점 저장소 초기화
    all_points = {idx: [] for idx in cam_list}
    cur_cam = cam_list[0]
    points = all_points[cur_cam]  # 콜백과 공유되는 참조

    # 단독 모드 카메라 핸들
    cap = None
    frame_h: int = 0

    def open_cap(cam_index: int) -> bool:
        """단독 실행 모드에서 카메라 열기 (기존 캡처 해제 후 재오픈)"""
        nonlocal cap, frame_h
        if cap is not None:
            cap.release()
            cap = None
        c = cv2.VideoCapture(cam_index)
        if not c.isOpened():
            print(f"[에러] 카메라 {cam_index}를 열 수 없습니다.")
            return False
        cap = c
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f"카메라 {cam_index} 해상도: {fw}x{frame_h}")
        return True

    # 단독 모드 초기 카메라 열기
    if frame_getters is None:
        if not open_cap(cur_cam):
            return

    window_name = "ROI Calibration Tool"
    # main.py의 창 설정을 그대로 적용
    if window_config is not None and window_config.get("fullscreen"):
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif window_config is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_config["width"], window_config["height"])
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _draw_roi_callback)

    # 키 안내 출력
    key_hint = "  ".join([f"'{i+1}'=CAM{c}" for i, c in enumerate(cam_list)])
    print("=== ROI 캘리브레이션 툴 ===")
    print(f"카메라 전환: {key_hint}")
    print(f"저장 경로: config/roi_config_cam{{index}}.json")
    print("'r': 현재 카메라 점 초기화 | 4점 완성 후 'q': 전체 저장 및 종료 | 4점 미만 'q': 저장 없이 종료")

    while True:
        # ── 프레임 획득 ──
        if frame_getters is not None:
            frame = frame_getters[cur_cam]()
            if frame is None:
                cv2.waitKey(1)
                continue
            if frame_h == 0:
                frame_h = frame.shape[0]
                print(f"카메라 {cur_cam} 해상도: {frame.shape[1]}x{frame_h}")
        else:
            ret, frame = cap.read()
            if not ret:
                print("[에러] 프레임을 읽을 수 없습니다.")
                break

        frame = frame.copy()

        # ── 현재 카메라의 점 그리기 ──
        for p in points:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], (0, 255, 0), 2)
        if len(points) == 4:
            cv2.line(frame, points[3], points[0], (0, 255, 0), 2)
            overlay = frame.copy()
            pts_arr = np.array(points, np.int32)
            cv2.fillPoly(overlay, [pts_arr], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
            cv2.putText(
                frame, "4 points ready. Press 'Q' to SAVE & quit.",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2,
            )

        # ── 하단 상태바 ──
        saved_str = ", ".join(
            [f"CAM{idx}" for idx, pts in all_points.items() if len(pts) == 4]
        ) or "없음"
        cam_keys = "  ".join([f"[{i+1}]CAM{c}" for i, c in enumerate(cam_list)])
        status = (
            f"현재: CAM{cur_cam}  |  Points: {len(points)}/4  |  "
            f"저장완료: {saved_str}  |  {cam_keys}  |  [r]초기화  [q]저장종료"
        )
        if frame_h > 0:
            cv2.putText(
                frame, status,
                (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        # ── 숫자키로 카메라 전환 ──
        switched = False
        for i, cam_index in enumerate(cam_list):
            if key == ord(str(i + 1)) and cur_cam != cam_index:
                all_points[cur_cam] = list(points)  # 현재 점 저장
                cur_cam = cam_index
                points = all_points[cur_cam]         # 전환할 카메라 점 로드
                frame_h = 0                          # 해상도 재결정
                if frame_getters is None:
                    open_cap(cur_cam)
                print(f"[전환] 카메라 {cur_cam}")
                switched = True
                break

        if switched:
            continue

        # ── 기타 키 처리 ──
        if key == ord("r"):
            points.clear()
            all_points[cur_cam] = points
            print(f"[초기화] CAM{cur_cam} 점을 다시 찍으세요.")
        elif key == ord("q"):
            all_points[cur_cam] = list(points)  # 현재 카메라 점 확정
            saved = []
            for cam_index, pts in all_points.items():
                if len(pts) == 4:
                    path = roi_config_path(cam_index)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"roi_polygon": pts}, f, ensure_ascii=False, indent=2)
                    saved.append(cam_index)
                    print(f"[완료] CAM{cam_index} ROI 저장됨: {path}")
            if not saved:
                print("[종료] 저장된 ROI가 없습니다. (4개 미만인 카메라만 있음)")
            break

    if cap is not None:
        cap.release()
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
