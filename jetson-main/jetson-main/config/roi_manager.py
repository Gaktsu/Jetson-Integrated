"""
ROI 설정 파일(JSON) 읽기/쓰기 전담 모듈.

live_settings_screen / roi_setup_screen 에서 'live_screen.ai' 로 참조되는
데이터 접근 책임만 분리했습니다.
"""
import json
import os

from ai.detector import load_roi_polygon
from config.settings import CAMERA_INDICES, PROJECT_ROOT


def roi_config_path(cam_id: int) -> str:
    """카메라 ID에 대응하는 ROI 설정 파일 경로를 반환합니다."""
    return os.path.join(PROJECT_ROOT, "config", f"roi_config_cam{cam_id}.json")


class RoiManager:
    """카메라별 ROI 좌표를 JSON 파일로 영속 관리하는 클래스."""

    _DEFAULT_ROI = [[220, 340], [420, 340], [420, 140], [220, 140]]

    def __init__(self) -> None:
        self.detection_enabled: bool = True

    def set_detection_enabled(self, enabled: bool) -> None:
        self.detection_enabled = enabled

    def _roi_path(self, cam_idx: int) -> str:
        cam_id = CAMERA_INDICES[cam_idx] if cam_idx < len(CAMERA_INDICES) else cam_idx
        return roi_config_path(cam_id)

    def get_roi_points(self, cam_idx: int) -> list:
        """JSON 파일에서 ROI 좌표 로드. 없으면 기본값 반환."""
        polygon = load_roi_polygon(self._roi_path(cam_idx))
        if polygon is not None:
            return polygon
        return [list(p) for p in self._DEFAULT_ROI]

    def set_roi_points(self, cam_idx: int, points: list) -> None:
        """ROI 좌표를 JSON 파일에 저장."""
        path = self._roi_path(cam_idx)
        try:
            with open(path, "w") as f:
                json.dump([[int(p[0]), int(p[1])] for p in points], f)
        except Exception as e:
            print(f"ROI 저장 실패 (cam_idx={cam_idx}): {e}")

    def cleanup(self) -> None:
        pass
