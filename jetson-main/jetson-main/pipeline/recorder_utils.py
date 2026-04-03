"""
Recorder helper functions extracted from legacy vision.yolo_infer.
"""
from __future__ import annotations

import cv2
import os
import shutil
import stat
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from config.settings import UPLOAD_ENABLED
from pipeline.uploader import upload_video_file
from utils.logger import get_logger, EventType

logger = get_logger("pipeline.recorder_utils")

def _transcode_to_h264(file_path: str) -> bool:
    """
    저장된 mp4v 파일을 H.264 코덱으로 변환 (ffmpeg 사용).
    변환 성공 시 원본(mp4v) 파일을 삭제하고 H.264 파일로 교체합니다.
    변환 실패 시 임시 파일을 삭제하고 원본을 보존합니다.

    Args:
        file_path: 변환할 mp4 파일 경로 (mp4v 코덱)
    """
    tmp_path = file_path.replace(".mp4", "_h264_tmp.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", file_path,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-movflags", "+faststart",
        tmp_path
    ]
    try:
        logger.event_info(
            EventType.MODULE_START,
            "H.264 변환 시작",
            {"original_file": file_path, "tmp_file": tmp_path}
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            # 1단계: 원본(mp4v) 파일 명시적 삭제
            try:
                os.remove(file_path)
                logger.event_info(
                    EventType.MODULE_STOP,
                    "원본 mp4v 파일 삭제 완료",
                    {"file": file_path}
                )
            except Exception as e:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "원본 mp4v 파일 삭제 실패",
                    {"file": file_path, "error": str(e)}
                )
                # 원본 삭제 실패 시 임시 파일도 정리
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                return False

            # 2단계: H.264 임시 파일을 원본 경로로 이동
            try:
                os.rename(tmp_path, file_path)
                # 3단계: chmod +x (소유자/그룹/기타 실행 권한 추가)
                current = stat.S_IMODE(os.stat(file_path).st_mode)
                os.chmod(file_path, current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                logger.event_info(
                    EventType.MODULE_STOP,
                    "H.264 변환 완료 (원본 교체됨, chmod +x 적용)",
                    {"file": file_path}
                )
                return True
            except Exception as e:
                logger.event_error(
                    EventType.ERROR_OCCURRED,
                    "H.264 파일 이동 실패 (원본은 이미 삭제됨)",
                    {"tmp_file": tmp_path, "dest": file_path, "error": str(e)}
                )
                return False
        else:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "H.264 변환 실패 (ffmpeg 오류) - 원본 파일 보존",
                {"file": file_path, "stderr": result.stderr[-500:]}
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    except subprocess.TimeoutExpired:
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "H.264 변환 타임아웃 (300초 초과) - 원본 파일 보존",
            {"file": file_path}
        )
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False
    except Exception as e:
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "H.264 변환 중 예외 - 원본 파일 보존",
            {"file": file_path, "error": str(e)}
        )
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False


def _transcode_and_upload(file_path: str) -> None:
    """H.264 변환 후 즉시 업로드를 수행한다."""
    ok = _transcode_to_h264(file_path)
    if not ok:
        return
    if not UPLOAD_ENABLED:
        return
    upload_video_file(file_path)


def _create_event_folder(
    save_dir: str,
    timestamp: float,
    sensor_data: Optional[Dict[str, Any]]
) -> str:
    """
    이벤트별 폴더 생성
    
    Args:
        save_dir: 기본 저장 디렉토리
        timestamp: 이벤트 시작 타임스탬프
        sensor_data: 센서 데이터
    
    Returns:
        생성된 폴더 경로
    """
    time_tag = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
    gps_tag = _format_gps_tag(sensor_data)
    folder_name = f"event_{time_tag}_{gps_tag}"
    folder_path = os.path.join(save_dir, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    logger.event_info(
        EventType.MODULE_START,
        "이벤트 폴더 생성",
        {"path": folder_path}
    )
    
    return folder_path


def _cleanup_old_folders(
    parent_dir: str,
    max_folders: int,
    is_full_mode: bool = False
) -> None:
    """
    오래된 폴더 정리
    
    Args:
        parent_dir: 부모 디렉토리 (event 모드: SaveVideos, full 모드: SaveVideos/full_recording)
        max_folders: 최대 폴더 개수 (0이면 정리하지 않음)
        is_full_mode: full 모드 여부
    """
    if max_folders <= 0:
        return
    
    try:
        # event_로 시작하는 폴더만 필터링
        folders = []
        for item in os.listdir(parent_dir):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path) and item.startswith("event_"):
                folders.append(item_path)
        
        # 폴더가 최대 개수를 초과하면 오래된 것부터 삭제
        if len(folders) > max_folders:
            # 생성 시간 기준 정렬 (오래된 것부터)
            folders.sort(key=lambda x: os.path.getctime(x))
            
            # 초과된 폴더 삭제
            folders_to_delete = folders[:len(folders) - max_folders]
            mode_name = "full" if is_full_mode else "event"
            
            for folder in folders_to_delete:
                try:
                    shutil.rmtree(folder)
                    logger.event_info(
                        EventType.MODULE_STOP,
                        f"{mode_name} 모드 오래된 폴더 삭제",
                        {"path": folder}
                    )
                except Exception as e:
                    logger.event_error(
                        EventType.ERROR_OCCURRED,
                        f"{mode_name} 모드 폴더 삭제 실패",
                        {"path": folder, "error": str(e)}
                    )
    except Exception as e:
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "폴더 정리 중 오류",
            {"parent_dir": parent_dir, "error": str(e)}
        )


def _create_writer(
    save_dir: str,
    cam_id: int,
    timestamp: float,
    frame: cv2.Mat,
    fps_map: dict[int, float],
    codec: str,
    sensor_data: Optional[Dict[str, Any]],
    event_folder: Optional[str] = None
) -> Optional[Tuple[cv2.VideoWriter, str]]:
    try:
        h, w = frame.shape[:2]
        fps = fps_map.get(cam_id, 30.0) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        logger.event_info(
            EventType.MODULE_START,
            "_create_writer 시작",
            {"cam_id": cam_id, "fps": fps, "codec": codec, "frame_size": (w, h), "event_folder": event_folder}
        )
        
        file_path = _build_event_filename(save_dir, cam_id, timestamp, sensor_data, event_folder)
        
        logger.event_info(
            EventType.MODULE_START,
            "파일 경로 생성 완료",
            {"file_path": file_path}
        )
        
        writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
        
        logger.event_info(
            EventType.MODULE_START,
            "VideoWriter 객체 생성 완료",
            {"isOpened": writer.isOpened() if writer else False}
        )
        
        if not writer.isOpened():
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "영상 저장기 생성 실패",
                {"camera": cam_id, "path": file_path}
            )
            return None
        
        logger.event_info(
            EventType.MODULE_START,
            "_create_writer 성공",
            {"cam_id": cam_id, "path": file_path}
        )
        
        return writer, file_path
    except Exception as e:
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "_create_writer 예외 발생",
            {"cam_id": cam_id, "error": str(e), "error_type": type(e).__name__}
        )
        return None


def _build_event_filename(
    save_dir: str,
    cam_id: int,
    event_ts: float,
    sensor_data: Optional[Dict[str, Any]],
    event_folder: Optional[str] = None
) -> str:
    """
    영상 파일 경로 생성
    
    Args:
        save_dir: 기본 저장 디렉토리
        cam_id: 카메라 ID
        event_ts: 이벤트 타임스탬프
        sensor_data: 센서 데이터
        event_folder: 이벤트 폴더 경로 (있으면 해당 폴더에 저장)
    
    Returns:
        파일 경로
    """
    if event_folder:
        # full_recording 폴더인 경우: 타임스탬프 포함
        if event_folder.endswith("full_recording"):
            time_tag = datetime.fromtimestamp(event_ts).strftime("%Y%m%d_%H%M%S")
            filename = f"camera{cam_id}_{time_tag}.mp4"
            return os.path.join(event_folder, filename)
        # 이벤트 폴더인 경우: 간단한 파일명
        else:
            filename = f"camera{cam_id}.mp4"
            return os.path.join(event_folder, filename)
    else:
        # 이벤트 폴더 없으면 기존 방식 (타임스탬프+GPS)
        time_tag = datetime.fromtimestamp(event_ts).strftime("%Y%m%d_%H%M%S")
        gps_tag = _format_gps_tag(sensor_data)
        filename = f"camera{cam_id}_{time_tag}_{gps_tag}.mp4"
        return os.path.join(save_dir, filename)


def _format_gps_tag(sensor_data: Optional[Dict[str, Any]]) -> str:
    if not sensor_data:
        return "gps_unknown"
    gps_data = sensor_data.get("gps")
    if not gps_data:
        return "gps_unknown"

    raw = gps_data.get("data") if isinstance(gps_data, dict) else None
    if not isinstance(raw, str):
        return "gps_unknown"

    coords = _parse_nmea_lat_lon(raw)
    if coords is None:
        return "gps_unknown"

    lat, lon = coords
    tag = f"lat{lat:.6f}_lon{lon:.6f}"
    return _sanitize_tag(tag)


def _parse_nmea_lat_lon(sentence: str) -> Optional[Tuple[float, float]]:
    if not sentence or not sentence.startswith("$"):
        return None

    parts = sentence.split(",")
    if not parts:
        return None

    msg = parts[0]
    if msg in {"$GPRMC", "$GNRMC"}:
        if len(parts) < 7 or parts[2] != "A":
            return None
        lat_raw, ns = parts[3], parts[4]
        lon_raw, ew = parts[5], parts[6]
    elif msg in {"$GPGGA", "$GNGGA"}:
        if len(parts) < 6 or parts[6] == "0":
            return None
        lat_raw, ns = parts[2], parts[3]
        lon_raw, ew = parts[4], parts[5]
    else:
        return None

    lat = _nmea_to_decimal(lat_raw, ns, is_lat=True)
    lon = _nmea_to_decimal(lon_raw, ew, is_lat=False)
    if lat is None or lon is None:
        return None
    return lat, lon


def _nmea_to_decimal(value: str, hemi: str, is_lat: bool) -> Optional[float]:
    if not value or not hemi:
        return None

    try:
        degrees_len = 2 if is_lat else 3
        degrees = float(value[:degrees_len])
        minutes = float(value[degrees_len:])
    except ValueError:
        return None

    decimal = degrees + (minutes / 60.0)
    if hemi.upper() in {"S", "W"}:
        decimal = -decimal
    return decimal


def _sanitize_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
