"""
Video upload helper using curl multipart form.
"""
from __future__ import annotations

import os
import subprocess
import time
import threading
from datetime import datetime
from typing import Optional

from config.settings import (
    EVENT_LOG_COOLDOWN_SEC,
    EVENT_LOG_ENABLED,
    EVENT_LOG_TIMEOUT_SEC,
    EVENT_LOG_URL,
    UPLOAD_DEVICE_ID,
    UPLOAD_DEVICE_KEY,
    UPLOAD_MAX_RETRIES,
    UPLOAD_REL_DIR,
    UPLOAD_RETRY_DELAY_SEC,
    UPLOAD_TIMEOUT_SEC,
    UPLOAD_URL,
)
from utils.logger import EventType, get_logger

logger = get_logger("pipeline.uploader")

# 카메라별 마지막 전송 시각 (쿨다운 관리)
_last_event_log_ts: dict[int, float] = {}


def _extract_date(file_path: str) -> str:
    """Extract YYYY-MM-DD from event folder if possible; fallback to current date."""
    # Example folder: event_20260326_110530_gps_unknown
    base = os.path.normpath(file_path)
    parts = base.split(os.sep)
    for part in parts:
        if not part.startswith("event_"):
            continue
        tokens = part.split("_")
        if len(tokens) < 2:
            continue
        date_token = tokens[1]
        if len(date_token) == 8 and date_token.isdigit():
            return f"{date_token[0:4]}-{date_token[4:6]}-{date_token[6:8]}"
    return datetime.now().strftime("%Y-%m-%d")


def upload_video_file(file_path: str, date_str: Optional[str] = None) -> bool:
    """
    Upload video file to server with multipart/form-data.

    Form fields:
      - file
      - device_id
            - device_key
      - rel_dir
      - date
    """
    if not os.path.exists(file_path):
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "업로드 실패: 파일이 존재하지 않음",
            {"file": file_path},
        )
        return False

    if not date_str:
        date_str = _extract_date(file_path)

    for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
        cmd = [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            str(int(UPLOAD_TIMEOUT_SEC)),
            "-F",
            f"file=@{file_path}",
            "-F",
            f"device_id={UPLOAD_DEVICE_ID}",
            "-F",
            f"device_key={UPLOAD_DEVICE_KEY}",
            "-F",
            f"rel_dir={UPLOAD_REL_DIR}",
            "-F",
            f"date={date_str}",
            UPLOAD_URL,
        ]

        logger.event_info(
            EventType.MODULE_START,
            "영상 업로드 시도",
            {
                "file": file_path,
                "attempt": attempt,
                "max_retries": UPLOAD_MAX_RETRIES,
                "url": UPLOAD_URL,
            },
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(UPLOAD_TIMEOUT_SEC) + 5,
            )
        except FileNotFoundError:
            logger.event_error(
                EventType.ERROR_OCCURRED,
                "업로드 실패: curl 명령을 찾을 수 없음",
                {"file": file_path},
            )
            return False
        except subprocess.TimeoutExpired:
            result = None

        if result is not None and result.returncode == 0:
            logger.event_info(
                EventType.MODULE_STOP,
                "영상 업로드 성공",
                {"file": file_path, "attempt": attempt},
            )
            return True

        stderr = "timeout" if result is None else (result.stderr or "")[-500:]
        logger.event_warning(
            EventType.RETRY_ATTEMPT,
            "영상 업로드 실패, 재시도 예정",
            {
                "file": file_path,
                "attempt": attempt,
                "stderr": stderr,
            },
        )
        if attempt < UPLOAD_MAX_RETRIES:
            sleep_sec = UPLOAD_RETRY_DELAY_SEC * attempt
            time.sleep(sleep_sec)

    logger.event_error(
        EventType.ERROR_OCCURRED,
        "영상 업로드 최종 실패",
        {"file": file_path, "retries": UPLOAD_MAX_RETRIES},
    )
    return False


# ──────────────────────────────────────────────
# JSON 이벤트 로그 전송 (yolo_test-main send_to_ec2_server 이식)
# ──────────────────────────────────────────────

def upload_event_log(
    event_type: str,
    cam_id: int,
    speed_level: int,
    *,
    blocking: bool = False,
) -> None:
    """
    경고 이벤트 발생 시 JSON 메타데이터를 서버에 전송.

    yolo_test-main/main_system.py의 send_to_ec2_server()를 이식:
    - 영상 파일 없이 경고 레벨·시각·속도를 JSON으로 즉시 전송
    - 기본적으로 백그라운드 스레드로 실행되어 추론 루프를 차단하지 않음
    - 카메라별 쿨다운(EVENT_LOG_COOLDOWN_SEC) 으로 중복 전송 방지

    Args:
        event_type:  "BLIND_SPOT" | "APPROACH" | "URGENT"
        cam_id:      카메라 인덱스
        speed_level: 지게차 속도 레벨 (0~5)
        blocking:    True 이면 호출 스레드에서 직접 실행 (테스트용)
    """
    if not EVENT_LOG_ENABLED:
        return

    # 쿨다운 체크 (모듈 수준 dict — 스레드 간 충돌 가능성 낮아 Lock 생략)
    now = time.time()
    if now - _last_event_log_ts.get(cam_id, 0.0) < EVENT_LOG_COOLDOWN_SEC:
        return
    _last_event_log_ts[cam_id] = now

    if blocking:
        _send_event_log(event_type, cam_id, speed_level)
    else:
        threading.Thread(
            target=_send_event_log,
            args=(event_type, cam_id, speed_level),
            daemon=True,
            name=f"event_log_{cam_id}",
        ).start()


def _send_event_log(event_type: str, cam_id: int, speed_level: int) -> None:
    """실제 HTTP POST 전송 (백그라운드 스레드에서 호출)."""
    import json
    import urllib.request
    import urllib.error

    payload = {
        "device_id": UPLOAD_DEVICE_ID,
        "event_type": event_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cam_id": cam_id,
        "speed_level": speed_level,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            EVENT_LOG_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=EVENT_LOG_TIMEOUT_SEC):
            pass
        logger.event_info(
            EventType.MODULE_STOP,
            "이벤트 로그 전송 성공",
            {"event_type": event_type, "cam_id": cam_id, "payload": payload},
        )
    except Exception as e:
        logger.event_error(
            EventType.ERROR_OCCURRED,
            "이벤트 로그 전송 실패 (서버 미연결 시 정상)",
            {"event_type": event_type, "cam_id": cam_id, "error": str(e)},
        )
