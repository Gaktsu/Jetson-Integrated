"""
Video upload helper using curl multipart form.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from typing import Optional

from config.settings import (
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
