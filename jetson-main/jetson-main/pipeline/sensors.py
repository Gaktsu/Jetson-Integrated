"""
Sensor thread helpers for GPS/IMU polling.
"""
from __future__ import annotations

import threading
import time
from typing import List

from hardware.gps import GPS
from hardware.imu import IMU
from utils.logger import get_logger
from utils.sensor_sync import SensorBuffer

logger = get_logger("pipeline.sensors")


def start_sensor_threads(
    gps: GPS,
    imu: IMU,
    stop_event: threading.Event,
    gps_buffer: SensorBuffer,
    imu_buffer: SensorBuffer,
) -> List[threading.Thread]:
    """GPS/IMU 폴링 스레드 시작. 하드웨어 없으면 해당 스레드 생략"""
    threads = []

    def gps_loop():
        while not stop_event.is_set():
            data = gps.read_data()
            if data is not None:
                gps_buffer.add(time.time(), data)
            time.sleep(1.0)

    def imu_loop():
        while not stop_event.is_set():
            data = imu.read_data()
            if data is not None:
                imu_buffer.add(time.time(), data)
            time.sleep(1.0)

    if gps.start():
        t = threading.Thread(target=gps_loop, daemon=True, name="gps_worker")
        t.start()
        threads.append(t)
        logger.debug("GPS 스레드 시작")
    if imu.start():
        t = threading.Thread(target=imu_loop, daemon=True, name="imu_worker")
        t.start()
        threads.append(t)
        logger.debug("IMU 스레드 시작")
    return threads
