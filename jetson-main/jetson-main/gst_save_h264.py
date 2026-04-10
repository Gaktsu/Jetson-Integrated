#!/usr/bin/env python3
"""
GStreamer 기반 카메라 -> H.264 (MP4) 저장 스크립트

요구사항 반영:
- Jetson Orin Nano: 소프트웨어 인코더(x264enc) 사용
- GStreamer 파이프라인 사용
- 실시간 처리를 위해 `speed-preset=ultrafast` 또는 `veryfast`

사용 예:
  python3 gst_save_h264.py --device /dev/video0 --width 1280 --height 720 --framerate 30 --bitrate 2000 --preset ultrafast --location output.mp4 --source v4l2
  python3 gst_save_h264.py --source nvargus --width 1280 --height 720 --framerate 30 --bitrate 2000 --location output.mp4

설치(우분투/Jetson):
  sudo apt install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav

주의: mp4 파일을 정상 종료하려면 Ctrl+C 후 EOS가 전달되어야 합니다.
"""

import signal
import sys
import argparse
import gi
from gi.repository import Gst, GLib

gi.require_version('Gst', '1.0')
Gst.init(None)


def build_pipeline(args):
    # 소스 선택: v4l2 (USB) 또는 nvargus (CSI on Jetson)
    if args.source == 'nvargus':
        # nvarguscamerasrc 출력은 NVMM 메모리일 수 있으므로 nvvidconv로 시스템 메모리로 변환
        src = (
            f"nvarguscamerasrc ! video/x-raw(memory:NVMM),width={args.width},height={args.height},framerate={args.framerate}/1 "
            "! nvvidconv ! video/x-raw,format=I420"
        )
    else:
        # 기본 v4l2src
        src = (
            f"v4l2src device={args.device} ! video/x-raw,width={args.width},height={args.height},framerate={args.framerate}/1"
        )

    # 변환 -> 소프트웨어 인코더 -> 파싱 -> mp4mux -> 파일
    # x264enc: tune=zerolatency, speed-preset=ultrafast|veryfast (요구사항)
    enc = (
        f"videoconvert ! videorate ! video/x-raw,framerate={args.framerate}/1 "
        f"! x264enc tune=zerolatency speed-preset={args.preset} bitrate={args.bitrate} key-int-max={max(1, args.framerate*2)} "
        "! h264parse"
    )

    pipeline_str = (
        src + " ! " + enc + " ! mp4mux faststart=true name=mux ! filesink location=" + args.location + " sync=false"
    )

    return pipeline_str


def main():
    parser = argparse.ArgumentParser(description='GStreamer: Camera -> H.264 (MP4) saver')
    parser.add_argument('--device', default='/dev/video0', help='v4l2 device (for v4l2 source)')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--framerate', type=int, default=30)
    parser.add_argument('--bitrate', type=int, default=2000, help='kbps')
    parser.add_argument('--preset', choices=['ultrafast', 'veryfast', 'faster', 'fast', 'medium'], default='ultrafast')
    parser.add_argument('--location', default='output.mp4', help='output mp4 file')
    parser.add_argument('--source', choices=['v4l2', 'nvargus'], default='v4l2', help='camera source type')
    args = parser.parse_args()

    pipeline_str = build_pipeline(args)
    print('Pipeline:')
    print(pipeline_str)

    pipeline = Gst.parse_launch(pipeline_str)

    loop = GLib.MainLoop()

    # 메시지 처리
    bus = pipeline.get_bus()

    def on_message(bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print('ERROR:', err, dbg)
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print('EOS received')
            loop.quit()

    bus.add_signal_watch()
    bus.connect('message', on_message)

    # SIGINT 처리: 종료 시 EOS 보내서 mp4를 finalize
    def handle_sigint(signum, frame):
        print('Interrupted: sending EOS to pipeline...')
        # send EOS event
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, handle_sigint)

    # 시작
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print('Failed to start pipeline')
        sys.exit(1)

    try:
        loop.run()
    except Exception as e:
        print('Main loop exception:', e)

    # 정리
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()
