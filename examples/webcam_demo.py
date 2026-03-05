#!/usr/bin/env python3
"""Live webcam cube detection demo.

Usage:
  python examples/webcam_demo.py --cube path/to/config.json
  python examples/webcam_demo.py --cube path/to/config.json --no-filter
  python examples/webcam_demo.py --cube path/to/config.json --fx 900 --fy 900
"""

import argparse
import time

import cv2
import numpy as np
import aprilcube


def main():
    parser = argparse.ArgumentParser(description="Webcam cube detection demo")
    parser.add_argument("--cube", required=True, help="Path to config.json")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-filter", action="store_true", help="Disable Kalman filter")
    parser.add_argument("--slow", action="store_true", help="Use accurate (slower) detector")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y")
    parser.add_argument("--width", type=int, default=640,
                        help="Processing width in pixels (default: 640)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read from camera")
        return

    raw_h, raw_w = frame.shape[:2]
    scale = args.width / raw_w if raw_w > args.width else 1.0
    proc_w = int(raw_w * scale)
    proc_h = int(raw_h * scale)

    fx = (args.fx or float(raw_w)) * scale
    fy = (args.fy or (args.fx or float(raw_w))) * scale
    cx = (args.cx or raw_w / 2.0) * scale
    cy = (args.cy or raw_h / 2.0) * scale

    det = aprilcube.detector(
        args.cube,
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        enable_filter=not args.no_filter,
        fast=not args.slow,
    )

    print(f"Camera {args.camera}: {raw_w}x{raw_h} -> {proc_w}x{proc_h}, "
          f"fx={fx:.0f} fy={fy:.0f}")
    print("Press 'q' or ESC to quit.")

    fps_t = time.time()
    fps_n = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        result = det.process_frame(frame)
        vis = det.draw_result(frame, result)

        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            fps = fps_n / elapsed
            fps_n = 0
            fps_t = time.time()

        cv2.putText(vis, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("aprilcube", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
