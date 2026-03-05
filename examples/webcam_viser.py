#!/usr/bin/env python3
"""Live webcam cube detection with web-based 3D visualization via viser.

Opens a viser server at http://localhost:8080 showing:
  - Textured 3D cube model with tracked pose
  - Coordinate axes for world, camera, and object
  - Camera feed with detection overlay in the GUI sidebar

Usage:
  pip install viser trimesh
  python examples/webcam_viser.py --cube models/2x2x2_30_cube
  python examples/webcam_viser.py --cube models/2x2x2_30_cube --port 8080
"""

import argparse
import time

import cv2
import aprilcube


def main():
    parser = argparse.ArgumentParser(description="Webcam cube detection + viser 3D viz")
    parser.add_argument("--cube", required=True,
                        help="Path to config.json or model directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-filter", action="store_true", help="Disable Kalman filter")
    parser.add_argument("--slow", action="store_true", help="Use accurate (slower) detector")
    parser.add_argument("--fx", type=float, default=None, help="Focal length x (pixels)")
    parser.add_argument("--fy", type=float, default=None, help="Focal length y (pixels)")
    parser.add_argument("--cx", type=float, default=None, help="Principal point x")
    parser.add_argument("--cy", type=float, default=None, help="Principal point y")
    parser.add_argument("--width", type=int, default=640,
                        help="Processing width in pixels (default: 640)")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    # Camera setup
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

    # Viser auto-renders latest process_frame result in the background
    server = det.build_viser(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    # Capture loop — viser visualization updates automatically
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale < 1.0:
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)

        det.process_frame(frame)
        time.sleep(0.001)

    cap.release()


if __name__ == "__main__":
    main()
