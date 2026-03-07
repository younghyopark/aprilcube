#!/usr/bin/env python3
"""Simple webcam cube detection with OpenCV window.

Opens the first available webcam and shows live detection overlay in a cv2 window.
No dependencies on pycaas or viser — just opencv-python and numpy.

Usage:
  python examples/webcam_simple.py --cube models/2x2x2_30_cube
  python examples/webcam_simple.py --cube models/2x2x2_30_cube --camera 1
  python examples/webcam_simple.py --cube models/2x2x2_30_cube --intrinsics calib.json
"""

import argparse

import aprilcube
import cv2


def main():
    parser = argparse.ArgumentParser(description="Simple webcam cube detection")
    parser.add_argument("--cube", required=True,
                        help="Path to config.json or model directory")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--intrinsics", default=None,
                        help="Path to intrinsics JSON (estimated from frame if omitted)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable Kalman filter")
    parser.add_argument("--slow", action="store_true",
                        help="Use accurate (slower) detector")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read from camera")
        cap.release()
        return

    h, w = frame.shape[:2]

    # Use provided intrinsics or estimate from frame size
    if args.intrinsics:
        intrinsics = args.intrinsics
    else:
        fx = fy = max(w, h)
        intrinsics = {"fx": fx, "fy": fy, "cx": w / 2.0, "cy": h / 2.0}
        print(f"No intrinsics provided, estimating: fx={fx} cx={w/2:.0f} cy={h/2:.0f}")

    det = aprilcube.detector(
        args.cube,
        intrinsic_cfg=intrinsics,
        enable_filter=not args.no_filter,
        fast=not args.slow,
    )

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = det.process_frame(frame)
        vis = det.draw_result(frame, result)

        cv2.imshow("AprilCube Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
