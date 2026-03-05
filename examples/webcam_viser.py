#!/usr/bin/env python3
"""Live webcam cube detection with web-based 3D visualization via viser.

Opens a viser server at http://localhost:8080 showing:
  - Textured 3D cube model with tracked pose
  - Coordinate axes for world, camera, and object
  - Camera feed with detection overlay in the GUI sidebar

Usage:
  pip install viser trimesh
  python examples/webcam_viser.py --cube models/2x2x2_30_cube/config.json
  python examples/webcam_viser.py --cube models/2x2x2_30_cube/config.json --port 8080
"""

import argparse

import aprilcube


def main():
    parser = argparse.ArgumentParser(description="Webcam cube detection + viser 3D viz")
    parser.add_argument("--cube", required=True,
                        help="Path to config.json (model dir is inferred)")
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

    # Use rough defaults; proper intrinsics should be calibrated
    fx = args.fx or 640.0
    fy = args.fy or fx
    cx = args.cx or 320.0
    cy = args.cy or 240.0

    det = aprilcube.detector(
        args.cube,
        {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        enable_filter=not args.no_filter,
        fast=not args.slow,
    )

    server = det.build_viser(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    det.run_viser(server, camera=args.camera, width=args.width)


if __name__ == "__main__":
    main()
