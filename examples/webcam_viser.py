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

import aprilcube
import numpy as np
from pycaas import PycaasClient
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

def main():
    parser = argparse.ArgumentParser(description="Webcam cube detection + viser 3D viz")
    parser.add_argument("--cube", required=True,
                        help="Path to config.json or model directory")
    parser.add_argument("--camera", type=str, default="rs0_color", help="pycaas stream ID")
    parser.add_argument("--no-filter", action="store_true", help="Disable Kalman filter")
    parser.add_argument("--slow", action="store_true", help="Use accurate (slower) detector")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--exposure", type=int, default=None,
                        help="RealSense exposure in microseconds (e.g. 2000). "
                             "Lower = less motion blur but darker image.")
    args = parser.parse_args()

    # Camera setup via pycaas
    client = PycaasClient()

    frame = client.get_frame(args.camera)
    if frame is None:
        print(f"Error: cannot read from {args.camera}")
        return

    ext_rot = np.load("assets/extrinsic/rotations.npy")
    ext_trans = np.load("assets/extrinsic/translations.npy")
    ext = np.eye(4) 
    ext[:3, :3] = ext_rot
    ext[:3, 3] = ext_trans

    # Debug: check intrinsic/frame resolution match
    K = client.get_intrinsics_matrix("rs0")
    print(f"Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print(f"Expected cx~{frame.shape[1]/2:.1f} cy~{frame.shape[0]/2:.1f}")

    det = aprilcube.detector(
        args.cube,
        intrinsic_cfg = K,
        extrinsic = ext,
        enable_filter=not args.no_filter,
        fast=not args.slow,
    )

    # Viser auto-renders latest process_frame result in the background
    server = det.build_viser(port=args.port)

    # Add Franka Panda robot at world origin
    urdf = load_robot_description("panda_description")
    viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/world")
    viser_urdf.update_cfg(np.zeros(len(viser_urdf.get_actuated_joint_limits())))

    print(f"Viser running at http://localhost:{args.port}")

    # Capture loop — viser visualization updates automatically
    while True:
        frame = client.get_frame(args.camera)
        if frame is None:
            break

        det.process_frame(frame)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
