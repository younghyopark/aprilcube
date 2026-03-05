#!/usr/bin/env python3
"""Record video + camera calibration for offline benchmarking.

Usage:
  python examples/record.py --camera rs0_color --output recording/session1
  python examples/record.py --camera rs0_color --output recording/session1 --duration 30
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from pycaas import PycaasClient


def main():
    parser = argparse.ArgumentParser(description="Record video for offline benchmark")
    parser.add_argument("--camera", type=str, default="rs0_color", help="pycaas stream ID")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--duration", type=float, default=0, help="Recording duration in seconds (0 = until Ctrl+C)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    client = PycaasClient()
    frame = client.get_frame(args.camera)
    if frame is None:
        print(f"Error: cannot read from {args.camera}")
        return

    h, w = frame.shape[:2]
    K = client.get_intrinsics_matrix("rs0")

    # Save intrinsics
    calib = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "resolution": [w, h],
    }
    with open(os.path.join(args.output, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)

    # Save extrinsic if available
    ext_rot_path = "assets/extrinsic/rotations.npy"
    ext_trans_path = "assets/extrinsic/translations.npy"
    if os.path.exists(ext_rot_path) and os.path.exists(ext_trans_path):
        ext = np.eye(4)
        ext[:3, :3] = np.load(ext_rot_path)
        ext[:3, 3] = np.load(ext_trans_path)
        np.save(os.path.join(args.output, "extrinsic.npy"), ext)
        print("Saved extrinsic.")

    # Video writer
    video_path = os.path.join(args.output, "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

    # Timestamps
    timestamps = []

    print(f"Recording {w}x{h} to {video_path}")
    if args.duration > 0:
        print(f"Duration: {args.duration}s")
    print("Press Ctrl+C to stop.")

    t0 = time.monotonic()
    n_frames = 0
    try:
        while True:
            frame = client.get_frame(args.camera)
            if frame is None:
                continue
            ts = time.monotonic()
            writer.write(frame)
            timestamps.append(ts - t0)
            n_frames += 1

            elapsed = ts - t0
            if n_frames % 30 == 0:
                print(f"\r  {elapsed:.1f}s  {n_frames} frames  ({n_frames/elapsed:.1f} fps)", end="", flush=True)

            if args.duration > 0 and elapsed >= args.duration:
                break
    except KeyboardInterrupt:
        pass

    writer.release()
    np.save(os.path.join(args.output, "timestamps.npy"), np.array(timestamps))
    print(f"\nSaved {n_frames} frames to {args.output}/")


if __name__ == "__main__":
    main()
