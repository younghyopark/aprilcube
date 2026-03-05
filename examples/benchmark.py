#!/usr/bin/env python3
"""Offline benchmark for cube detection quality.

Reads a recorded session (from record.py) and evaluates detection in both
pixel space (2D) and 3D pose space.

Usage:
  python examples/benchmark.py --cube models/2x2x2_30_cube --recording recording/session1
  python examples/benchmark.py --cube models/2x2x2_30_cube --recording recording/session1 --slow --save-video
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

import aprilcube


def angle_between_rotations(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    R_diff = R1.T @ R2
    cos_val = (np.trace(R_diff) - 1) / 2
    return float(np.arccos(np.clip(cos_val, -1, 1)))


def main():
    parser = argparse.ArgumentParser(description="Offline detection benchmark")
    parser.add_argument("--cube", required=True, help="Path to config.json or model directory")
    parser.add_argument("--recording", required=True, help="Recording directory from record.py")
    parser.add_argument("--no-filter", action="store_true", help="Disable Kalman filter")
    parser.add_argument("--slow", action="store_true", help="Use accurate (slower) detector")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    args = parser.parse_args()

    # Load calibration
    calib_path = os.path.join(args.recording, "calibration.json")
    with open(calib_path) as f:
        calib = json.load(f)
    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)

    # Load extrinsic if available
    ext_path = os.path.join(args.recording, "extrinsic.npy")
    ext = np.load(ext_path) if os.path.exists(ext_path) else None

    # Load timestamps
    ts_path = os.path.join(args.recording, "timestamps.npy")
    timestamps = np.load(ts_path) if os.path.exists(ts_path) else None

    # Open video
    video_path = os.path.join(args.recording, "video.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create detector
    det = aprilcube.detector(
        args.cube,
        intrinsic_cfg=K,
        dist_coeffs=dist,
        extrinsic=ext,
        enable_filter=not args.no_filter,
        fast=not args.slow,
    )

    # Output video writer
    out_writer = None
    if args.save_video:
        out_path = os.path.join(args.recording, "benchmark_output.mp4")
        out_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --- Per-frame collection ---
    frame_idx = 0
    results_log = []      # per-frame dict

    # 3D pose sequences (for jitter / smoothness)
    pose_tvecs = []       # (N, 3) mm
    pose_rvecs = []       # (N, 3)
    pose_times = []       # timestamps
    pose_frame_ids = []   # which frames had pose

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = timestamps[frame_idx] if timestamps is not None and frame_idx < len(timestamps) else frame_idx / fps
        result = det.process_frame(frame, timestamp=ts)

        # Log
        entry = {
            "frame": frame_idx,
            "timestamp": ts,
            "success": result["success"],
            "predicted": result.get("predicted", False),
            "n_tags": result["n_tags"],
            "n_inliers": result["n_inliers"],
            "reproj_error": result["reproj_error"] if result["success"] else None,
            "tag_ids": result["tag_ids"],
        }
        results_log.append(entry)

        if result["success"]:
            tvec = result["tvec"].flatten()
            rvec = result["rvec"].flatten()
            pose_tvecs.append(tvec)
            pose_rvecs.append(rvec)
            pose_times.append(ts)
            pose_frame_ids.append(frame_idx)

        if out_writer is not None:
            vis = det.draw_result(frame, result)
            out_writer.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            print(f"\r  Processing: {frame_idx}/{total_frames} ({elapsed:.1f}s)", end="", flush=True)

    cap.release()
    if out_writer:
        out_writer.release()

    elapsed = time.time() - t_start
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)\n")

    # =====================================================================
    # PIXEL-SPACE (2D) METRICS
    # =====================================================================
    n_total = len(results_log)
    n_detected = sum(1 for r in results_log if r["n_tags"] > 0)
    n_success = sum(1 for r in results_log if r["success"])
    n_predicted = sum(1 for r in results_log if r["predicted"])
    n_measured = n_success - n_predicted

    reproj_errors = [r["reproj_error"] for r in results_log if r["reproj_error"] is not None and not r["predicted"]]
    tag_counts = [r["n_tags"] for r in results_log if r["n_tags"] > 0]

    print("=" * 60)
    print("PIXEL-SPACE (2D) METRICS")
    print("=" * 60)
    print(f"  Total frames:          {n_total}")
    print(f"  Detection rate:        {n_detected}/{n_total} ({100*n_detected/n_total:.1f}%)")
    print(f"  Pose success rate:     {n_success}/{n_total} ({100*n_success/n_total:.1f}%)")
    print(f"    Measured:            {n_measured}")
    print(f"    KF predicted:        {n_predicted}")
    print(f"  Dropout frames:        {n_total - n_success}")

    if reproj_errors:
        re = np.array(reproj_errors)
        print(f"  Reprojection error (px):")
        print(f"    mean: {re.mean():.3f}  median: {np.median(re):.3f}  std: {re.std():.3f}")
        print(f"    p95:  {np.percentile(re, 95):.3f}  max: {re.max():.3f}")

    if tag_counts:
        tc = np.array(tag_counts)
        print(f"  Tags per frame (when detected):")
        print(f"    mean: {tc.mean():.1f}  min: {tc.min()}  max: {tc.max()}")

    # Dropout streak analysis
    streaks = []
    current_streak = 0
    for r in results_log:
        if not r["success"]:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    if streaks:
        s = np.array(streaks)
        print(f"  Dropout streaks:       {len(s)} events")
        print(f"    mean: {s.mean():.1f} frames  max: {s.max()} frames ({s.max()/fps*1000:.0f}ms)")
    else:
        print(f"  Dropout streaks:       none")

    # =====================================================================
    # 3D POSE METRICS
    # =====================================================================
    print()
    print("=" * 60)
    print("3D POSE METRICS")
    print("=" * 60)

    if len(pose_tvecs) < 2:
        print("  Not enough pose data for 3D metrics.")
        print()
        return

    tvecs = np.array(pose_tvecs)  # (N, 3) mm
    times = np.array(pose_times)

    # Frame-to-frame translation jitter
    dt_trans = np.diff(tvecs, axis=0)  # (N-1, 3) mm
    dt_time = np.diff(times)           # (N-1,) seconds
    dt_time[dt_time == 0] = 1e-6

    trans_speed = np.linalg.norm(dt_trans, axis=1) / dt_time  # mm/s
    trans_jitter = np.linalg.norm(dt_trans, axis=1)            # mm per frame

    # Frame-to-frame rotation jitter
    rot_jitter = []
    for i in range(len(pose_rvecs) - 1):
        R1, _ = cv2.Rodrigues(pose_rvecs[i].reshape(3, 1))
        R2, _ = cv2.Rodrigues(pose_rvecs[i + 1].reshape(3, 1))
        rot_jitter.append(angle_between_rotations(R1, R2))
    rot_jitter = np.array(rot_jitter)
    rot_speed = rot_jitter / dt_time  # rad/s

    print(f"  Pose frames:           {len(tvecs)}")
    print()

    # Translation jitter
    print(f"  Translation jitter (mm/frame):")
    print(f"    mean: {trans_jitter.mean():.3f}  median: {np.median(trans_jitter):.3f}  std: {trans_jitter.std():.3f}")
    print(f"    p95:  {np.percentile(trans_jitter, 95):.3f}  max: {trans_jitter.max():.3f}")

    print(f"  Translation velocity (mm/s):")
    print(f"    mean: {trans_speed.mean():.1f}  median: {np.median(trans_speed):.1f}  max: {trans_speed.max():.1f}")

    print()

    # Rotation jitter
    rot_jitter_deg = np.degrees(rot_jitter)
    rot_speed_deg = np.degrees(rot_speed)
    print(f"  Rotation jitter (deg/frame):")
    print(f"    mean: {rot_jitter_deg.mean():.4f}  median: {np.median(rot_jitter_deg):.4f}  std: {rot_jitter_deg.std():.4f}")
    print(f"    p95:  {np.percentile(rot_jitter_deg, 95):.4f}  max: {rot_jitter_deg.max():.4f}")

    print(f"  Rotation velocity (deg/s):")
    print(f"    mean: {rot_speed_deg.mean():.1f}  median: {np.median(rot_speed_deg):.1f}  max: {rot_speed_deg.max():.1f}")

    print()

    # Sliding-window jitter (stationary proxy): std of position over 10-frame windows
    win = 10
    if len(tvecs) >= win:
        window_stds = []
        for i in range(len(tvecs) - win + 1):
            window_stds.append(tvecs[i:i+win].std(axis=0))
        window_stds = np.array(window_stds)  # (M, 3)
        window_std_norm = np.linalg.norm(window_stds, axis=1)

        # Find the calmest 10% of windows as "stationary" segments
        threshold = np.percentile(window_std_norm, 10)
        stationary_mask = window_std_norm <= threshold

        if stationary_mask.any():
            stat_stds = window_stds[stationary_mask]
            print(f"  Stationary jitter (quietest 10% of {win}-frame windows):")
            print(f"    position std:  x={stat_stds[:,0].mean():.3f}  y={stat_stds[:,1].mean():.3f}  z={stat_stds[:,2].mean():.3f} mm")
            print(f"    norm std:      {np.linalg.norm(stat_stds.mean(axis=0)):.3f} mm")

    # Acceleration spikes (translation)
    if len(tvecs) >= 3:
        accel = np.diff(dt_trans, axis=0)  # (N-2, 3)
        dt2 = ((dt_time[:-1] + dt_time[1:]) / 2)
        dt2[dt2 == 0] = 1e-6
        accel_norm = np.linalg.norm(accel, axis=1) / (dt2 ** 2)  # mm/s²
        print(f"\n  Acceleration spikes (mm/s^2):")
        print(f"    mean: {accel_norm.mean():.1f}  p95: {np.percentile(accel_norm, 95):.1f}  max: {accel_norm.max():.1f}")

    # Summary score
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    detection_pct = 100 * n_success / n_total if n_total > 0 else 0
    mean_reproj = np.mean(reproj_errors) if reproj_errors else float("inf")
    mean_trans_jitter = trans_jitter.mean()
    mean_rot_jitter = rot_jitter_deg.mean()
    print(f"  Detection rate:      {detection_pct:.1f}%")
    print(f"  Mean reproj error:   {mean_reproj:.3f} px")
    print(f"  Mean trans jitter:   {mean_trans_jitter:.3f} mm/frame")
    print(f"  Mean rot jitter:     {mean_rot_jitter:.4f} deg/frame")

    if args.save_video:
        print(f"\n  Output video: {os.path.join(args.recording, 'benchmark_output.mp4')}")
    print()


if __name__ == "__main__":
    main()
