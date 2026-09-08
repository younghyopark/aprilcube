#!/usr/bin/env python3
"""Simple webcam cube detection with OpenCV window.

Opens the first available webcam and shows live detection overlay in a cv2 window.
No dependencies on pycaas or viser — just opencv-python and numpy.

Usage:
  python examples/webcam_simple.py --cube models/2x2x2_30_cube
  python examples/webcam_simple.py --cube models/tiny_1x1x1_cubes
  python examples/webcam_simple.py --cube models/2x2x2_30_cube --camera 1
  python examples/webcam_simple.py --cube models/2x2x2_30_cube --intrinsics calib.json
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import aprilcube
import cv2
import numpy as np


COLORS_BGR = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 128, 255),
    (255, 255, 0),
    (160, 80, 255),
    (80, 220, 120),
    (255, 120, 120),
    (120, 180, 255),
    (180, 255, 120),
    (255, 180, 80),
]


@dataclass
class Target:
    label: str
    config_path: Path
    detector: aprilcube.CubePoseEstimator
    color: tuple[int, int, int]


def find_config_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise ValueError(f"{root} is not a file or directory")

    direct = root / "config.json"
    configs = []
    if direct.is_file():
        configs.append(direct)
    configs.extend(p for p in root.rglob("config.json") if p != direct)
    return sorted(configs, key=lambda p: str(p.relative_to(root)))


def target_label(config_path: Path, root: Path) -> str:
    if root.is_dir():
        try:
            rel = config_path.parent.relative_to(root)
            if str(rel) != ".":
                return str(rel)
        except ValueError:
            pass
    return config_path.parent.name


def load_targets(
    cube_path: str | Path,
    intrinsics,
    enable_filter: bool,
    fast: bool,
) -> list[Target]:
    root = Path(cube_path)
    config_paths = find_config_paths(root)
    if not config_paths:
        raise ValueError(f"No config.json files found under {root}")

    targets = []
    for idx, config_path in enumerate(config_paths):
        det = aprilcube.detector(
            config_path,
            intrinsic_cfg=intrinsics,
            enable_filter=enable_filter,
            fast=fast,
        )
        label = target_label(config_path, root)
        targets.append(Target(
            label=label,
            config_path=config_path,
            detector=det,
            color=COLORS_BGR[idx % len(COLORS_BGR)],
        ))
    return targets


def warn_duplicate_marker_ids(targets: list[Target]) -> None:
    owners: dict[tuple[str, int], str] = {}
    duplicates: list[tuple[str, int, str, str]] = []
    for target in targets:
        dict_name = target.detector.config.dict_name
        for tag_id in sorted(target.detector.valid_ids):
            key = (dict_name, tag_id)
            if key in owners:
                duplicates.append((dict_name, tag_id, owners[key], target.label))
            else:
                owners[key] = target.label

    if not duplicates:
        return
    print("Warning: duplicate marker IDs across targets with the same dictionary.")
    print("         Distinguishing those targets may be ambiguous.")
    for dict_name, tag_id, first, second in duplicates[:10]:
        print(f"  {dict_name}:{tag_id} appears in {first} and {second}")
    if len(duplicates) > 10:
        print(f"  ... and {len(duplicates) - 10} more duplicates")


def process_targets(
    frame: np.ndarray,
    targets: list[Target],
    timestamp: float,
) -> list[tuple[Target, dict]]:
    grouped: dict[int, list[Target]] = defaultdict(list)
    for target in targets:
        grouped[target.detector.config.dict_id].append(target)

    results = []
    for group_targets in grouped.values():
        markers = group_targets[0].detector.detect_markers(frame)
        for target in group_targets:
            result = target.detector.process_marker_detections(
                frame,
                markers,
                timestamp=timestamp,
                store_latest=False,
            )
            results.append((target, result))
    return results


def draw_text_box(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 4
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = xy
    x = max(0, min(x, image.shape[1] - tw - 2 * pad - 1))
    y = max(th + pad, min(y, image.shape[0] - baseline - pad - 1))
    cv2.rectangle(
        image,
        (x, y - th - pad),
        (x + tw + 2 * pad, y + baseline + pad),
        (20, 20, 20),
        cv2.FILLED,
    )
    cv2.rectangle(
        image,
        (x, y - th - pad),
        (x + tw + 2 * pad, y + baseline + pad),
        color,
        1,
    )
    cv2.putText(image, text, (x + pad, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_target_overlay(image: np.ndarray, target: Target, result: dict) -> None:
    det = target.detector
    color = target.color

    for _, corners_2d in result["detections"]:
        pts = corners_2d.astype(np.int32)
        cv2.polylines(image, [pts], True, color, 3)

    if not result["success"]:
        return

    rvec = result["rvec"]
    tvec = result["tvec"]

    projected, _ = cv2.projectPoints(
        det.box_corners_3d,
        rvec,
        tvec,
        det.camera_matrix,
        det.dist_coeffs,
    )
    pts = projected.reshape(-1, 2).astype(int)
    for i, j in det.box_edges:
        cv2.line(image, tuple(pts[i]), tuple(pts[j]), color, 3)

    axis_len = float(max(det.config.box_dims) / 2)
    axes_3d = np.float64([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ])
    axis_pts, _ = cv2.projectPoints(
        axes_3d,
        rvec,
        tvec,
        det.camera_matrix,
        det.dist_coeffs,
    )
    axis_pts = axis_pts.reshape(-1, 2).astype(int)
    origin = tuple(axis_pts[0])
    cv2.arrowedLine(image, origin, tuple(axis_pts[1]), (0, 0, 255), 3, tipLength=0.15)
    cv2.arrowedLine(image, origin, tuple(axis_pts[2]), (0, 255, 0), 3, tipLength=0.15)
    cv2.arrowedLine(image, origin, tuple(axis_pts[3]), (255, 0, 0), 3, tipLength=0.15)

    predicted = " pred" if result.get("predicted") else ""
    text = (
        f"{target.label}  {result['n_tags']}/{len(det.valid_ids)} "
        f"{result['reproj_error']:.1f}px{predicted}"
    )
    label_xy = (int(axis_pts[0][0]) + 8, int(axis_pts[0][1]) - 8)
    draw_text_box(image, text, label_xy, color)


def draw_status_panel(
    image: np.ndarray,
    results: list[tuple[Target, dict]],
    fps: float,
) -> None:
    detected = sum(1 for _, result in results if result["success"])
    lines = [
        f"Targets: {detected}/{len(results)}   FPS: {fps:.0f}",
        "q: quit",
    ]
    max_lines = 14
    for target, result in results[:max_lines]:
        status = "OK" if result["success"] else "--"
        lines.append(f"{status} {target.label} ({result['n_tags']}/{len(target.detector.valid_ids)})")
    if len(results) > max_lines:
        lines.append(f"... {len(results) - max_lines} more")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    pad = 8
    line_h = 22
    width = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        width = max(width, tw)
    panel_w = width + 2 * pad
    panel_h = len(lines) * line_h + 2 * pad

    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    for i, line in enumerate(lines):
        y = 8 + pad + (i + 1) * line_h - 5
        cv2.putText(image, line, (8 + pad, y), font, scale, (230, 230, 230), thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Simple webcam multi-cube detection")
    parser.add_argument("--cube", required=True,
                        help="Path to config.json, model directory, or directory containing config.json files")
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

    try:
        targets = load_targets(
            args.cube,
            intrinsics,
            enable_filter=not args.no_filter,
            fast=not args.slow,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        cap.release()
        return

    print(f"Loaded {len(targets)} target(s):")
    for idx, target in enumerate(targets, start=1):
        cfg = target.detector.config
        bx, by, bz = cfg.box_dims
        print(
            f"  {idx:02d}. {target.label}: {len(target.detector.valid_ids)} tags, "
            f"{cfg.dict_name}, {bx:.4g}x{by:.4g}x{bz:.4g} mm"
        )
    warn_duplicate_marker_ids(targets)

    print("Press 'q' to quit.")
    fps_timer = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = time.monotonic()
        results = process_targets(frame, targets, timestamp)

        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        vis = frame.copy()
        for target, result in results:
            draw_target_overlay(vis, target, result)
        draw_status_panel(vis, results, fps_display)

        cv2.imshow("AprilCube Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
