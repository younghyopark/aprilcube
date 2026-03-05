#!/usr/bin/env python3
"""Test detect_cube.py: discrete viewpoint tests and continuous video trajectories.

Renders the cube at known 6-DOF poses, runs the detector, and compares with
ground truth.  Two modes:

  Discrete (default) — systematic viewpoints, pass/fail per threshold.
  Video (--video)    — smooth trajectory, records annotated .mp4.

Usage:
  python test_detection.py
  python test_detection.py --cube models/2x2x2_30_cube/config.json -v
  python test_detection.py --video orbit.mp4 --trajectory orbit
  python test_detection.py --video full.mp4 --trajectory full --frames 300
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

from aprilcube.generate import (
    CubeConfig,
    DICT_MAP,
    FACE_DEFS,
    TagPatternGenerator,
    build_face_grid,
    parse_grid,
    render_face_texture,
    _camera_from_angles,
    _face_quad_corners,
)
from aprilcube.detect import CubePoseEstimator, load_cube_config


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════
def render_at_pose(
    face_textures: dict[str, np.ndarray],
    config: CubeConfig,
    rvec: np.ndarray,
    tvec: np.ndarray,
    cam_matrix: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Render cube at an arbitrary object-to-camera pose.  Returns BGR image."""
    R_cam, _ = cv2.Rodrigues(rvec)
    cam_pos_obj = -R_cam.T @ tvec.flatten()  # camera position in object frame

    dist_coeffs = np.zeros(5)
    bg = np.full((img_h, img_w, 3), 180, dtype=np.uint8)

    # Back-face culling + depth sort (painter's algorithm)
    visible: list[tuple[float, str, np.ndarray]] = []
    for face_def in FACE_DEFS:
        normal = np.zeros(3)
        normal[face_def[1]] = face_def[2]
        if np.dot(normal, cam_pos_obj) > 0:
            corners = _face_quad_corners(face_def, config.box_dims)
            z_cam = (R_cam @ corners.mean(axis=0) + tvec.flatten())[2]
            visible.append((z_cam, face_def[0], corners))
    visible.sort(reverse=True)

    for _, name, corners_3d in visible:
        proj, _ = cv2.projectPoints(corners_3d, rvec, tvec, cam_matrix, dist_coeffs)
        pts_2d = proj.reshape(-1, 2).astype(np.float32)

        tex = face_textures[name]
        th, tw = tex.shape[:2]
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, pts_2d)
        warped = cv2.warpPerspective(
            tex, M, (img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(180, 180, 180),
        )
        mask = cv2.warpPerspective(
            np.full((th, tw), 255, dtype=np.uint8),
            M, (img_w, img_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bg = np.where(mask[:, :, np.newaxis] > 0, warped, bg)

    return bg


def render_for_detection(
    face_textures: dict[str, np.ndarray],
    config: CubeConfig,
    elev_deg: float,
    azim_deg: float,
    cam_matrix: np.ndarray,
    img_w: int,
    img_h: int,
    distance_factor: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render from spherical coordinates.  Returns (image, rvec, tvec)."""
    diag = math.sqrt(sum(d**2 for d in config.box_dims))
    rvec, tvec, _ = _camera_from_angles(elev_deg, azim_deg, diag * distance_factor)
    image = render_at_pose(face_textures, config, rvec, tvec, cam_matrix, img_w, img_h)
    return image, rvec, tvec


# ═══════════════════════════════════════════════════════════════════════════
# Pose comparison
# ═══════════════════════════════════════════════════════════════════════════
def rotation_error_deg(rvec_a: np.ndarray, rvec_b: np.ndarray) -> float:
    R_a, _ = cv2.Rodrigues(rvec_a.reshape(3, 1))
    R_b, _ = cv2.Rodrigues(rvec_b.reshape(3, 1))
    cos_a = np.clip((np.trace(R_a @ R_b.T) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def translation_error_mm(tvec_a: np.ndarray, tvec_b: np.ndarray) -> float:
    return float(np.linalg.norm(tvec_a.flatten() - tvec_b.flatten()))


def translation_error_pct(tvec_a: np.ndarray, tvec_b: np.ndarray) -> float:
    d = float(np.linalg.norm(tvec_a.flatten()))
    return translation_error_mm(tvec_a, tvec_b) / d * 100.0 if d > 1e-6 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Face texture builder
# ═══════════════════════════════════════════════════════════════════════════
def build_face_textures(
    config: CubeConfig, pixels_per_cell: int = 20,
    chamfer_radius: int = 0,
) -> dict[str, np.ndarray]:
    tag_gen = TagPatternGenerator(config.dict_id)
    patterns = [tag_gen.generate(tid) for tid in config.tag_ids]
    textures: dict[str, np.ndarray] = {}
    cur = 0
    for fd in FACE_DEFS:
        fr, fc, dc, rc = config.face_layout(fd)
        n = fr * fc
        grid = build_face_grid(
            patterns[cur : cur + n], fr, fc, dc, rc,
            config.marker_pixels, config.margin_cells, config.invert,
        )
        cur += n
        g = render_face_texture(grid, pixels_per_cell)
        if chamfer_radius > 0:
            g = apply_edge_rounding(g, chamfer_radius)
        textures[fd[0]] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return textures


# ═══════════════════════════════════════════════════════════════════════════
# Image augmentations (edge rounding, occlusion, motion blur)
# ═══════════════════════════════════════════════════════════════════════════
def apply_edge_rounding(
    texture: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Round corners of black/white cells via morphological open+close.

    Simulates 3D-printed tags where sharp cell corners become rounded arcs.

    Args:
        texture: grayscale uint8 image (0=black, 255=white).
        radius: radius of the circular structuring element in pixels.
    """
    if radius <= 0:
        return texture
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1),
    )
    # Open rounds convex corners of white regions
    # Close rounds convex corners of black regions
    result = cv2.morphologyEx(texture, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def apply_occlusion(
    image: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw random filled polygons over the image to simulate occlusion.

    Args:
        fraction: approximate fraction of image area to occlude (0–1).
        rng: numpy random generator for reproducibility.
    """
    if fraction <= 0:
        return image
    h, w = image.shape[:2]
    img = image.copy()
    total_area = h * w
    target_area = total_area * np.clip(fraction, 0, 0.8)
    covered = 0.0
    n_patches = rng.integers(2, 6)

    for _ in range(n_patches):
        if covered >= target_area:
            break
        patch_area = (target_area - covered) / max(1, n_patches)
        radius = int(math.sqrt(patch_area / math.pi))
        radius = max(10, radius)

        # center biased toward image center (where the cube is)
        cx = int(rng.normal(w / 2, w * 0.2))
        cy = int(rng.normal(h / 2, h * 0.2))

        n_verts = rng.integers(3, 7)
        angles = np.sort(rng.uniform(0, 2 * math.pi, n_verts))
        radii = rng.uniform(0.5, 1.0, n_verts) * radius
        pts = np.array([
            [cx + int(r * math.cos(a)), cy + int(r * math.sin(a))]
            for a, r in zip(angles, radii)
        ], dtype=np.int32)

        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.fillPoly(img, [pts], color)
        covered += 0.5 * radius * radius * math.pi  # approximate

    return img


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int,
    angle_deg: float,
) -> np.ndarray:
    """Apply directional motion blur.

    Args:
        kernel_size: length of the blur kernel in pixels (odd, >= 3).
        angle_deg: direction of motion in degrees.
    """
    if kernel_size < 3:
        return image
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Build a line kernel
    kernel = np.zeros((ks, ks), dtype=np.float32)
    center = ks // 2
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    for i in range(ks):
        t = i - center
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < ks and 0 <= y < ks:
            kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1.0)

    return cv2.filter2D(image, -1, kernel)


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory generators
# ═══════════════════════════════════════════════════════════════════════════
def _diag(config: CubeConfig) -> float:
    return math.sqrt(sum(d**2 for d in config.box_dims))


def generate_trajectory(
    name: str, n_frames: int, config: CubeConfig, distance_factor: float = 2.5,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Return [(rvec, tvec, description), ...] for the named trajectory."""
    diag = _diag(config)
    base_dist = diag * distance_factor
    TWO_PI = 2 * math.pi
    out: list[tuple[np.ndarray, np.ndarray, str]] = []

    if name == "orbit":
        elev = 25.0
        for i in range(n_frames):
            azim = 360.0 * i / n_frames
            rv, tv, _ = _camera_from_angles(elev, azim, base_dist)
            out.append((rv, tv, f"a={azim:.0f}"))

    elif name == "spiral":
        for i in range(n_frames):
            t = i / n_frames
            azim = 720.0 * t
            elev = 30.0 * math.sin(TWO_PI * t)
            rv, tv, _ = _camera_from_angles(elev, azim, base_dist)
            out.append((rv, tv, f"e={elev:.0f} a={azim:.0f}"))

    elif name == "approach":
        elev = 25.0
        for i in range(n_frames):
            t = i / n_frames
            azim = 90.0 * t
            d = diag * max(1.3, 2.5 + 1.5 * math.sin(TWO_PI * t * 2))
            rv, tv, _ = _camera_from_angles(elev, azim, d)
            out.append((rv, tv, f"d={d:.0f}"))

    elif name == "tumble":
        cam_rv, cam_tv, _ = _camera_from_angles(30, 45, base_dist)
        R_cam, _ = cv2.Rodrigues(cam_rv)
        axis = np.array([1.0, 0.6, 0.3], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        for i in range(n_frames):
            angle = TWO_PI * i / n_frames
            R_obj, _ = cv2.Rodrigues(axis * angle)
            rv_eff, _ = cv2.Rodrigues(R_cam @ R_obj)
            out.append((rv_eff, cam_tv.copy(), f"rot={math.degrees(angle):.0f}"))

    elif name == "translate":
        cam_rv, cam_tv, _ = _camera_from_angles(30, 45, base_dist)
        R_cam, _ = cv2.Rodrigues(cam_rv)
        radius = diag * 0.3
        for i in range(n_frames):
            angle = TWO_PI * i / n_frames
            obj_pos = np.array([
                radius * math.cos(angle),
                radius * math.sin(angle),
                0.0,
            ])
            tv_eff = (R_cam @ obj_pos).reshape(3, 1) + cam_tv
            out.append((cam_rv.copy(), tv_eff, f"pos={math.degrees(angle):.0f}"))

    elif name == "full":
        for i in range(n_frames):
            t = i / n_frames
            azim = 360.0 * t
            elev = 20.0 + 20.0 * math.sin(3 * TWO_PI * t)
            d = diag * max(1.3, 2.5 + 0.8 * math.sin(2 * TWO_PI * t))
            cam_rv, cam_tv, _ = _camera_from_angles(elev, azim, d)
            R_cam, _ = cv2.Rodrigues(cam_rv)
            # slow object tumble
            axis = np.array([0.3, 1.0, 0.5], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            R_obj, _ = cv2.Rodrigues(axis * (math.pi * t))
            rv_eff, _ = cv2.Rodrigues(R_cam @ R_obj)
            out.append((rv_eff, cam_tv.copy(), f"e={elev:.0f} a={azim:.0f}"))

    elif name == "wander":
        # Large random-walk lateral motion + object tumble
        rng = np.random.default_rng(42)
        # Build smooth random walk in XY via exponential smoothing
        raw_x = rng.standard_normal(n_frames)
        raw_y = rng.standard_normal(n_frames)
        alpha = 0.05  # smoothing factor — lower = smoother
        sx, sy = 0.0, 0.0
        walk_x, walk_y = [], []
        for j in range(n_frames):
            sx = (1 - alpha) * sx + alpha * raw_x[j]
            sy = (1 - alpha) * sy + alpha * raw_y[j]
            walk_x.append(sx)
            walk_y.append(sy)
        # Normalize to ±0.8 * diagonal
        wx = np.array(walk_x)
        wy = np.array(walk_y)
        max_abs = max(np.abs(wx).max(), np.abs(wy).max(), 1e-9)
        wx = wx / max_abs * diag * 0.8
        wy = wy / max_abs * diag * 0.8

        # Smooth distance variation
        dist_walk = rng.standard_normal(n_frames).cumsum()
        dist_walk -= dist_walk.min()
        dist_walk = dist_walk / max(dist_walk.max(), 1e-9)  # 0–1
        # map to [1.3, 4.0] * diag
        dists = diag * (1.3 + 2.7 * dist_walk)
        # smooth it
        for j in range(1, n_frames):
            dists[j] = 0.9 * dists[j - 1] + 0.1 * dists[j]

        axis = np.array([0.7, 1.0, 0.4], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        base_elev = 20.0
        for i in range(n_frames):
            t = i / n_frames
            azim = 180.0 * t  # half orbit
            cam_rv, cam_tv, _ = _camera_from_angles(base_elev, azim, dists[i])
            R_cam, _ = cv2.Rodrigues(cam_rv)
            # lateral offset in object frame
            obj_offset = np.array([wx[i], wy[i], 0.0])
            tv_eff = (R_cam @ obj_offset).reshape(3, 1) + cam_tv
            # object tumble
            R_obj, _ = cv2.Rodrigues(axis * (TWO_PI * t))
            rv_eff, _ = cv2.Rodrigues(R_cam @ R_obj)
            out.append((rv_eff, tv_eff, f"wander t={t:.2f}"))

    elif name == "shake":
        # Slow orbit with high-frequency handheld jitter
        rng = np.random.default_rng(77)
        for i in range(n_frames):
            t = i / n_frames
            base_azim = 90.0 * t  # quarter orbit
            base_elev = 25.0
            # jitter
            azim = base_azim + rng.uniform(-5, 5)
            elev = base_elev + rng.uniform(-5, 5)
            d = base_dist * (1.0 + rng.uniform(-0.15, 0.15))
            cam_rv, cam_tv, _ = _camera_from_angles(elev, azim, d)
            R_cam, _ = cv2.Rodrigues(cam_rv)
            # small XY offset jitter
            jitter = np.array([
                rng.uniform(-1, 1) * diag * 0.15,
                rng.uniform(-1, 1) * diag * 0.15,
                0.0,
            ])
            tv_eff = (R_cam @ jitter).reshape(3, 1) + cam_tv
            out.append((cam_rv.copy(), tv_eff, f"shake t={t:.2f}"))

    elif name == "stress":
        # Everything at once: fast orbit, big elevation, distance, offset, tumble
        for i in range(n_frames):
            t = i / n_frames
            azim = 720.0 * t  # 2 full orbits
            elev = 40.0 * math.sin(5 * TWO_PI * t)  # ±40°, 5 oscillations
            d = diag * max(1.2, 2.5 + 2.0 * math.sin(3 * TWO_PI * t))
            cam_rv, cam_tv, _ = _camera_from_angles(elev, azim, d)
            R_cam, _ = cv2.Rodrigues(cam_rv)
            # large lateral offset: circular + sine
            ox = diag * 0.5 * math.sin(TWO_PI * t * 2)
            oy = diag * 0.5 * math.cos(TWO_PI * t * 3)
            obj_offset = np.array([ox, oy, 0.0])
            tv_eff = (R_cam @ obj_offset).reshape(3, 1) + cam_tv
            # aggressive object tumble
            axis = np.array([1.0, 0.8, 0.6], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            R_obj, _ = cv2.Rodrigues(axis * (TWO_PI * t))
            rv_eff, _ = cv2.Rodrigues(R_cam @ R_obj)
            out.append((rv_eff, tv_eff, f"stress t={t:.2f}"))

    else:
        raise ValueError(f"Unknown trajectory: {name}")

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Discrete test viewpoints
# ═══════════════════════════════════════════════════════════════════════════
def generate_test_viewpoints() -> list[tuple[float, float, str]]:
    vps: list[tuple[float, float, str]] = []
    vps.extend([
        (0, 0, "+X face head-on"),
        (0, 90, "+Y face head-on"),
        (0, 180, "-X face head-on"),
        (0, 270, "-Y face head-on"),
        (89, 0, "+Z face head-on"),
        (-89, 0, "-Z face head-on"),
    ])
    for a in [45, 135, 225, 315]:
        vps.append((0, a, f"edge elev=0 azim={a}"))
    for e in [30, -30]:
        for a in [45, 135, 225, 315]:
            vps.append((e, a, f"corner e={e} a={a}"))
    for e in [-45, -15, 15, 45, 60]:
        for a in [0, 60, 120, 180, 240, 300]:
            if not any(ev == e and av == a for ev, av, _ in vps):
                vps.append((e, a, f"oblique e={e} a={a}"))
    return vps


# ═══════════════════════════════════════════════════════════════════════════
# Config / camera helpers
# ═══════════════════════════════════════════════════════════════════════════
def build_config(args) -> tuple[CubeConfig, dict[str, set[int]]]:
    if args.cube:
        cfg, fids = load_cube_config(args.cube)
        print(f"Loaded config: {args.cube}")
        return cfg, fids

    gx, gy, gz = parse_grid(args.grid)
    cfg = CubeConfig(
        grid_x=gx, grid_y=gy, grid_z=gz,
        dict_id=DICT_MAP[args.dict], dict_name=args.dict,
        tag_ids=[], tag_size_mm=args.tag_size,
    )
    cfg.compute()
    cfg.tag_ids = list(range(cfg.total_tags()))
    fids: dict[str, set[int]] = {}
    cur = 0
    for fd in FACE_DEFS:
        fr, fc, _, _ = cfg.face_layout(fd)
        n = fr * fc
        fids[fd[0]] = set(cfg.tag_ids[cur : cur + n])
        cur += n
    print(f"Generated config: grid={args.grid} dict={args.dict} "
          f"tag_size={args.tag_size}mm")
    return cfg, fids


def build_camera(W: int, H: int) -> np.ndarray:
    fx = fy = float(W) * 1.8
    return np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Discrete test mode
# ═══════════════════════════════════════════════════════════════════════════
def run_discrete_test(args, config, face_id_sets, face_textures, cam_matrix, W, H):
    estimator = CubePoseEstimator(
        config, face_id_sets, cam_matrix,
        np.zeros(5, dtype=np.float64), enable_filter=False,
    )
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)

    viewpoints = generate_test_viewpoints()
    n_total = len(viewpoints)
    print(f"\nRunning {n_total} viewpoints...\n")

    n_det = n_pass = 0
    rot_errs: list[float] = []
    t_errs_mm: list[float] = []
    t_errs_pct: list[float] = []
    missed: list[tuple[int, str, int]] = []
    failed: list[tuple[int, str, float, float]] = []

    for i, (elev, azim, desc) in enumerate(viewpoints):
        estimator.prev_rvec = estimator.prev_tvec = None
        image, rv_gt, tv_gt = render_for_detection(
            face_textures, config, elev, azim, cam_matrix, W, H,
        )
        rng = np.random.default_rng(i)
        if args.occlusion > 0:
            image = apply_occlusion(image, args.occlusion, rng)
        if args.blur > 0:
            image = apply_motion_blur(image, args.blur, rng.uniform(0, 360))
        res = estimator.process_frame(image, timestamp=float(i))

        if res["success"]:
            n_det += 1
            re = rotation_error_deg(rv_gt, res["rvec"])
            te = translation_error_mm(tv_gt, res["tvec"])
            tp = translation_error_pct(tv_gt, res["tvec"])
            rot_errs.append(re)
            t_errs_mm.append(te)
            t_errs_pct.append(tp)

            ok = re <= args.rot_threshold and tp <= args.trans_threshold
            if ok:
                n_pass += 1
                st = "PASS"
            else:
                st = "FAIL"
                failed.append((i, desc, re, tp))
            if args.verbose:
                print(f"  [{i+1:3d}] {st:4s}  {desc:30s}  "
                      f"rot={re:6.2f}deg  trans={te:6.1f}mm ({tp:.2f}%)  "
                      f"tags={res['n_tags']}  reproj={res['reproj_error']:.2f}px")
        else:
            st = "MISS"
            re = tp = float("nan")
            missed.append((i, desc, res["n_tags"]))
            if args.verbose:
                print(f"  [{i+1:3d}] MISS  {desc:30s}  "
                      f"tags={res['n_tags']}  (no pose)")

        if args.save_images:
            vis = estimator.draw_result(image, res)
            gt_t = tv_gt.flatten()
            cv2.putText(vis, f"GT: [{gt_t[0]:.0f},{gt_t[1]:.0f},{gt_t[2]:.0f}]mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            cv2.putText(vis, f"elev={elev:.0f} azim={azim:.0f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
            if res["success"]:
                c = (0, 200, 0) if st == "PASS" else (0, 0, 255)
                cv2.putText(vis, f"err: rot={re:.1f}deg trans={tp:.1f}%",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
            fname = f"{i:03d}_e{elev:+04.0f}_a{azim:03.0f}_{st}.png"
            cv2.imwrite(os.path.join(args.save_images, fname), vis)

    # Summary
    print(f"\n{'=' * 65}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"Viewpoints tested : {n_total}")
    print(f"Pose detected     : {n_det}/{n_total} "
          f"({100 * n_det / n_total:.1f}%)")
    print(f"Passed thresholds : {n_pass}/{n_total} "
          f"({100 * n_pass / n_total:.1f}%)")
    print(f"  (rot <= {args.rot_threshold}deg, trans <= {args.trans_threshold}%)")
    if rot_errs:
        re_a = np.array(rot_errs)
        tm_a = np.array(t_errs_mm)
        tp_a = np.array(t_errs_pct)
        print(f"\nRotation error (deg):")
        print(f"  mean={re_a.mean():.3f}  median={np.median(re_a):.3f}  "
              f"max={re_a.max():.3f}  std={re_a.std():.3f}")
        print(f"Translation error (mm):")
        print(f"  mean={tm_a.mean():.2f}  median={np.median(tm_a):.2f}  "
              f"max={tm_a.max():.2f}  std={tm_a.std():.2f}")
        print(f"Translation error (% of distance):")
        print(f"  mean={tp_a.mean():.3f}  median={np.median(tp_a):.3f}  "
              f"max={tp_a.max():.3f}  std={tp_a.std():.3f}")
    if missed:
        print(f"\nMissed detections ({len(missed)}):")
        for idx, d, nt in missed:
            print(f"  [{idx+1:3d}] {d} (tags detected: {nt})")
    if failed:
        print(f"\nFailed — high error ({len(failed)}):")
        for idx, d, r, t in failed:
            print(f"  [{idx+1:3d}] {d}  rot={r:.2f}deg  trans={t:.2f}%")
    print(f"{'=' * 65}")
    if args.save_images:
        print(f"Images saved to {args.save_images}/")
    return 0 if n_pass == n_det and n_det > 0 else 1


# ═══════════════════════════════════════════════════════════════════════════
# Video mode
# ═══════════════════════════════════════════════════════════════════════════
def _draw_video_overlay(
    vis: np.ndarray,
    res: dict,
    rv_gt: np.ndarray,
    tv_gt: np.ndarray,
    frame_idx: int,
    n_frames: int,
    traj_name: str,
    desc: str,
) -> np.ndarray:
    """Add ground-truth, error, and progress overlays to a detection frame."""
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.45
    C_HEADER = (200, 200, 200)
    C_GT = (0, 200, 255)
    C_DET = (0, 255, 0)
    C_ERR = (100, 255, 255)
    C_MISS = (0, 0, 255)

    # Semi-transparent top panel
    panel_h = 80
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (40, 40, 40), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

    gt_r = rv_gt.flatten()
    gt_t = tv_gt.flatten()

    cv2.putText(vis,
                f"Frame {frame_idx+1}/{n_frames}  {traj_name}  {desc}",
                (10, 18), font, sc, C_HEADER, 1)
    cv2.putText(vis,
                f"GT:  T=[{gt_t[0]:+.0f},{gt_t[1]:+.0f},{gt_t[2]:+.0f}]mm  "
                f"R=[{gt_r[0]:+.3f},{gt_r[1]:+.3f},{gt_r[2]:+.3f}]",
                (10, 38), font, sc, C_GT, 1)

    if res["success"]:
        dt = res["tvec"].flatten()
        dr = res["rvec"].flatten()
        re = rotation_error_deg(rv_gt, res["rvec"])
        te = translation_error_mm(tv_gt, res["tvec"])
        tp = translation_error_pct(tv_gt, res["tvec"])
        cv2.putText(vis,
                    f"Det: T=[{dt[0]:+.0f},{dt[1]:+.0f},{dt[2]:+.0f}]mm  "
                    f"R=[{dr[0]:+.3f},{dr[1]:+.3f},{dr[2]:+.3f}]",
                    (10, 58), font, sc, C_DET, 1)
        cv2.putText(vis,
                    f"Err: rot={re:.2f}deg  trans={te:.1f}mm ({tp:.2f}%)  "
                    f"reproj={res['reproj_error']:.2f}px",
                    (10, 78), font, sc, C_ERR, 1)
    else:
        cv2.putText(vis, f"Det: NO POSE  (tags={res['n_tags']})",
                    (10, 58), font, sc, C_MISS, 1)

    # Progress bar
    bar_h = 4
    prog = (frame_idx + 1) / n_frames
    cv2.rectangle(vis, (0, h - bar_h), (int(w * prog), h), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(vis, (int(w * prog), h - bar_h), (w, h), (60, 60, 60), cv2.FILLED)

    return vis


def run_video_mode(args, config, face_id_sets, face_textures, cam_matrix, W, H):
    trajectory = generate_trajectory(
        args.trajectory, args.frames, config,
        distance_factor=2.5,
    )
    n_frames = len(trajectory)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.video, fourcc, args.fps, (W, H))
    if not writer.isOpened():
        print(f"Error: cannot open video writer for {args.video}", file=sys.stderr)
        sys.exit(1)

    enable_filter = not args.no_filter
    estimator = CubePoseEstimator(
        config, face_id_sets, cam_matrix,
        np.zeros(5, dtype=np.float64), enable_filter=enable_filter,
    )
    filter_label = "on" if enable_filter else "off"
    print(f"Generating {n_frames} frames  trajectory={args.trajectory}  "
          f"fps={args.fps}  filter={filter_label}")

    n_det = 0
    n_stale = 0
    n_predicted = 0
    rot_errs: list[float] = []
    t_errs_mm: list[float] = []
    t_errs_pct: list[float] = []
    stale_remaining = 0  # burst counter
    stale_image = None   # frozen frame fed to detector during stall

    for i, (rv_gt, tv_gt, desc) in enumerate(trajectory):
        image = render_at_pose(face_textures, config, rv_gt, tv_gt, cam_matrix, W, H)
        rng = np.random.default_rng(i)
        if args.occlusion > 0:
            image = apply_occlusion(image, args.occlusion, rng)
        if args.blur > 0:
            blur_angle = (i / n_frames) * 360.0
            image = apply_motion_blur(image, args.blur, blur_angle)

        # Frame stall logic: repeat a stale frame, then jump to current
        is_stale = False
        if stale_remaining > 0:
            stale_remaining -= 1
            is_stale = True
        elif args.drop_rate > 0 and rng.random() < args.drop_rate:
            stale_remaining = rng.integers(1, max(2, args.drop_burst + 1))
            stale_image = image.copy()  # freeze this frame
            stale_remaining -= 1
            is_stale = True

        if is_stale:
            n_stale += 1
            # Feed the frozen (old) image but with current timestamp
            res = estimator.process_frame(stale_image, timestamp=i / args.fps)
        else:
            stale_image = None
            res = estimator.process_frame(image, timestamp=i / args.fps)

        if res.get("predicted"):
            n_predicted += 1

        if res["success"]:
            n_det += 1
            re = rotation_error_deg(rv_gt, res["rvec"])
            te = translation_error_mm(tv_gt, res["tvec"])
            tp = translation_error_pct(tv_gt, res["tvec"])
            rot_errs.append(re)
            t_errs_mm.append(te)
            t_errs_pct.append(tp)

        vis_image = stale_image if is_stale else image
        vis = estimator.draw_result(vis_image, res)
        vis = _draw_video_overlay(vis, res, rv_gt, tv_gt, i, n_frames,
                                  args.trajectory, desc)
        writer.write(vis)

        if args.verbose and (i % 30 == 0 or i == n_frames - 1):
            if res["success"]:
                tag = " (stale)" if is_stale else ""
                if res.get("predicted"):
                    tag = " (predicted)"
                print(f"  [{i+1:4d}/{n_frames}] tags={res['n_tags']}  "
                      f"rot={re:.2f}deg  trans={tp:.2f}%{tag}")
            else:
                print(f"  [{i+1:4d}/{n_frames}] tags={res['n_tags']}  NO POSE")

    writer.release()
    print(f"\nWrote {args.video}")

    # Summary
    print(f"\n{'=' * 65}")
    print(f"VIDEO SUMMARY  ({args.trajectory}, {n_frames} frames, "
          f"filter={filter_label})")
    print(f"{'=' * 65}")
    if n_stale:
        print(f"Stale frames: {n_stale}/{n_frames} ({100 * n_stale / n_frames:.1f}%)")
    print(f"Detected: {n_det}/{n_frames} ({100 * n_det / n_frames:.1f}%)")
    if n_predicted:
        print(f"Predicted (KF, no tags): {n_predicted}")
    n_miss = n_frames - n_det
    if n_miss:
        print(f"Missed:   {n_miss}")
    if rot_errs:
        re_a = np.array(rot_errs)
        tm_a = np.array(t_errs_mm)
        tp_a = np.array(t_errs_pct)
        print(f"Rotation error (deg):  "
              f"mean={re_a.mean():.3f}  max={re_a.max():.3f}")
        print(f"Translation error:     "
              f"mean={tm_a.mean():.2f}mm ({tp_a.mean():.3f}%)  "
              f"max={tm_a.max():.2f}mm ({tp_a.max():.3f}%)")
    print(f"{'=' * 65}")
    return 0 if n_det == n_frames else 1


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Test cube pose detection: discrete viewpoints or video trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s                                        # discrete test\n"
            "  %(prog)s --cube models/2x2x2_30_cube/config.json -v\n"
            "  %(prog)s --video orbit.mp4 --trajectory orbit   # video\n"
            "  %(prog)s --video full.mp4 --trajectory full --frames 300\n"
            "  %(prog)s --video tumble.mp4 --trajectory tumble -v\n"
        ),
    )

    # ── Cube config ───────────────────────────────────────────────────────
    parser.add_argument("--cube", type=str, default=None,
                        help="config.json from generate_cube.py")
    parser.add_argument("--grid", type=str, default="2x2x2")
    parser.add_argument("--dict", type=str, default="4x4_50",
                        choices=sorted(DICT_MAP.keys()))
    parser.add_argument("--tag-size", type=float, default=30.0)

    # ── Rendering ─────────────────────────────────────────────────────────
    parser.add_argument("--resolution", type=int, default=800,
                        help="Image resolution (default: 800)")
    parser.add_argument("--pixels-per-cell", type=int, default=20,
                        help="Texture resolution per cell (default: 20)")

    # ── Discrete test ─────────────────────────────────────────────────────
    parser.add_argument("--save-images", type=str, default=None,
                        help="Save discrete test images to directory")
    parser.add_argument("--rot-threshold", type=float, default=5.0,
                        help="Max rotation error in degrees (default: 5)")
    parser.add_argument("--trans-threshold", type=float, default=2.0,
                        help="Max translation error in %% (default: 2)")

    # ── Video mode ────────────────────────────────────────────────────────
    parser.add_argument("--video", type=str, default=None,
                        help="Output video path (.mp4)")
    parser.add_argument("--trajectory", type=str, default="orbit",
                        choices=["orbit", "spiral", "approach",
                                 "tumble", "translate", "full",
                                 "wander", "shake", "stress"],
                        help="Trajectory type (default: orbit)")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of video frames (default: 300)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS (default: 30)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable temporal pose filter (video mode)")

    # ── Augmentation ──────────────────────────────────────────────────────
    parser.add_argument("--occlusion", type=float, default=0.0,
                        help="Occlusion fraction 0.0–1.0 (default: 0, off)")
    parser.add_argument("--blur", type=int, default=0,
                        help="Motion blur kernel size in px (default: 0, off)")
    parser.add_argument("--drop-rate", type=float, default=0.0,
                        help="Frame stall probability 0.0–1.0: repeats a frozen frame (default: 0, off)")
    parser.add_argument("--chamfer", type=float, default=0.0,
                        help="Edge rounding radius as fraction of cell size, "
                             "e.g. 0.3 = 30%% (default: 0, off)")
    parser.add_argument("--drop-burst", type=int, default=1,
                        help="Max consecutive stale frames per stall event (default: 1)")

    # ── Common ────────────────────────────────────────────────────────────
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    config, face_id_sets = build_config(args)
    bx, by, bz = config.box_dims
    print(f"Box: {bx:.1f} x {by:.1f} x {bz:.1f} mm, "
          f"{len(config.tag_ids)} tags ({config.dict_name})")

    chamfer_px = int(round(args.chamfer * args.pixels_per_cell)) if args.chamfer > 0 else 0
    face_textures = build_face_textures(config, args.pixels_per_cell, chamfer_px)

    W = H = args.resolution
    cam_matrix = build_camera(W, H)
    fx = cam_matrix[0, 0]
    print(f"Camera: {W}x{H}, fx=fy={fx:.0f}, "
          f"cx={cam_matrix[0, 2]:.0f}, cy={cam_matrix[1, 2]:.0f}")
    if chamfer_px > 0 or args.occlusion > 0 or args.blur > 0 or args.drop_rate > 0:
        parts = []
        if chamfer_px > 0:
            parts.append(f"chamfer={args.chamfer:.0%} ({chamfer_px}px)")
        if args.occlusion > 0:
            parts.append(f"occlusion={args.occlusion:.0%}")
        if args.blur > 0:
            parts.append(f"blur={args.blur}px")
        if args.drop_rate > 0:
            parts.append(f"stall={args.drop_rate:.0%} burst={args.drop_burst}")
        print(f"Augmentation: {', '.join(parts)}")

    if args.video:
        rc = run_video_mode(args, config, face_id_sets, face_textures,
                            cam_matrix, W, H)
    else:
        rc = run_discrete_test(args, config, face_id_sets, face_textures,
                               cam_matrix, W, H)
    sys.exit(rc)


if __name__ == "__main__":
    main()
