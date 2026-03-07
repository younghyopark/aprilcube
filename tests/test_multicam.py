#!/usr/bin/env python3
"""Synthetic multi-camera benchmark for AprilCube.

Stress-tests the multi-camera calibration and fusion pipeline by rendering
the cube from two cameras on opposite ends. The cube traverses between
cameras — sometimes close to primary, sometimes far (near aux) — with
lateral motion across the baseline.

Scenarios:
  1. Baseline:             cube traverses full range, no occlusion
  2. Near-pri occluded:    cube close to primary, primary blocked
  3. Near-aux occluded:    cube far from primary (near aux), aux blocked
  4. Both occluded:        partial occlusion on both
  5. Primary blind blocks: intermittent total primary loss
  6. Far cube:             large depth, small tags
  7. Lateral offset:       cube stays to one side (different faces per cam)
  8. Alternating occ:      which camera is blocked switches rapidly

Usage:
  python tests/test_multicam.py
  python tests/test_multicam.py --cube models/3x3x3_30_cube/config.json
  python tests/test_multicam.py -v                     # verbose per-frame output
  python tests/test_multicam.py --save-images debug/   # save visualization frames
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

# Reuse rendering infrastructure from test_detection
from test_detection import (
    build_config,
    build_face_textures,
    render_at_pose,
    apply_occlusion,
    rotation_error_deg,
    translation_error_mm,
    translation_error_pct,
)

from aprilcube.detect import (
    AuxCamera,
    CubePoseEstimator,
    load_cube_config,
    build_tag_corner_map,
    estimate_pose,
)
from aprilcube.generate import CubeConfig, FACE_DEFS, _camera_from_angles


def _apply_blur(image: np.ndarray, strength: float) -> np.ndarray:
    """Degrade image with heavy Gaussian blur.

    Args:
        strength: 0.0 = no blur, 1.0 = maximally blurred (tags undetectable).
                  Kernel size scales from 1 to ~61px.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return image
    # Map strength to kernel size: 0->1, 0.5->31, 0.8->49, 1.0->61
    ksize = int(1 + 60 * strength)
    ksize = ksize if ksize % 2 == 1 else ksize + 1  # must be odd
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


# ═══════════════════════════════════════════════════════════════════════════
# Test geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_T_primary_aux(distance_mm: float,
                       yaw_deg: float = 170.0,
                       lateral_mm: float = 0.0,
                       elevation_deg: float = 10.0) -> np.ndarray:
    """Build T_primary_aux for two cameras facing each other.

    The aux camera is placed in front of primary (along +Z), offset
    laterally, and rotated to look back towards primary. This creates
    genuinely different viewpoints — each camera sees different cube faces.

    Args:
        distance_mm: Depth separation along Z-axis (how far apart cameras
                     are along the viewing direction).
        yaw_deg: Aux camera yaw rotation around Y-axis. 180 = exactly
                 facing primary. 160 = facing back with 20deg convergence.
        lateral_mm: Lateral (X-axis) offset of aux camera.
        elevation_deg: Vertical tilt difference (rotation around X-axis).
    """
    # Aux camera position in primary frame
    T = np.eye(4)
    T[0, 3] = lateral_mm
    T[2, 3] = distance_mm

    # Yaw: rotate aux around Y (180 = facing back at primary)
    ay = np.radians(yaw_deg)
    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)],
    ])

    # Elevation difference
    ax = np.radians(elevation_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)],
    ])

    T[:3, :3] = Ry @ Rx
    return T


def generate_orbit_poses(config: CubeConfig, n_frames: int,
                         distance_factor: float = 2.5,
                         elev_range: tuple = (-20, 35),
                         azim_range: tuple = (0, 360)
                         ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (rvec, tvec) pairs orbiting the cube at fixed distance."""
    diag = math.sqrt(sum(d ** 2 for d in config.box_dims))
    dist = diag * distance_factor
    poses = []
    for i in range(n_frames):
        t = i / n_frames
        azim = azim_range[0] + (azim_range[1] - azim_range[0]) * t
        elev = (elev_range[0] + elev_range[1]) / 2 + \
               (elev_range[1] - elev_range[0]) / 2 * math.sin(2 * math.pi * t)
        rv, tv, _ = _camera_from_angles(elev, azim, dist)
        poses.append((rv, tv))
    return poses


def generate_traverse_poses(config: CubeConfig, n_frames: int,
                            T_pa: np.ndarray,
                            depth_frac_range: tuple = (0.25, 0.75),
                            lateral_cycles: float = 1.5,
                            elev_range: tuple = (-10, 20),
                            ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate poses where the cube traverses between the two cameras.

    The cube moves laterally and oscillates in depth between the cameras.
    Depth is specified as a fraction of the camera separation along Z.

    rvec/tvec are T_cam_cube: rotation and translation of the cube in
    the primary camera's coordinate frame.

    Args:
        config: Cube configuration.
        n_frames: Number of frames to generate.
        T_pa: T_primary_aux (needed for camera geometry).
        depth_frac_range: (min, max) depth as fraction of Z-separation.
                          0.0 = at primary, 1.0 = at aux.
        lateral_cycles: Number of full lateral sweep cycles.
        elev_range: (min, max) elevation angle in degrees.
    """
    diag = math.sqrt(sum(d ** 2 for d in config.box_dims))
    aux_pos = T_pa[:3, 3]  # aux camera position in primary frame
    z_sep = abs(aux_pos[2]) if abs(aux_pos[2]) > 1.0 else float(np.linalg.norm(aux_pos))
    x_offset = aux_pos[0]  # lateral offset of aux

    # Minimum depth: at least 1.5x cube diagonal so tags are resolvable
    min_depth = max(diag * 1.5, z_sep * depth_frac_range[0])
    max_depth = max(diag * 3.0, z_sep * depth_frac_range[1])

    poses = []
    for i in range(n_frames):
        t = i / n_frames
        phase = 2 * math.pi * t

        # Depth oscillates: near primary <-> near aux
        depth_frac = 0.5 + 0.5 * math.sin(phase)
        z = min_depth + (max_depth - min_depth) * depth_frac

        # Lateral sweep: cube moves across, centered on midpoint between cams
        lat_phase = 2 * math.pi * lateral_cycles * t
        lat_center = x_offset * 0.5
        lat_amp = max(abs(x_offset) * 0.5, diag)
        x = lat_center + lat_amp * math.sin(lat_phase)

        # Small vertical wobble
        elev_mid = (elev_range[0] + elev_range[1]) / 2
        elev_amp = (elev_range[1] - elev_range[0]) / 2
        elev = elev_mid + elev_amp * math.sin(3 * phase)
        y = z * math.tan(math.radians(elev))

        # Cube orientation: slowly rotates so different faces are visible
        yaw = phase * 0.7
        pitch = 0.3 * math.sin(phase * 1.3)
        Ry = np.array([
            [math.cos(yaw), 0, math.sin(yaw)],
            [0, 1, 0],
            [-math.sin(yaw), 0, math.cos(yaw)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch), math.cos(pitch)],
        ])
        R_cube = Ry @ Rx

        rvec, _ = cv2.Rodrigues(R_cube)
        tvec = np.array([x, y, z], dtype=np.float64).reshape(3, 1)

        poses.append((rvec.astype(np.float64), tvec))
    return poses


# ═══════════════════════════════════════════════════════════════════════════
# Core benchmark runner
# ═══════════════════════════════════════════════════════════════════════════

class MultiCamBenchmark:
    """Run a multi-camera scenario and collect statistics."""

    def __init__(self, config: CubeConfig, face_id_sets: dict,
                 face_textures: dict, K_primary: np.ndarray,
                 K_aux: np.ndarray, T_pa_true: np.ndarray,
                 W: int = 640, H: int = 480,
                 bootstrap_samples: int = 30):
        self.config = config
        self.face_id_sets = face_id_sets
        self.face_textures = face_textures
        self.K_primary = K_primary
        self.K_aux = K_aux
        self.T_pa_true = T_pa_true.copy()
        self.T_ap_true = np.linalg.inv(T_pa_true)
        self.W, self.H = W, H
        self.bootstrap_samples = bootstrap_samples
        self.dist = np.zeros(5, dtype=np.float64)

    def _make_estimators(self):
        """Create fresh primary-only and multi-cam estimators."""
        config, fids = self.config, self.face_id_sets
        tag_corner_map = build_tag_corner_map(config)

        # Primary-only estimator (for comparison)
        est_pri = CubePoseEstimator(
            config, fids, self.K_primary, self.dist,
            enable_filter=False, fast=True,
        )

        # Multi-cam estimator
        est_multi = CubePoseEstimator(
            config, fids, self.K_primary, self.dist,
            enable_filter=False, fast=True,
        )
        est_multi.add_camera(
            intrinsics=self.K_aux, dist_coeffs=self.dist,
            fixed=True, bootstrap_samples=self.bootstrap_samples,
        )
        return est_pri, est_multi

    def _render_pair(self, rv_primary, tv_primary,
                     occlude_primary: float = 0.0,
                     occlude_aux: float = 0.0,
                     rng: np.random.Generator | None = None):
        """Render a frame from both cameras.

        Returns (primary_image, aux_image).
        """
        # Primary camera image
        img_pri = render_at_pose(
            self.face_textures, self.config,
            rv_primary, tv_primary,
            self.K_primary, self.W, self.H,
        )

        # Aux camera: T_aux_cube = T_aux_primary @ T_primary_cube
        T_cube = np.eye(4)
        T_cube[:3, :3], _ = cv2.Rodrigues(rv_primary)
        T_cube[:3, 3] = tv_primary.flatten()
        T_aux_cube = self.T_ap_true @ T_cube
        rv_aux, _ = cv2.Rodrigues(T_aux_cube[:3, :3])
        tv_aux = T_aux_cube[:3, 3].reshape(3, 1)

        img_aux = render_at_pose(
            self.face_textures, self.config,
            rv_aux, tv_aux,
            self.K_aux, self.W, self.H,
        )

        # Apply degradation via heavy blur (simulates defocus / motion blur)
        if occlude_primary > 0:
            img_pri = _apply_blur(img_pri, occlude_primary)
        if occlude_aux > 0:
            img_aux = _apply_blur(img_aux, occlude_aux)

        return img_pri, img_aux

    def run_scenario(
        self,
        name: str,
        poses: list[tuple[np.ndarray, np.ndarray]],
        occlude_primary_fn=None,
        occlude_aux_fn=None,
        blind_primary_fn=None,
        verbose: bool = False,
        save_dir: str | None = None,
    ) -> dict:
        """Run a full scenario.

        Args:
            name: Scenario label.
            poses: List of (rvec, tvec) for primary camera.
            occlude_primary_fn: f(frame_idx) -> occlusion fraction for primary.
            occlude_aux_fn: f(frame_idx) -> occlusion fraction for aux.
            blind_primary_fn: f(frame_idx) -> bool, if True, primary gets blank image.
            verbose: Print per-frame details.
            save_dir: If set, save visualization images.

        Returns:
            Dict of aggregate statistics.
        """
        est_pri, est_multi = self._make_estimators()
        n_frames = len(poses)

        if save_dir:
            scenario_dir = os.path.join(save_dir, name.replace(" ", "_"))
            os.makedirs(scenario_dir, exist_ok=True)

        stats = {
            "name": name,
            "n_frames": n_frames,
            # Primary-only
            "pri_ok": 0, "pri_rot_errs": [], "pri_t_errs_mm": [], "pri_t_errs_pct": [],
            # Multi-cam
            "multi_ok": 0, "multi_rot_errs": [], "multi_t_errs_mm": [], "multi_t_errs_pct": [],
            "multi_aux_only": 0, "multi_predicted": 0,
            "multi_joint_reproj": [],
            # Calibration
            "calibrated_at_frame": None,
            "extrinsic_t_err": None,
            "extrinsic_r_err": None,
        }

        # --- Bootstrap phase: use clean frames ---
        # Use 3x the target to ensure enough dual-detection frames survive
        bootstrap_poses = generate_traverse_poses(
            self.config, self.bootstrap_samples * 3,
            self.T_pa_true,
            depth_frac_range=(0.3, 0.6),
        )
        if verbose:
            print(f"  Bootstrapping with {len(bootstrap_poses)} clean frames...")

        for bi, (rv, tv) in enumerate(bootstrap_poses):
            img_pri, img_aux = self._render_pair(rv, tv)
            est_multi.process_frame(img_pri, timestamp=float(bi) / 30.0,
                                    aux_frames=[img_aux])
            cam = est_multi._aux_cameras[0]
            if cam.calibrated:
                stats["calibrated_at_frame"] = bi
                # Measure extrinsic accuracy
                T_pa_est = cam.T_primary_aux
                t_err = float(np.linalg.norm(
                    T_pa_est[:3, 3] - self.T_pa_true[:3, 3]))
                R_cos = np.clip(
                    (np.trace(T_pa_est[:3, :3].T @ self.T_pa_true[:3, :3]) - 1) / 2,
                    -1, 1)
                r_err = float(np.degrees(np.arccos(R_cos)))
                stats["extrinsic_t_err"] = t_err
                stats["extrinsic_r_err"] = r_err
                if verbose:
                    print(f"  Calibrated at frame {bi}: "
                          f"t_err={t_err:.2f}mm, r_err={r_err:.3f}deg")
                break

        if not est_multi._aux_cameras[0].calibrated:
            if verbose:
                print("  WARNING: calibration failed during bootstrap!")

        # Also warm up primary-only estimator with the same clean frames
        for rv, tv in bootstrap_poses[:5]:
            img_pri, _ = self._render_pair(rv, tv)
            est_pri.process_frame(img_pri, timestamp=0.0)

        # --- Test phase ---
        if verbose:
            print(f"  Running {n_frames} test frames...")

        for i, (rv_gt, tv_gt) in enumerate(poses):
            rng = np.random.default_rng(i + 1000)

            occ_pri = occlude_primary_fn(i) if occlude_primary_fn else 0.0
            occ_aux = occlude_aux_fn(i) if occlude_aux_fn else 0.0
            blind_pri = blind_primary_fn(i) if blind_primary_fn else False

            img_pri, img_aux = self._render_pair(
                rv_gt, tv_gt, occ_pri, occ_aux, rng)

            if blind_pri:
                img_pri = np.full_like(img_pri, 180)

            ts = float(i) / 30.0

            # Primary-only
            res_pri = est_pri.process_frame(img_pri, timestamp=ts)
            if res_pri["success"]:
                stats["pri_ok"] += 1
                re = rotation_error_deg(rv_gt, res_pri["rvec"])
                te = translation_error_mm(tv_gt, res_pri["tvec"])
                tp = translation_error_pct(tv_gt, res_pri["tvec"])
                stats["pri_rot_errs"].append(re)
                stats["pri_t_errs_mm"].append(te)
                stats["pri_t_errs_pct"].append(tp)

            # Multi-cam
            res_multi = est_multi.process_frame(
                img_pri, timestamp=ts, aux_frames=[img_aux])
            if res_multi["success"]:
                stats["multi_ok"] += 1
                re = rotation_error_deg(rv_gt, res_multi["rvec"])
                te = translation_error_mm(tv_gt, res_multi["tvec"])
                tp = translation_error_pct(tv_gt, res_multi["tvec"])
                stats["multi_rot_errs"].append(re)
                stats["multi_t_errs_mm"].append(te)
                stats["multi_t_errs_pct"].append(tp)
                if res_multi.get("aux_only"):
                    stats["multi_aux_only"] += 1
                if res_multi.get("predicted"):
                    stats["multi_predicted"] += 1
                je = res_multi.get("joint_reproj_error")
                if je is not None:
                    stats["multi_joint_reproj"].append(je)

            # Check if calibration happened during test phase
            cam = est_multi._aux_cameras[0]
            if cam.calibrated and stats["calibrated_at_frame"] is None:
                stats["calibrated_at_frame"] = -(i + 1)  # negative = during test
                T_pa_est = cam.T_primary_aux
                t_err = float(np.linalg.norm(
                    T_pa_est[:3, 3] - self.T_pa_true[:3, 3]))
                R_cos = np.clip(
                    (np.trace(T_pa_est[:3, :3].T @ self.T_pa_true[:3, :3]) - 1) / 2,
                    -1, 1)
                r_err = float(np.degrees(np.arccos(R_cos)))
                stats["extrinsic_t_err"] = t_err
                stats["extrinsic_r_err"] = r_err
                if verbose:
                    print(f"  Calibrated during test at frame {i}: "
                          f"t_err={t_err:.2f}mm, r_err={r_err:.3f}deg")

            if verbose and (i % 20 == 0 or i == n_frames - 1):
                pri_tag = f"rot={stats['pri_rot_errs'][-1]:.2f}d" if res_pri["success"] else "MISS"
                multi_tag = f"rot={stats['multi_rot_errs'][-1]:.2f}d" if res_multi["success"] else "MISS"
                if res_multi.get("aux_only"):
                    multi_tag += " AUX"
                flag = ""
                if occ_pri > 0:
                    flag += f" occ_pri={occ_pri:.0%}"
                if occ_aux > 0:
                    flag += f" occ_aux={occ_aux:.0%}"
                if blind_pri:
                    flag += " BLIND"
                print(f"    [{i:4d}] pri={pri_tag:>12s}  multi={multi_tag:>16s}"
                      f"  tags_pri={res_pri['n_tags']:2d}  tags_multi={res_multi['n_tags']:2d}"
                      f"{flag}")

            if save_dir:
                # Top row: raw camera views (what each camera actually sees)
                raw_pri = img_pri.copy()
                raw_aux = img_aux.copy()
                cv2.putText(raw_pri, "PRIMARY VIEW", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(raw_aux, "AUX VIEW", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Show occlusion info on raw views
                if occ_pri > 0:
                    cv2.putText(raw_pri, f"blur {occ_pri:.0%}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if occ_aux > 0:
                    cv2.putText(raw_aux, f"blur {occ_aux:.0%}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if blind_pri:
                    cv2.putText(raw_pri, "BLIND", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                top_row = np.hstack([raw_pri, raw_aux])

                # Bottom row: detection results
                vis_pri = est_pri.draw_result(img_pri, res_pri)
                vis_multi = est_multi.draw_result(img_pri, res_multi)
                cv2.putText(vis_pri, "PRIMARY ONLY", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(vis_multi, "MULTI-CAM", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                bot_row = np.hstack([vis_pri, vis_multi])

                combined = np.vstack([top_row, bot_row])
                # Scale down if too large
                if combined.shape[1] > 1600:
                    scale = 1600 / combined.shape[1]
                    combined = cv2.resize(combined, None, fx=scale, fy=scale)
                cv2.imwrite(
                    os.path.join(scenario_dir, f"{i:04d}.png"), combined)

        return stats


# ═══════════════════════════════════════════════════════════════════════════
# Scenario definitions
# ═══════════════════════════════════════════════════════════════════════════

def define_scenarios(config: CubeConfig, T_pa: np.ndarray,
                     n_frames: int = 60):
    """Return list of (name, poses, occlude_pri, occlude_aux, blind_pri).

    Scenarios use traverse poses where the cube moves between cameras,
    varying in depth and lateral position.
    """

    def _traverse(**kw):
        return generate_traverse_poses(config, n_frames, T_pa, **kw)

    scenarios = []

    # 1. Baseline: cube traverses between cameras, no occlusion
    scenarios.append((
        "baseline",
        _traverse(),
        None, None, None,
    ))

    # 2. Cube near primary with primary heavily occluded — aux must help from far
    scenarios.append((
        "near_pri_occluded",
        _traverse(depth_frac_range=(0.15, 0.35)),
        lambda i: 0.7 + 0.1 * math.sin(2 * math.pi * i / n_frames),
        None, None,
    ))

    # 3. Cube near aux with aux heavily occluded — primary must cope from far
    scenarios.append((
        "near_aux_occluded",
        _traverse(depth_frac_range=(0.65, 0.85)),
        None,
        lambda i: 0.7 + 0.1 * math.sin(2 * math.pi * i / n_frames),
        None,
    ))

    # 4. Both heavily occluded while cube traverses full range
    scenarios.append((
        "both_occluded",
        _traverse(),
        lambda i: 0.5 + 0.3 * math.sin(2 * math.pi * i / n_frames),
        lambda i: 0.5 + 0.3 * math.cos(2 * math.pi * i / n_frames),
        None,
    ))

    # 5. Primary intermittently blind — cube moves across, aux-only fallback
    scenarios.append((
        "primary_blind_blocks",
        _traverse(),
        None, None,
        lambda i: (i // 10) % 2 == 1,
    ))

    # 6. Far cube: cube near aux end, small in both views
    scenarios.append((
        "far_cube",
        _traverse(depth_frac_range=(0.7, 0.95)),
        None, None, None,
    ))

    # 7. Cube stays to one side — each camera sees different faces
    scenarios.append((
        "lateral_offset",
        _traverse(lateral_cycles=0.25),  # slow sweep, stays offset
        None, None, None,
    ))

    # 8. Alternating occlusion while cube traverses
    def alt_pri(i):
        phase = (i % 30) / 30.0
        return 0.8 if phase < 0.5 else 0.0

    def alt_aux(i):
        phase = (i % 30) / 30.0
        return 0.0 if phase < 0.5 else 0.8

    scenarios.append((
        "alternating_occlusion",
        _traverse(),
        alt_pri, alt_aux, None,
    ))

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(all_stats: list[dict]):
    """Print a comparison table of all scenarios."""

    def _arr(vals):
        if not vals:
            return None
        return np.array(vals)

    def _fmt(arr, unit=""):
        if arr is None or len(arr) == 0:
            return "     -     "
        u = unit
        return f"{np.mean(arr):6.2f}{u} (p95={np.percentile(arr, 95):6.2f}{u})"

    col_w = 24
    sep = "=" * 120

    print(f"\n{sep}")
    print("MULTI-CAMERA BENCHMARK RESULTS")
    print(sep)

    # Header
    print(f"\n{'Scenario':<25s} {'Det Rate':>18s}   {'Rot Error (deg)':>{col_w}s}   "
          f"{'Trans Error (mm)':>{col_w}s}   {'Notes'}")
    print(f"{'':<25s} {'Pri / Multi':>18s}   {'Pri / Multi':>{col_w}s}   "
          f"{'Pri / Multi':>{col_w}s}")
    print("-" * 120)

    for s in all_stats:
        n = s["n_frames"]
        pri_rate = f"{s['pri_ok']}/{n}"
        multi_rate = f"{s['multi_ok']}/{n}"
        rate_str = f"{pri_rate:>8s} / {multi_rate:<8s}"

        pri_rot = _arr(s["pri_rot_errs"])
        multi_rot = _arr(s["multi_rot_errs"])
        pri_t = _arr(s["pri_t_errs_mm"])
        multi_t = _arr(s["multi_t_errs_mm"])

        def _short(arr):
            if arr is None or len(arr) == 0:
                return "   -   "
            return f"{np.mean(arr):.2f}"

        rot_str = f"{_short(pri_rot):>7s} / {_short(multi_rot):<7s}"
        t_str = f"{_short(pri_t):>7s} / {_short(multi_t):<7s}"

        notes = []
        if s["multi_aux_only"] > 0:
            notes.append(f"aux_only={s['multi_aux_only']}")
        if s["multi_predicted"] > 0:
            notes.append(f"predicted={s['multi_predicted']}")
        if s["extrinsic_t_err"] is not None:
            notes.append(f"ext_err={s['extrinsic_t_err']:.1f}mm/{s['extrinsic_r_err']:.2f}deg")
        if s["calibrated_at_frame"] is not None:
            notes.append(f"calib@{s['calibrated_at_frame']}")

        print(f"{s['name']:<25s} {rate_str:>18s}   {rot_str:>{col_w}s}   "
              f"{t_str:>{col_w}s}   {', '.join(notes)}")

    # Detailed per-scenario stats
    print(f"\n{sep}")
    print("DETAILED STATISTICS")
    print(sep)

    for s in all_stats:
        print(f"\n--- {s['name']} ({s['n_frames']} frames) ---")
        print(f"  Detection rate:  primary={s['pri_ok']}/{s['n_frames']}  "
              f"multi={s['multi_ok']}/{s['n_frames']}")

        if s["calibrated_at_frame"] is not None:
            print(f"  Calibration:     frame {s['calibrated_at_frame']}, "
                  f"t_err={s['extrinsic_t_err']:.2f}mm, "
                  f"r_err={s['extrinsic_r_err']:.3f}deg")
        else:
            print("  Calibration:     FAILED")

        for label, rot, tmm, tpct in [
            ("Primary-only", s["pri_rot_errs"], s["pri_t_errs_mm"], s["pri_t_errs_pct"]),
            ("Multi-cam",    s["multi_rot_errs"], s["multi_t_errs_mm"], s["multi_t_errs_pct"]),
        ]:
            if not rot:
                print(f"  {label:14s}:  no detections")
                continue
            ra = np.array(rot)
            ta = np.array(tmm)
            pa = np.array(tpct)
            print(f"  {label:14s}:  rot: mean={ra.mean():.3f} med={np.median(ra):.3f} "
                  f"p95={np.percentile(ra, 95):.3f} max={ra.max():.3f} deg")
            print(f"  {'':14s}   trans: mean={ta.mean():.2f} med={np.median(ta):.2f} "
                  f"p95={np.percentile(ta, 95):.2f} max={ta.max():.2f} mm "
                  f"({pa.mean():.3f}%)")

        if s["multi_joint_reproj"]:
            ja = np.array(s["multi_joint_reproj"])
            print(f"  Joint reproj:    mean={ja.mean():.3f} "
                  f"med={np.median(ja):.3f} max={ja.max():.3f} px")

        if s["multi_aux_only"] > 0:
            print(f"  Aux-only frames: {s['multi_aux_only']}")
        if s["multi_predicted"] > 0:
            print(f"  Predicted (KF):  {s['multi_predicted']}")

        # Improvement summary
        if s["pri_rot_errs"] and s["multi_rot_errs"]:
            pri_mean_r = np.mean(s["pri_rot_errs"])
            multi_mean_r = np.mean(s["multi_rot_errs"])
            pri_mean_t = np.mean(s["pri_t_errs_mm"])
            multi_mean_t = np.mean(s["multi_t_errs_mm"])
            r_improv = (pri_mean_r - multi_mean_r) / pri_mean_r * 100 if pri_mean_r > 0 else 0
            t_improv = (pri_mean_t - multi_mean_t) / pri_mean_t * 100 if pri_mean_t > 0 else 0
            det_improv = s["multi_ok"] - s["pri_ok"]
            parts = []
            if abs(r_improv) > 1:
                parts.append(f"rot {'improved' if r_improv > 0 else 'degraded'} {abs(r_improv):.0f}%")
            if abs(t_improv) > 1:
                parts.append(f"trans {'improved' if t_improv > 0 else 'degraded'} {abs(t_improv):.0f}%")
            if det_improv > 0:
                parts.append(f"+{det_improv} more detections")
            elif det_improv < 0:
                parts.append(f"{det_improv} fewer detections")
            if parts:
                print(f"  vs primary:      {', '.join(parts)}")

    print(f"\n{sep}")


# ═══════════════════════════════════════════════════════════════════════════
# Baseline comparisons (varying camera baselines)
# ═══════════════════════════════════════════════════════════════════════════

def run_baseline_sweep(bench_cls, config, face_id_sets, face_textures,
                       K_primary, K_aux, W, H, n_frames, verbose, save_dir):
    """Test different camera separations and toe-in angles."""
    print("\n" + "=" * 80)
    print("BASELINE SWEEP: varying camera separation")
    print("=" * 80)

    distances = [300, 600, 900]
    yaws = [120, 160]

    sweep_stats = []

    for dist in distances:
        for yaw in yaws:
            name = f"dist_{dist}mm_yaw{yaw}deg"
            T_pa = make_T_primary_aux(dist, yaw_deg=yaw)
            poses = generate_traverse_poses(config, n_frames, T_pa)

            bench = bench_cls(
                config, face_id_sets, face_textures,
                K_primary, K_aux, T_pa, W, H,
                bootstrap_samples=30,
            )
            if verbose:
                print(f"\n[{name}]")
            stats = bench.run_scenario(name, poses, verbose=verbose,
                                       save_dir=save_dir)
            sweep_stats.append(stats)

    # Summary table
    print(f"\n{'Config':<30s} {'Calib t_err':>12s} {'Calib r_err':>12s} "
          f"{'Det Rate':>10s} {'Rot Err':>10s} {'Trans Err':>10s}")
    print("-" * 90)
    for s in sweep_stats:
        det = f"{s['multi_ok']}/{s['n_frames']}"
        te = f"{s['extrinsic_t_err']:.2f}mm" if s['extrinsic_t_err'] is not None else "FAIL"
        re = f"{s['extrinsic_r_err']:.3f}deg" if s['extrinsic_r_err'] is not None else "FAIL"
        rot = f"{np.mean(s['multi_rot_errs']):.3f}deg" if s['multi_rot_errs'] else "-"
        trans = f"{np.mean(s['multi_t_errs_mm']):.2f}mm" if s['multi_t_errs_mm'] else "-"
        print(f"{s['name']:<30s} {te:>12s} {re:>12s} {det:>10s} {rot:>10s} {trans:>10s}")

    return sweep_stats


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic multi-camera benchmark for AprilCube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cube", type=str, default=None,
                        help="config.json (default: models/2x2x2_30_cube)")
    parser.add_argument("--grid", type=str, default="2x2x2")
    parser.add_argument("--dict", type=str, default="4x4_50")
    parser.add_argument("--tag-size", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=60,
                        help="Test frames per scenario (default: 60)")
    parser.add_argument("--bootstrap", type=int, default=30,
                        help="Bootstrap samples for calibration (default: 30)")
    parser.add_argument("--distance", type=float, default=600.0,
                        help="Depth separation between cameras in mm (default: 600)")
    parser.add_argument("--yaw", type=float, default=170.0,
                        help="Aux camera yaw: 180=facing primary exactly (default: 170)")
    parser.add_argument("--lateral", type=float, default=0.0,
                        help="Lateral offset of aux camera in mm (default: 0)")
    parser.add_argument("--elevation", type=float, default=10.0,
                        help="Elevation difference in deg (default: 10)")
    parser.add_argument("--resolution", type=int, default=640,
                        help="Image resolution width (default: 640)")
    parser.add_argument("--pixels-per-cell", type=int, default=20)
    parser.add_argument("--save-images", type=str, default=None,
                        help="Save debug visualization images to directory")
    parser.add_argument("--sweep", action="store_true",
                        help="Run baseline sweep (varying camera separations)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Default cube
    if args.cube is None:
        args.cube = "models/2x2x2_30_cube/config.json"

    config, face_id_sets = build_config(args)
    bx, by, bz = config.box_dims
    print(f"Cube: {bx:.1f} x {by:.1f} x {bz:.1f} mm, "
          f"{len(config.tag_ids)} tags ({config.dict_name})")

    face_textures = build_face_textures(config, args.pixels_per_cell)

    W = args.resolution
    H = int(W * 3 / 4)  # 4:3 aspect
    # Use a wider FOV camera (fx ~ W) so both cameras can see the cube
    # even with a significant baseline offset
    fx = float(W)
    K_primary = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]],
                         dtype=np.float64)
    # Slightly different aux intrinsics (realistic)
    K_aux = K_primary.copy()
    K_aux[0, 0] *= 0.95
    K_aux[1, 1] *= 0.95

    print(f"Primary: {W}x{H}, fx={K_primary[0,0]:.0f}")
    print(f"Aux:     {W}x{H}, fx={K_aux[0,0]:.0f}")
    print(f"Cam separation: dist={args.distance:.0f}mm, lateral={args.lateral:.0f}mm, "
          f"yaw={args.yaw:.0f}deg, elev={args.elevation:.0f}deg")
    print(f"Bootstrap samples: {args.bootstrap}")

    T_pa = make_T_primary_aux(args.distance, yaw_deg=args.yaw,
                              lateral_mm=args.lateral,
                              elevation_deg=args.elevation)

    # Run all scenarios
    scenarios = define_scenarios(config, T_pa, n_frames=args.frames)
    all_stats = []

    for name, poses, occ_pri, occ_aux, blind_pri in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"{'='*60}")

        bench = MultiCamBenchmark(
            config, face_id_sets, face_textures,
            K_primary, K_aux, T_pa, W, H,
            bootstrap_samples=args.bootstrap,
        )
        stats = bench.run_scenario(
            name, poses,
            occlude_primary_fn=occ_pri,
            occlude_aux_fn=occ_aux,
            blind_primary_fn=blind_pri,
            verbose=args.verbose,
            save_dir=args.save_images,
        )
        all_stats.append(stats)

    print_summary(all_stats)

    # Optional baseline sweep
    if args.sweep:
        run_baseline_sweep(
            MultiCamBenchmark, config, face_id_sets, face_textures,
            K_primary, K_aux, W, H, args.frames, args.verbose,
            args.save_images,
        )

    # Overall pass/fail
    n_calib = sum(1 for s in all_stats if s["calibrated_at_frame"] is not None)
    n_scenarios = len(all_stats)
    print(f"\nCalibration succeeded: {n_calib}/{n_scenarios} scenarios")

    # Check that multi-cam never significantly degrades vs primary-only
    n_degraded = 0
    for s in all_stats:
        if s["pri_rot_errs"] and s["multi_rot_errs"]:
            if np.mean(s["multi_rot_errs"]) > np.mean(s["pri_rot_errs"]) * 1.5:
                n_degraded += 1
                print(f"  WARNING: {s['name']} — multi-cam degraded rotation accuracy")
    if n_degraded == 0:
        print("Multi-cam never significantly degraded vs primary-only: OK")

    return 0 if n_calib == n_scenarios and n_degraded == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
