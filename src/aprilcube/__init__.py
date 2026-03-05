"""aprilcube - Generate and detect ArUco/AprilTag cubes for 6-DOF pose estimation."""

from __future__ import annotations

__version__ = "0.1.0"

import json
from pathlib import Path
from typing import Union

import numpy as np

from aprilcube.generate import CubeConfig, DICT_MAP, FACE_DEFS
from aprilcube.detect import (
    CubePoseEstimator,
    KalmanFilterConfig,
    KalmanPoseFilter,
    PoseSnapshot,
    load_cube_config,
    build_tag_corner_map,
)

__all__ = [
    "detector",
    "CubeConfig",
    "CubePoseEstimator",
    "KalmanFilterConfig",
    "KalmanPoseFilter",
    "PoseSnapshot",
    "load_cube_config",
    "build_tag_corner_map",
    "DICT_MAP",
    "FACE_DEFS",
]


def detector(
    cube_cfg: Union[str, Path],
    intrinsic_cfg: Union[str, Path, dict, np.ndarray],
    *,
    extrinsic: np.ndarray | None = None,
    enable_filter: bool = True,
    filter_config: KalmanFilterConfig | None = None,
    dist_coeffs: np.ndarray | None = None,
    fast: bool = False,
) -> CubePoseEstimator:
    """Create a CubePoseEstimator from config file paths or inline parameters.

    Args:
        cube_cfg: Path to config.json produced by ``aprilcube generate``.
            When a file path is given, its parent directory is used to locate
            the 3D model (``mujoco/cube.obj``) for viser visualization.
        intrinsic_cfg: Camera intrinsics, one of:
            - str/Path to calibration JSON (keys: ``camera_matrix``, optional ``dist_coeffs``)
            - dict with keys ``fx``, ``fy``, ``cx``, ``cy`` (and optional ``dist_coeffs``)
            - 3x3 numpy camera matrix directly
        extrinsic: Optional 4x4 world-to-camera transform (T_world_cam).
            When set, ``world_pose()`` returns object pose in world frame.
        enable_filter: Whether to enable Kalman temporal smoothing (default True).
        filter_config: Custom KalmanFilterConfig, or None for defaults.
        dist_coeffs: Distortion coefficients override (5-element array).
            If intrinsic_cfg is a JSON file, dist_coeffs from that file are used
            unless this parameter explicitly overrides them.
        fast: Use faster detector parameters suited for real-time webcam use.
            Trades some accuracy for speed (fewer threshold passes, cheaper
            corner refinement).

    Returns:
        A configured CubePoseEstimator ready to call ``process_frame(image)``.

    Examples::

        import aprilcube

        det = aprilcube.detector("my_cube/config.json", "calib.json")
        result = det.process_frame(frame)

        # With viser visualization
        det = aprilcube.detector(
            "my_cube/config.json",
            {"fx": 800, "fy": 800, "cx": 320, "cy": 240},
        )
        server = det.build_viser(port=8080)  # auto-renders in background
        # ... in your loop:
        result = det.process_frame(frame)  # viser updates automatically
    """
    cube_path = Path(cube_cfg)
    if cube_path.is_dir():
        cube_path = cube_path / "config.json"
    config, face_id_sets = load_cube_config(str(cube_path))

    # Resolve model directory from config path
    model_dir = str(cube_path.parent)

    camera_matrix, dc = _resolve_intrinsics(intrinsic_cfg)
    if dist_coeffs is not None:
        dc = np.asarray(dist_coeffs, dtype=np.float64)

    return CubePoseEstimator(
        config=config,
        face_id_sets=face_id_sets,
        camera_matrix=camera_matrix,
        dist_coeffs=dc,
        enable_filter=enable_filter,
        filter_config=filter_config,
        fast=fast,
        extrinsic=extrinsic,
        model_dir=model_dir,
    )


def _resolve_intrinsics(
    intrinsic_cfg: Union[str, Path, dict, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Parse intrinsic_cfg into (camera_matrix, dist_coeffs)."""
    zero_dist = np.zeros(5, dtype=np.float64)

    # numpy array (3x3 camera matrix)
    if isinstance(intrinsic_cfg, np.ndarray):
        if intrinsic_cfg.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 camera matrix, got shape {intrinsic_cfg.shape}"
            )
        return intrinsic_cfg.astype(np.float64), zero_dist

    # dict with fx/fy/cx/cy
    if isinstance(intrinsic_cfg, dict):
        fx = float(intrinsic_cfg["fx"])
        fy = float(intrinsic_cfg.get("fy", fx))
        cx = float(intrinsic_cfg["cx"])
        cy = float(intrinsic_cfg["cy"])
        mat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dc = zero_dist
        if "dist_coeffs" in intrinsic_cfg:
            dc = np.array(intrinsic_cfg["dist_coeffs"], dtype=np.float64)
        return mat, dc

    # file path (str or Path)
    path = Path(intrinsic_cfg)
    with open(path) as f:
        data = json.load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dc = np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    return camera_matrix, dc
