# AprilCube Usage Guide

This guide contains the detailed CLI, YAML, Python API, output, and detector notes that are intentionally kept out of the concise [README](../README.md).

## Python API

```python
import aprilcube

# Create a detector from config.json and camera intrinsics
det = aprilcube.detector(
    "my_target/config.json",
    {"fx": 800, "fy": 800, "cx": 320, "cy": 240},
)

# Process a frame (BGR numpy array)
result = det.process_frame(frame)

if result["success"]:
    rvec = result["rvec"]              # Rodrigues rotation vector (3x1)
    tvec = result["tvec"]              # Translation vector in mm (3x1)
    T = result["T"]                    # 4x4 camera-frame pose matrix
    error = result["reproj_error"]     # Reprojection error in pixels
    faces = result["visible_faces"]    # Set of visible face names
```

### `aprilcube.detector(cube_cfg, intrinsic_cfg, **kwargs)`

Creates a `CubePoseEstimator` ready to process frames. The class name is kept for API compatibility, but configs may describe either cuboid or voxel-composed targets.

| Arg | Type | Description |
|-----|------|-------------|
| `cube_cfg` | `str \| Path` | Path to `config.json` or generated model directory |
| `intrinsic_cfg` | `str \| Path \| dict \| np.ndarray` | Camera intrinsics |
| `extrinsic` | `np.ndarray \| None` | 4x4 world-to-camera transform (default: `None`) |
| `enable_filter` | `bool` | Enable Kalman temporal smoothing (default: `True`) |
| `filter_config` | `KalmanFilterConfig \| None` | Custom filter tuning |
| `dist_coeffs` | `np.ndarray \| None` | Override distortion coefficients |
| `fast` | `bool` | Faster detection for real-time use (default: `False`) |

### Camera Intrinsics

```python
# Path to calibration JSON (keys: "camera_matrix", optional "dist_coeffs")
det = aprilcube.detector("config.json", "calib.json")

# Dict with fx, fy, cx, cy
det = aprilcube.detector("config.json", {"fx": 800, "fy": 800, "cx": 320, "cy": 240})

# 3x3 numpy camera matrix directly
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
det = aprilcube.detector("config.json", K)
```

### 3D Visualization With `viser`

The detector has built-in web-based 3D visualization via [viser](https://github.com/nerfstudio-project/viser). Call `build_viser()` to start a server. It automatically renders the latest `process_frame` result in a background thread.

```bash
pip install viser trimesh
```

```python
import cv2
import aprilcube

det = aprilcube.detector(
    "models/my_target",
    {"fx": 800, "fy": 800, "cx": 320, "cy": 240},
    fast=True,
)

server = det.build_viser(port=8080)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    det.process_frame(frame)
```

The scene includes world, camera, and object coordinate frames, a ground grid, and the textured target mesh loaded from `mujoco/cube.obj` in the model directory.

### World-Frame Poses

Pass `extrinsic` as a 4x4 `T_world_cam` matrix to get object poses in world frame:

```python
import numpy as np

det = aprilcube.detector("my_target", intrinsics, extrinsic=np.eye(4))
result = det.process_frame(frame)
T_world_obj = det.world_pose(result)  # 4x4 numpy array or None
```

### Async Detection

Run detection in a background thread so `get_latest()` never blocks your main loop:

```python
import cv2
import aprilcube

det = aprilcube.detector("models/my_target", intrinsics, fast=True)
det.start_async()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    det.submit_frame(frame)       # Non-blocking, replaces any pending frame
    result = det.get_latest()     # Returns last completed result instantly
    if result and result["success"]:
        print(result["T"])

det.stop_async()
```

`submit_frame` uses a swap-not-queue design. If detection is slower than capture, intermediate frames are skipped so the detector works on the freshest frame.

### Direct Class Access

```python
from aprilcube import CubePoseEstimator, CubeConfig, KalmanFilterConfig
from aprilcube.generate import TagPatternGenerator, CubeMeshBuilder, ThreeMFWriter
```

## CLI

```bash
aprilcube generate [options]
```

The CLI can generate classic cuboids from flags or arbitrary voxel-composed targets from YAML specs:

```bash
# Simple cube, one tag per face
aprilcube generate --grid 1x1x1 --dict 4x4_50 --tag-size 30

# 2x2 cube with AprilTags
aprilcube generate --grid 2x2x2 --dict apriltag_36h11 --tag-size 20

# Flat calibration box
aprilcube generate --grid 5x4x1 --dict 4x4_100 --tag-size 15 -o flat_box

# Large cube with fine cell control
aprilcube generate --grid 3x3x3 --dict 6x6_250 --cell-size 2.5 --margin-cell 2 --border-cell 2

# YAML cuboid spec
aprilcube generate examples/cuboid_target.yaml

# Voxel-cuboid T-shaped target
aprilcube generate examples/t_shape_target.yaml

# Higher-ID-count voxel target
aprilcube generate examples/chair_target.yaml

# Add the included robot end-effector connector on the +Z face
aprilcube generate --grid 1x1x1 --dict 4x4_50 --tag-size 30 \
  --end-effector-connector --connector-rod-length 40
```

### Standalone Voxel Designer

Open the browser-based voxel target designer:

```bash
aprilcube web
```

The designer runs as a standalone local HTML page. It lets you edit voxel occupancy, preview marker placement, choose marker settings, and download a YAML spec that can be passed back to `aprilcube generate`.

To print the local HTML path without opening a browser:

```bash
aprilcube web --no-open
```

### YAML Generation Specs

`aprilcube generate` can read a YAML file. Supported shapes are `cuboid`, `voxel_cuboids`, and `voxel_grid`. Voxel targets place one marker on each exposed voxel face and write explicit marker corner coordinates into `config.json`.

```yaml
output: models/my_yaml_cube
shape:
  type: cuboid
  grid: [2, 2, 1]
dictionary: 4x4_50
markers:
  ids: 0-15
size:
  tag_size_mm: 24
layout:
  margin_cells: 1
  border_cells: 1
material:
  extruder: 1
  invert: false
attachment:
  type: end_effector_connector
  rod_length_mm: 40
  rod_radius_mm: 10
```

A T-shaped target can be expressed as a union of axis-aligned voxel cuboids:

```yaml
output: models/t_shape_target
shape:
  type: voxel_cuboids
  voxel_size_mm: 24
  cuboids:
    - name: stem
      origin: [1, 0, 0]
      size: [1, 1, 3]
    - name: crossbar
      origin: [0, 0, 2]
      size: [3, 1, 1]
dictionary: 4x4_50
markers:
  ids: 0-63
size:
  tag_size_mm: 18
layout:
  margin_cells: 1
  border_cells: 1
```

The repository includes several ready-to-generate voxel examples:

| Spec | Model directory | Dictionary | Markers | Shape idea |
|------|-----------------|------------|---------|------------|
| `examples/t_shape_target.yaml` | `models/t_shape_target` | `4x4_50` | 22 | T-shaped target |
| `examples/l_shape_target.yaml` | `models/l_shape_target` | `4x4_50` | 22 | L/corner target |
| `examples/stair_step_target.yaml` | `models/stair_step_target` | `4x4_50` | 24 | Stepped target |
| `examples/u_shape_target.yaml` | `models/u_shape_target` | `4x4_50` | 30 | U-shaped target |
| `examples/plus_shape_target.yaml` | `models/plus_shape_target` | `4x4_50` | 30 | 3D plus target |
| `examples/spiral_tower_target.yaml` | `models/spiral_tower_target` | `4x4_50` | 34 | Rising spiral tower |
| `examples/zigzag_snake_target.yaml` | `models/zigzag_snake_target` | `4x4_50` | 46 | Flat serpentine target |
| `examples/window_frame_target.yaml` | `models/window_frame_target` | `4x4_50` | 48 | Frame with open center |
| `examples/chair_target.yaml` | `models/chair_target` | `4x4_100` | 78 | Chair with four legs and backrest |

### CLI Options

| Arg | Default | Description |
|-----|---------|-------------|
| `spec` / `-c, --config` | - | YAML generation spec |
| `-g, --grid` | `1x1x1` | Tags per dimension: `WxHxD` |
| `-d, --dict` | `4x4_50` | ArUco/AprilTag dictionary |
| `-t, --ids` | auto | Tag IDs: range (`0-23`) or comma-separated |
| `--tag-size` | `30` | Tag size in mm |
| `--cell-size` | - | Cell size in mm, alternative to `--tag-size` |
| `--margin-cell` | `1` | Gap between adjacent tags, in cells |
| `--border-cell` | `1` | Outer border per face edge, in 0.5-cell increments |
| `-o, --output` | `aruco_cube` | Output directory |
| `--extruder` | `1` | Bambu Studio extruder number |
| `--invert` | - | Swap black and white |
| `--end-effector-connector` | - | Attach `assets/end_effector_connector.stl` on the target +Z surface |
| `--connector-rod-length` | - | Rod length in mm from the +Z surface to the connector |
| `--connector-rod-radius` | `10` | Connector rod radius in mm |
| `--connector-stl` | `assets/end_effector_connector.stl` | Alternate connector STL path |

Half-cell borders keep markers centered by rendering each logical marker cell
as a 2x2 surface raster. For example, `--border-cell 0.5` leaves half a
logical cell at each outer edge while preserving the requested physical tag
size and a full-cell gap between adjacent tags.

## Cuboid Grid Format

For cuboid targets, the grid specifies how many tags along each axis:

| Grid | Shape | Faces |
|------|-------|-------|
| `1x1x1` | Cube | 1 tag per face, 6 total |
| `2x2x2` | Cube | 4 tags per face, 24 total |
| `5x4x1` | Flat box | 20 tags top/bottom, narrow side strips |
| `1x1x3` | Tall pillar | 3 tags on tall sides, 1 on caps |

A 2D shorthand `RxC` is also supported for backward compatibility. For example, `2x3` expands to a cuboid.

For voxel targets, `grid` is inferred from the occupied voxel extent and each exposed voxel face gets one marker. The generated `config.json` contains a `markers` list with `id`, `face`, `voxel`, `normal`, `corners_mm`, and `face_corners_mm` for every marker.

## Supported Dictionaries

**ArUco:** `4x4_50`, `4x4_100`, `4x4_250`, `4x4_1000`, `5x5_*`, `6x6_*`, `7x7_*`, `aruco_original`

**AprilTag:** `apriltag_16h5`, `apriltag_25h9`, `apriltag_36h10`, `apriltag_36h11`

## Output

The output directory contains:

```text
my_target/
  cube.3mf              # Multi-color 3MF for Bambu Studio
  config.json           # Parameters needed by the detector
  thumbnail.png         # Preview with dimensions, tag IDs, and axis indicators
  mujoco/
    cube.xml            # MuJoCo MJCF model
    cube.obj            # Wavefront OBJ mesh with UV coordinates
    cube.mtl            # Material file referencing the atlas texture
    cube_atlas.png      # Texture atlas for MuJoCo/OBJ visualization
```

Cuboid `config.json` example:

```json
{
  "dict": "4x4_100",
  "grid": "2x2x2",
  "tag_ids": [0, 1, 2, "..."],
  "faces": {
    "+X": [0, 1, 2, 3],
    "-X": [4, 5, 6, 7],
    "+Y": [8, 9, 10, 11],
    "-Y": [12, 13, 14, 15],
    "+Z": [16, 17, 18, 19],
    "-Z": [20, 21, 22, 23]
  },
  "tag_size_mm": 24.0,
  "cell_size_mm": 4.0,
  "box_dims": [60.0, 60.0, 60.0]
}
```

Voxel-target configs additionally include explicit marker geometry:

```json
{
  "target": {
    "type": "voxel_cuboids",
    "voxel_size_mm": 24.0,
    "occupied_voxels": 5
  },
  "markers": [
    {
      "id": 0,
      "face": "-X",
      "voxel": [0, 0, 2],
      "normal": [-1.0, 0.0, 0.0],
      "corners_mm": [[-36.0, 9.0, 33.0], "..."]
    }
  ]
}
```

Load the generated MJCF in MuJoCo:

```bash
python -m mujoco.viewer --mjcf my_target/mujoco/cube.xml
```

The OBJ mesh and atlas texture are standard formats and can also be opened in Blender, MeshLab, and similar tools. The coordinate frame matches the detector's 6-DOF pose output with origin at the target bounding-box center and units in meters.

## How the Detector Works

The detection pipeline runs per frame and combines several techniques for robust 6-DOF pose estimation from 3D-printed fiducial targets.

### ArUco Detection

The OpenCV ArUco detector is configured with parameters optimized for markers printed on FDM surfaces:

- Sub-pixel corner refinement with `CORNER_REFINE_SUBPIX`.
- Wide adaptive thresholding for uneven surface texture and color bleed.
- Relaxed candidate filtering for oblique viewing angles.
- High-resolution bit sampling for perspective-distorted markers.

### Multi-Marker PnP

All detected tag corners across visible faces are aggregated into a single PnP solve:

- `>=6` points: `solvePnPRansac` with the SQPNP solver, 200 iterations, 3 px reprojection threshold, and 99% confidence.
- `4-5` points: direct `solvePnP` with SQPNP.
- Levenberg-Marquardt refinement with `solvePnPRefineLM` on the RANSAC inlier set.

Having tags on multiple faces of known 3D geometry eliminates the planar ambiguity and provides 3D point spread. For voxel targets, the PnP map comes directly from the explicit `markers[*].corners_mm` records rather than from cuboid grid assumptions.

### Error-State Kalman Filter

For video and streaming mode, an error-state extended Kalman filter provides temporal smoothing and prediction:

- Translation state: `[x, y, z, vx, vy, vz]`, using a constant-velocity model.
- Rotation state: multiplicative error-state formulation on unit quaternions.
- Adaptive measurement noise based on reprojection error, detected tag count, and RANSAC inlier ratio.
- Mahalanobis gating for outlier rejection and state re-initialization.

Pipeline summary:

```text
Frame -> Grayscale -> ArUco detect -> Filter to target IDs -> Identify visible faces
  -> Aggregate 2D-3D correspondences -> PnP -> LM refinement
  -> Kalman update -> Filtered 6-DOF pose
```

## Target Size Calculation

For cuboids, all dimensions are quantized to `cell_size`:

```text
cell_size = tag_size / marker_pixels
axis_cells = 2 * border + N * marker_pixels + (N - 1) * margin
axis_mm = axis_cells * cell_size
```

For `--grid 2x2x2 --dict 4x4_100 --tag-size 24 --margin-cell 1 --border-cell 1`:

- `cell_size = 24 / 6 = 4 mm`
- `axis_cells = 2*1 + 2*6 + 1*1 = 15`
- `box = 15 * 4 = 60 mm` per axis, producing a 60 x 60 x 60 mm cube

For voxel targets:

```text
face_cells = voxel_size / cell_size
box_dims = occupied_voxel_extent * voxel_size
```

Each exposed voxel face receives one centered marker. For `voxel_size_mm: 24` and `tag_size_mm: 18` with a 4x4 dictionary, `marker_pixels = 6`, so `cell_size = 3 mm` and each voxel face is an 8x8 cell grid.

## Face Coordinate System

Cuboid tags are assigned to faces in this order: `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`. Each face has a defined right and down direction when viewed from outside such that `cross(right, down) = outward normal`, ensuring correct triangle winding.

Voxel target markers use the same face coordinate convention, but each marker is tied to a specific exposed voxel face in the `markers` list.

## Technical Report and Citation

Read the technical report: [AprilCube: 3D-Printable Fiducial Targets for Reliable 6-DoF Pose Estimation](paper.pdf).

If you use AprilCube in research, please cite:

```bibtex
@software{park2026aprilcube,
  title={AprilCube: 3D-Printable Fiducial Targets for Reliable 6-DoF Pose Estimation},
  author={Park, Younghyo and Agrawal, Pulkit},
  year={2026},
  url={https://github.com/younghyopark/aprilcube},
}
```
