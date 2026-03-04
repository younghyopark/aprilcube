# aprilcube

Generate 3D-printable cubes/cuboids with ArUco or AprilTag fiducial markers on all 6 faces, then detect their 6-DOF pose from a camera.

## Overview

**aprilcube** is a two-part pipeline:

1. **`generate_cube.py`** — Creates a multi-color 3MF file with markers on every face, ready for dual-color 3D printing (Bambu Studio / AMS)
2. **`detect_cube.py`** — Detects the cube in a camera image and estimates its full 6-DOF pose (rotation + translation)

The cube geometry is fully parameterized: grid layout, tag dictionary, tag size, margins, borders. Both scripts share the same config, so the detector knows the exact 3D position of every tag corner.


## Installation

```bash
pip install opencv-contrib-python numpy
```

## Quick Start

```bash
# Generate a cube with 2x2 ArUco tags per face
python generate_cube.py --grid 2x2x2 --dict 4x4_100 --tag-size 24 -o my_cube

# Detect from webcam
python detect_cube.py --cube my_cube/config.json --viz

# Detect from an image
python detect_cube.py --cube my_cube/config.json --image photo.jpg --viz
```

## Generating Cubes

```
python generate_cube.py [options]
```

### Grid Format (`--grid WxHxD`)

The grid specifies how many tags along each axis (X, Y, Z):

| Grid | Shape | Faces |
|------|-------|-------|
| `1x1x1` | Cube | 1 tag per face, 6 total |
| `2x2x2` | Cube | 4 tags per face, 24 total |
| `5x4x1` | Flat box | 20 tags top/bottom, narrow side strips |
| `1x1x3` | Tall pillar | 3 tags on tall sides, 1 on caps |

A 2D shorthand `RxC` is also supported for backward compatibility (e.g., `2x3` expands to a cuboid).

### Options

| Arg | Default | Description |
|-----|---------|-------------|
| `-g, --grid` | `1x1x1` | Tags per dimension: `WxHxD` |
| `-d, --dict` | `4x4_50` | ArUco/AprilTag dictionary |
| `-t, --ids` | auto | Tag IDs: range (`0-23`) or comma-separated |
| `--tag-size` | `30` | Tag size in mm |
| `--cell-size` | — | Cell size in mm (alternative to `--tag-size`) |
| `--margin-cell` | `1` | Gap between adjacent tags, in cells |
| `--border-cell` | `1` | Outer border per face edge, in cells |
| `-o, --output` | `aruco_cube` | Output directory |
| `--extruder` | `1` | Bambu Studio extruder number |
| `--invert` | — | Swap black/white |

### Supported Dictionaries

**ArUco:** `4x4_50`, `4x4_100`, `4x4_250`, `4x4_1000`, `5x5_*`, `6x6_*`, `7x7_*`, `aruco_original`

**AprilTag:** `apriltag_16h5`, `apriltag_25h9`, `apriltag_36h10`, `apriltag_36h11`

### Output

The output directory contains:

```
my_cube/
  cube.3mf              # Multi-color 3MF for Bambu Studio (paint_color attribute)
  config.json           # All parameters needed by the detector
  thumbnail.png         # 6-view preview with dimensions, tag IDs, and axis indicators
  mujoco/
    cube.xml            # MuJoCo MJCF model (references cube.obj + cube_atlas.png)
    cube.obj            # Wavefront OBJ mesh with UV coordinates
    cube.mtl            # Material file referencing the atlas texture
    cube_atlas.png      # Texture atlas (3×2 grid of all 6 face textures)
```

**`config.json`** example:

```json
{
  "dict": "4x4_100",
  "grid": "2x2x2",
  "tag_ids": [0, 1, 2, ...],
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

**`mujoco/cube.xml`** can be loaded directly in MuJoCo for simulation:

```bash
python -m mujoco.viewer --mjcf my_cube/mujoco/cube.xml
```

The OBJ mesh + atlas texture are standard formats and can also be opened in Blender, MeshLab, etc. The coordinate frame matches `detect_cube.py`'s 6-DOF pose output (origin at cube center, units in meters).

### Examples

```bash
# Simple cube, one tag per face
python generate_cube.py --grid 1x1x1 --dict 4x4_50 --tag-size 30

# 2x2 cube with AprilTags
python generate_cube.py --grid 2x2x2 --dict apriltag_36h11 --tag-size 20

# Flat calibration box
python generate_cube.py --grid 5x4x1 --dict 4x4_100 --tag-size 15 -o flat_box

# Large cube with fine cell control
python generate_cube.py --grid 3x3x3 --dict 6x6_250 --cell-size 2.5 --margin-cell 2 --border-cell 2
```

## Detecting Pose

```
python detect_cube.py --cube <config.json> [options]
```

Loads the cube config and detects markers from a camera, image, or video. Estimates the cube's 6-DOF pose using multi-face PnP.

### Options

| Arg | Default | Description |
|-----|---------|-------------|
| `--cube` | required | Path to `config.json` |
| `--camera` | `0` | Camera index (default when no `--image`/`--video`) |
| `--image` | — | Single image file |
| `--video` | — | Video file |
| `--calib` | — | Camera calibration JSON |
| `--fx/--fy/--cx/--cy` | — | Direct intrinsic parameters |
| `--no-filter` | — | Disable temporal smoothing |
| `--viz` | — | Show debug visualization |

### Camera Calibration

Provide calibration via JSON file:

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3]
}
```

Or directly via CLI:

```bash
python detect_cube.py --cube my_cube/config.json --fx 800 --fy 800 --cx 320 --cy 240 --viz
```

If no calibration is provided, a rough default is used (assumes ~60 deg FOV).

### Examples

```bash
# Live webcam with visualization
python detect_cube.py --cube my_cube/config.json --viz

# Single image
python detect_cube.py --cube my_cube/config.json --image photo.jpg --viz

# Video with calibration
python detect_cube.py --cube my_cube/config.json --video recording.mp4 --calib calib.json --viz

# Specific camera index, no smoothing
python detect_cube.py --cube my_cube/config.json --camera 1 --no-filter --viz
```

### How It Works

1. **ArUco detection** with parameters tuned for 3D-printed surfaces (sub-pixel corner refinement, adaptive thresholding)
2. **Multi-face PnP**: all detected tag corners across visible faces are aggregated into a single `solvePnPRansac` call with SQPNP solver (no planar degeneracy)
3. **Levenberg-Marquardt refinement** on the inlier set
4. **Temporal filtering** (EMA) for smooth tracking in video/camera mode

## Printing

Designed for dual-color FDM printing on Bambu Lab printers with AMS (Automatic Material System).

### Supported Printers

Any Bambu Lab printer with AMS or AMS Lite:

- **Bambu X1 / X1C** — AMS (4 slots)
- **Bambu P1S / P1P** — AMS Lite (4 slots)
- **Bambu A1 / A1 mini** — AMS Lite (4 slots)

### Setup

1. Open `cube.3mf` in Bambu Studio
2. Assign filament colors: extruder 1 = black, extruder 2 = white (PLA recommended)
3. Slice and print — the 3MF uses `paint_color` attributes for automatic color assignment

## How Cube Size Is Calculated

All dimensions are quantized to cell_size:

```
cell_size = tag_size / marker_pixels
axis_cells = 2 * border + N * marker_pixels + (N-1) * margin
axis_mm = axis_cells * cell_size
```

For `--grid 2x2x2 --dict 4x4_100 --tag-size 24 --margin-cell 1 --border-cell 1`:
- `cell_size = 24 / 6 = 4 mm`
- `axis_cells = 2*1 + 2*6 + 1*1 = 15`
- `box = 15 * 4 = 60 mm` per axis → 60 x 60 x 60 mm cube

## Face Coordinate System

Tags are assigned to faces in this order: +X, -X, +Y, -Y, +Z, -Z. Each face has a defined "right" and "down" direction (viewed from outside) such that `cross(right, down) = outward normal`, ensuring correct triangle winding.

## License

MIT
