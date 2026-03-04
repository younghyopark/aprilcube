# ArUco Cube — 2x1x1

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `4x4_100` |
| Grid | 2x1x1 (X x Y x Z tags) |
| Box dimensions | 60 x 32 x 32 mm |
| Tag size | 24 mm (6x6 cells) |
| Cell size | 4 mm |
| Margin | 1 cell (4 mm) |
| Border | 1 cell (4 mm) |
| Total tags | 10 |
| Tag IDs | 0–9 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 0 |
| -X | 1 |
| +Y | 2, 3 |
| -Y | 4, 5 |
| +Z | 6, 7 |
| -Z | 8, 9 |

## Files

| File | Description |
|------|-------------|
| `cube.3mf` | Multi-color 3MF for Bambu Studio |
| `config.json` | Detector config (used by `detect_cube.py`) |
| `thumbnail.png` | 6-view preview |
| `mujoco/cube.xml` | MuJoCo MJCF model |
| `mujoco/cube.obj` | Wavefront OBJ mesh (UV-mapped) |
| `mujoco/cube.mtl` | OBJ material file |
| `mujoco/cube_atlas.png` | Texture atlas |

## Config JSON

```json
{
  "dict": "4x4_100",
  "grid": "2x1x1",
  "tag_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9
  ],
  "faces": {
    "+X": [
      0
    ],
    "-X": [
      1
    ],
    "+Y": [
      2,
      3
    ],
    "-Y": [
      4,
      5
    ],
    "+Z": [
      6,
      7
    ],
    "-Z": [
      8,
      9
    ]
  },
  "tag_size_mm": 24.0,
  "cell_size_mm": 4.0,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 6,
  "box_dims": [
    60.0,
    32.0,
    32.0
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 2x1x1 --dict 4x4_100 --tag-size 24 --margin-cell 1 --border-cell 1 -o 2x1x1_24_cube
```
