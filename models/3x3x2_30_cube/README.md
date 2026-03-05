# ArUco Cube — 3x3x2

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `4x4_100` |
| Grid | 3x3x2 (X x Y x Z tags) |
| Box dimensions | 110 x 110 x 75 mm |
| Tag size | 30 mm (6x6 cells) |
| Cell size | 5 mm |
| Margin | 1 cell (5 mm) |
| Border | 1 cell (5 mm) |
| Total tags | 42 |
| Tag IDs | 0–41 |

## Face Layout

| Face | Tag IDs |
|------|---------|
| +X | 0, 1, 2, 3, 4, 5 |
| -X | 6, 7, 8, 9, 10, 11 |
| +Y | 12, 13, 14, 15, 16, 17 |
| -Y | 18, 19, 20, 21, 22, 23 |
| +Z | 24, 25, 26, 27, 28, 29, 30, 31, 32 |
| -Z | 33, 34, 35, 36, 37, 38, 39, 40, 41 |

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
  "grid": "3x3x2",
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
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41
  ],
  "faces": {
    "+X": [
      0,
      1,
      2,
      3,
      4,
      5
    ],
    "-X": [
      6,
      7,
      8,
      9,
      10,
      11
    ],
    "+Y": [
      12,
      13,
      14,
      15,
      16,
      17
    ],
    "-Y": [
      18,
      19,
      20,
      21,
      22,
      23
    ],
    "+Z": [
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31,
      32
    ],
    "-Z": [
      33,
      34,
      35,
      36,
      37,
      38,
      39,
      40,
      41
    ]
  },
  "tag_size_mm": 30.0,
  "cell_size_mm": 5.0,
  "margin_cells": 1,
  "border_cells": 1,
  "marker_pixels": 6,
  "box_dims": [
    110.0,
    110.0,
    75.0
  ]
}
```

## Regenerate

```bash
python generate_cube.py --grid 3x3x2 --dict 4x4_100 --tag-size 30 --margin-cell 1 --border-cell 1 -o 3x3x2_30_cube
```
