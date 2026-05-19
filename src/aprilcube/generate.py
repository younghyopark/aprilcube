#!/usr/bin/env python3
"""Generate a 3MF cuboid with ArUco/AprilTag markers for multi-color 3D printing.

For a square grid (e.g. 2x2), all 6 faces are identical → cube.
For a non-square grid (e.g. 2x3):
  - 4 large faces get R×C tags
  - 2 small end faces get min(R,C)×min(R,C) tags
  - Box proportions: min(R,C) × min(R,C) × max(R,C)
"""

import argparse
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dictionary name mapping
# ---------------------------------------------------------------------------
DICT_MAP = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "apriltag_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "aruco_original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# ---------------------------------------------------------------------------
# Face definitions: (name, normal_axis, normal_sign, right_axis, right_sign, down_axis, down_sign)
# "right" = column-increasing direction when viewed from outside
# "down"  = row-increasing direction when viewed from outside
# cross(right, down) must equal outward normal for correct triangle winding
# ---------------------------------------------------------------------------
FACE_DEFS = [
    # name  normal_ax  normal_sign  right_ax  right_sign  down_ax  down_sign
    ("+X",  0,         +1,          1,        -1,         2,       -1),
    ("-X",  0,         -1,          1,        +1,         2,       -1),
    ("+Y",  1,         +1,          0,        +1,         2,       -1),
    ("-Y",  1,         -1,          0,        -1,         2,       -1),
    ("+Z",  2,         +1,          0,        +1,         1,       +1),
    ("-Z",  2,         -1,          0,        +1,         1,       -1),
]

BAMBU_STUDIO_VERSION = "02.00.00.00"
BAMBU_STUDIO_APPLICATION = f"BambuStudio-{BAMBU_STUDIO_VERSION}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class CubeConfig:
    grid_x: int           # tags in X dimension
    grid_y: int           # tags in Y dimension
    grid_z: int           # tags in Z dimension
    dict_id: int
    dict_name: str
    tag_ids: list[int]
    tag_size_mm: float    # 0 if computed from cell_size
    margin_cells: int = 1    # cells between adjacent tags
    border_cells: int = 1    # cells of outer border per face edge
    cell_size_mm: float = 0.0  # 0 = derive from tag_size
    extruder: int = 1
    invert: bool = False
    target_type: str = "cuboid"
    marker_corners: dict[int, list[list[float]]] | None = None
    marker_normals: dict[int, list[float]] | None = None

    # derived (set by compute())
    marker_pixels: int = 0
    cell_size: float = 0.0
    x_cells: int = 0          # total cells in X dimension
    y_cells: int = 0          # total cells in Y dimension
    z_cells: int = 0          # total cells in Z dimension
    box_dims: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (X, Y, Z) mm

    def _axis_cells(self, n_tags: int) -> int:
        mp, mc, bc = self.marker_pixels, self.margin_cells, self.border_cells
        return 2 * bc + n_tags * mp + max(0, n_tags - 1) * mc

    def compute(self):
        dictionary = cv2.aruco.getPredefinedDictionary(self.dict_id)
        self.marker_pixels = dictionary.markerSize + 2  # +2 for 1-cell border

        if self.cell_size_mm > 0:
            self.cell_size = self.cell_size_mm
            self.tag_size_mm = self.cell_size * self.marker_pixels
        else:
            self.cell_size = self.tag_size_mm / self.marker_pixels

        self.x_cells = self._axis_cells(self.grid_x)
        self.y_cells = self._axis_cells(self.grid_y)
        self.z_cells = self._axis_cells(self.grid_z)
        self.box_dims = (
            self.x_cells * self.cell_size,
            self.y_cells * self.cell_size,
            self.z_cells * self.cell_size,
        )

    def face_layout(self, face_def: tuple) -> tuple[int, int, int, int]:
        """Return (face_rows, face_cols, down_cells, right_cells) for a face."""
        _name, _nax, _ns, right_ax, _rs, down_ax, _ds = face_def
        grid = [self.grid_x, self.grid_y, self.grid_z]
        cells = [self.x_cells, self.y_cells, self.z_cells]
        return grid[down_ax], grid[right_ax], cells[down_ax], cells[right_ax]

    def total_tags(self) -> int:
        gx, gy, gz = self.grid_x, self.grid_y, self.grid_z
        return 2 * (gx * gy + gx * gz + gy * gz)


@dataclass
class GenerationSpec:
    """Normalized user intent for one generation run.

    The current generator supports ``shape_type="cuboid"``.  The shape field is
    kept explicit so the YAML schema can grow into voxel/polycube targets
    without changing the CLI contract again.
    """
    output: str | None = None
    dict_name: str | None = None
    grid: str | list[int] | tuple[int, ...] | None = None
    ids: Any = None
    tag_size_mm: float | None = None
    cell_size_mm: float | None = None
    margin_cells: int | None = None
    border_cells: int | None = None
    extruder: int | None = None
    invert: bool | None = None
    shape_type: str = "cuboid"
    shape: dict[str, Any] | None = None
    source_path: str | None = None


# ---------------------------------------------------------------------------
# Tag pattern generation
# ---------------------------------------------------------------------------
class TagPatternGenerator:
    def __init__(self, dict_id: int):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.marker_pixels = self.dictionary.markerSize + 2

    @property
    def max_id(self) -> int:
        return len(self.dictionary.bytesList)

    def generate(self, tag_id: int) -> np.ndarray:
        """Return boolean grid (True=black) of shape (marker_pixels, marker_pixels)."""
        img = cv2.aruco.generateImageMarker(self.dictionary, tag_id, self.marker_pixels)
        return np.fliplr(img < 128)


# ---------------------------------------------------------------------------
# Face pixel grid layout
# ---------------------------------------------------------------------------
def build_face_grid(
    tag_patterns: list[np.ndarray],
    face_rows: int,
    face_cols: int,
    down_cells: int,
    right_cells: int,
    marker_pixels: int,
    margin_cells: int,
    invert: bool,
) -> np.ndarray:
    """Compose the full pixel grid for one face.  True = black."""
    grid = np.zeros((down_cells, right_cells), dtype=bool)

    tag_block_w = face_cols * marker_pixels + max(0, face_cols - 1) * margin_cells
    tag_block_h = face_rows * marker_pixels + max(0, face_rows - 1) * margin_cells
    row_off = (down_cells - tag_block_h) // 2
    col_off = (right_cells - tag_block_w) // 2

    for r in range(face_rows):
        for c in range(face_cols):
            idx = r * face_cols + c
            if idx >= len(tag_patterns):
                continue
            pat = tag_patterns[idx]
            rs = row_off + r * (marker_pixels + margin_cells)
            cs = col_off + c * (marker_pixels + margin_cells)
            grid[rs:rs + marker_pixels, cs:cs + marker_pixels] = pat

    if invert:
        grid = ~grid
    return grid


def render_face_texture(grid: np.ndarray, pixels_per_cell: int = 8) -> np.ndarray:
    """Render face grid as grayscale image. True=black(0), False=white(255)."""
    img = np.where(grid, 0, 255).astype(np.uint8)
    return np.kron(img, np.ones((pixels_per_cell, pixels_per_cell), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Texture atlas + OBJ + MuJoCo writer
# ---------------------------------------------------------------------------
FACE_TEX_NAMES = {
    "+X": "px", "-X": "nx", "+Y": "py", "-Y": "ny", "+Z": "pz", "-Z": "nz",
}

# Atlas layout: 3 columns × 2 rows
_ATLAS_LAYOUT = [
    ("+X", "-X", "+Y"),  # row 0
    ("-Y", "+Z", "-Z"),  # row 1
]


def build_texture_atlas(
    face_textures: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Build a single texture atlas from 6 face textures (3×2 grid).

    Returns (atlas_image, regions) where regions maps face_name →
    (x_offset, y_offset, width, height) in pixels.
    """
    # Compute column widths and row heights
    col_widths = [0, 0, 0]
    row_heights = [0, 0]
    for r, row_names in enumerate(_ATLAS_LAYOUT):
        for c, name in enumerate(row_names):
            tex = face_textures[name]
            h, w = tex.shape[:2]
            col_widths[c] = max(col_widths[c], w)
            row_heights[r] = max(row_heights[r], h)

    atlas_w = sum(col_widths)
    atlas_h = sum(row_heights)
    atlas = np.full((atlas_h, atlas_w), 255, dtype=np.uint8)

    regions: dict[str, tuple[int, int, int, int]] = {}
    y_off = 0
    for r, row_names in enumerate(_ATLAS_LAYOUT):
        x_off = 0
        for c, name in enumerate(row_names):
            tex = face_textures[name]
            th, tw = tex.shape[:2]
            atlas[y_off:y_off + th, x_off:x_off + tw] = tex
            regions[name] = (x_off, y_off, tw, th)
            x_off += col_widths[c]
        y_off += row_heights[r]

    return atlas, regions


def write_cube_obj(
    config: CubeConfig,
    atlas_regions: dict[str, tuple[int, int, int, int]],
    atlas_w: int,
    atlas_h: int,
    obj_path: str,
    mtl_path: str,
):
    """Write Wavefront OBJ + MTL for the cube with UV-mapped atlas texture."""
    bx, by, bz = config.box_dims
    # Half-extents in meters
    hx, hy, hz = bx / 2000.0, by / 2000.0, bz / 2000.0

    # 8 cube corner vertices (centered at origin)
    # Index scheme:  bit0=X sign, bit1=Y sign, bit2=Z sign
    #   0=(-,-,-)  1=(+,-,-)  2=(-,+,-)  3=(+,+,-)
    #   4=(-,-,+)  5=(+,-,+)  6=(-,+,+)  7=(+,+,+)
    corners = [
        (-hx, -hy, -hz),  # 0
        (+hx, -hy, -hz),  # 1
        (-hx, +hy, -hz),  # 2
        (+hx, +hy, -hz),  # 3
        (-hx, -hy, +hz),  # 4
        (+hx, -hy, +hz),  # 5
        (-hx, +hy, +hz),  # 6
        (+hx, +hy, +hz),  # 7
    ]

    vt_list: list[tuple[float, float]] = []  # UV coordinates
    face_lines: list[str] = []  # OBJ face lines

    for face_def in FACE_DEFS:
        name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
        x_off, y_off, tw, th = atlas_regions[name]

        # UV corners: TL, TR, BR, BL of the face as seen from outside
        u0 = x_off / atlas_w
        u1 = (x_off + tw) / atlas_w
        v0 = 1.0 - (y_off + th) / atlas_h  # bottom of face region
        v1 = 1.0 - y_off / atlas_h          # top of face region

        vt_base = len(vt_list) + 1  # OBJ is 1-indexed
        vt_list.append((u0, v1))  # TL uv
        vt_list.append((u1, v1))  # TR uv
        vt_list.append((u1, v0))  # BR uv
        vt_list.append((u0, v0))  # BL uv

        # Compute which cube corner indices correspond to TL, TR, BR, BL
        # TL: row=0,col=0 → right_neg, down_neg
        # TR: row=0,col=max → right_pos, down_neg
        # BR: row=max,col=max → right_pos, down_pos
        # BL: row=max,col=0 → right_neg, down_pos
        def _corner_index(r_sign: int, d_sign: int) -> int:
            """Map (right_value_sign, down_value_sign) to corner index 0-7."""
            # r_sign/d_sign are the signs of the coordinate on right_ax/down_ax
            c = [0, 0, 0]  # signs for X, Y, Z: 0=negative, 1=positive
            c[normal_ax] = 1 if normal_sign > 0 else 0
            c[right_ax] = 1 if r_sign > 0 else 0
            c[down_ax] = 1 if d_sign > 0 else 0
            return c[0] + 2 * c[1] + 4 * c[2]

        # right_neg = right_sign * (-half) → sign is -right_sign
        # right_pos = right_sign * (+half) → sign is +right_sign
        # down_neg = down_sign * (-half) → sign is -down_sign
        # down_pos = down_sign * (+half) → sign is +down_sign
        tl_i = _corner_index(-right_sign, -down_sign)
        tr_i = _corner_index(+right_sign, -down_sign)
        br_i = _corner_index(+right_sign, +down_sign)
        bl_i = _corner_index(-right_sign, +down_sign)

        # OBJ face: CCW winding from outside = TL TR BR BL
        # (cross(right, down) = outward normal, per FACE_DEFS convention)
        v_tl = tl_i + 1  # OBJ 1-indexed
        v_tr = tr_i + 1
        v_br = br_i + 1
        v_bl = bl_i + 1
        t_tl = vt_base
        t_tr = vt_base + 1
        t_br = vt_base + 2
        t_bl = vt_base + 3
        face_lines.append(
            f"f {v_tl}/{t_tl} {v_tr}/{t_tr} {v_br}/{t_br} {v_bl}/{t_bl}"
        )

    # Write MTL
    mtl_name = os.path.basename(mtl_path)
    with open(mtl_path, "w") as f:
        f.write("# ArUco cube material\n")
        f.write("newmtl cube_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("map_Kd cube_atlas.png\n")

    # Write OBJ
    with open(obj_path, "w") as f:
        f.write("# ArUco cube mesh\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl cube_material\n\n")
        for x, y, z in corners:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for u, v in vt_list:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")
        for fl in face_lines:
            f.write(fl + "\n")

    print(f"Wrote {obj_path}")


def write_mujoco_xml(config: CubeConfig, xml_path: str):
    """Write MuJoCo MJCF XML referencing the OBJ mesh and atlas texture."""
    bx, by, bz = config.box_dims
    hx_m, hy_m, hz_m = bx / 2000.0, by / 2000.0, bz / 2000.0

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mujoco model="aruco_cube">\n'
        f'  <!-- Box: {bx:.4g} x {by:.4g} x {bz:.4g} mm -->\n'
        '  <!-- Origin at cube center, XYZ axes match detect_cube.py -->\n'
        '  <!-- Units: meters -->\n'
        '\n'
        '  <compiler angle="radian" meshdir="."/>\n'
        '\n'
        '  <asset>\n'
        '    <texture name="cube_tex" type="2d" file="cube_atlas.png"/>\n'
        '    <material name="cube_mat" texture="cube_tex" specular="0.1" shininess="0.1"/>\n'
        '    <mesh name="cube_mesh" file="cube.obj"/>\n'
        '  </asset>\n'
        '\n'
        '  <worldbody>\n'
        '    <body name="cube" pos="0 0 0">\n'
        '      <freejoint name="cube_joint"/>\n'
        f'      <geom name="cube_visual" type="mesh" mesh="cube_mesh" material="cube_mat"'
        f' contype="0" conaffinity="0" group="1" density="0"/>\n'
        f'      <geom name="cube_collision" type="box"'
        f' size="{hx_m:.6f} {hy_m:.6f} {hz_m:.6f}"'
        f' density="1250" rgba="0.5 0.5 0.5 0" contype="1" conaffinity="1" group="2"/>\n'
        '      <site name="cube_center" pos="0 0 0" size="0.001"/>\n'
        '    </body>\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    )
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"Wrote {xml_path}")


def write_mujoco_assets(config: CubeConfig, face_grids: dict[str, np.ndarray],
                        out_dir: str, pixels_per_cell: int = 8):
    """Write all MuJoCo assets: atlas texture, OBJ mesh, MTL, and MJCF XML."""
    mj_dir = os.path.join(out_dir, "mujoco")
    os.makedirs(mj_dir, exist_ok=True)

    # Render face textures
    face_textures = {
        name: render_face_texture(grid, pixels_per_cell)
        for name, grid in face_grids.items()
    }

    # Build atlas
    atlas, regions = build_texture_atlas(face_textures)
    atlas_path = os.path.join(mj_dir, "cube_atlas.png")
    cv2.imwrite(atlas_path, atlas)
    print(f"Wrote {atlas_path} ({atlas.shape[1]}x{atlas.shape[0]})")

    atlas_h, atlas_w = atlas.shape[:2]

    # Write OBJ + MTL
    obj_path = os.path.join(mj_dir, "cube.obj")
    mtl_path = os.path.join(mj_dir, "cube.mtl")
    write_cube_obj(config, regions, atlas_w, atlas_h, obj_path, mtl_path)

    # Write MuJoCo XML
    xml_path = os.path.join(mj_dir, "cube.xml")
    write_mujoco_xml(config, xml_path)


def build_voxel_texture_atlas(
    marker_faces: list[dict[str, Any]],
    pixels_per_cell: int = 8,
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    """Build a texture atlas for arbitrary exposed voxel faces."""
    textures = [
        render_face_texture(face["grid"], pixels_per_cell)
        for face in marker_faces
    ]
    if not textures:
        raise ValueError("cannot build texture atlas with no marker faces")

    tile_h = max(tex.shape[0] for tex in textures)
    tile_w = max(tex.shape[1] for tex in textures)
    cols = int(np.ceil(np.sqrt(len(textures))))
    rows = int(np.ceil(len(textures) / cols))
    atlas = np.full((rows * tile_h, cols * tile_w), 255, dtype=np.uint8)
    regions: dict[int, tuple[int, int, int, int]] = {}

    for idx, tex in enumerate(textures):
        row = idx // cols
        col = idx % cols
        y_off = row * tile_h
        x_off = col * tile_w
        th, tw = tex.shape[:2]
        atlas[y_off:y_off + th, x_off:x_off + tw] = tex
        regions[idx] = (x_off, y_off, tw, th)

    return atlas, regions


def write_voxel_obj(
    marker_faces: list[dict[str, Any]],
    atlas_regions: dict[int, tuple[int, int, int, int]],
    atlas_w: int,
    atlas_h: int,
    obj_path: str,
    mtl_path: str,
) -> None:
    """Write Wavefront OBJ + MTL for a voxel target with per-face UVs."""
    mtl_name = os.path.basename(mtl_path)
    with open(mtl_path, "w") as f:
        f.write("# AprilCube voxel target material\n")
        f.write("newmtl cube_material\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("map_Kd cube_atlas.png\n")

    vertices: list[tuple[float, float, float]] = []
    vt_list: list[tuple[float, float]] = []
    face_lines: list[str] = []

    for idx, face in enumerate(marker_faces):
        x_off, y_off, tw, th = atlas_regions[idx]
        u0 = x_off / atlas_w
        u1 = (x_off + tw) / atlas_w
        v0 = 1.0 - (y_off + th) / atlas_h
        v1 = 1.0 - y_off / atlas_h

        v_base = len(vertices) + 1
        for p in face["face_corners_mm"]:
            vertices.append((p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0))

        vt_base = len(vt_list) + 1
        vt_list.extend([
            (u0, v1),
            (u1, v1),
            (u1, v0),
            (u0, v0),
        ])
        face_lines.append(
            f"f {v_base}/{vt_base} {v_base + 1}/{vt_base + 1} "
            f"{v_base + 2}/{vt_base + 2} {v_base + 3}/{vt_base + 3}"
        )

    with open(obj_path, "w") as f:
        f.write("# AprilCube voxel target mesh\n")
        f.write(f"mtllib {mtl_name}\n")
        f.write("usemtl cube_material\n\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for u, v in vt_list:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        f.write("\n")
        for fl in face_lines:
            f.write(fl + "\n")

    print(f"Wrote {obj_path}")


def _voxel_collision_boxes(
    source_cuboids: list[dict[str, Any]],
    occupied: set[tuple[int, int, int]],
    voxel_size: float,
    center_abs: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Return collision boxes for voxel target MJCF export."""
    boxes = source_cuboids
    if not boxes:
        boxes = [
            {"name": f"voxel_{i}", "origin": list(v), "size": [1, 1, 1]}
            for i, v in enumerate(sorted(occupied))
        ]

    result = []
    for idx, box in enumerate(boxes):
        origin = _vec3f(box["origin"], f"collision box {idx}.origin")
        size = _vec3f(box["size"], f"collision box {idx}.size")
        center = tuple(
            (origin[i] + size[i] / 2.0) * voxel_size - center_abs[i]
            for i in range(3)
        )
        half = tuple(size[i] * voxel_size / 2.0 for i in range(3))
        result.append({
            "name": str(box.get("name", f"voxel_collision_{idx}")),
            "center_m": tuple(v / 1000.0 for v in center),
            "half_m": tuple(v / 1000.0 for v in half),
        })
    return result


def write_voxel_mujoco_xml(
    config: CubeConfig,
    xml_path: str,
    collision_boxes: list[dict[str, Any]],
) -> None:
    """Write MJCF XML for a voxel target."""
    bx, by, bz = config.box_dims
    geom_lines = []
    for idx, box in enumerate(collision_boxes):
        cx, cy, cz = box["center_m"]
        hx, hy, hz = box["half_m"]
        name = box["name"].replace(" ", "_")
        geom_lines.append(
            f'      <geom name="{name}_collision" type="box"'
            f' pos="{cx:.6f} {cy:.6f} {cz:.6f}"'
            f' size="{hx:.6f} {hy:.6f} {hz:.6f}"'
            f' density="1250" rgba="0.5 0.5 0.5 0" contype="1" conaffinity="1" group="2"/>\n'
        )

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mujoco model="aprilcube_voxel_target">\n'
        f'  <!-- Bounding box: {bx:.4g} x {by:.4g} x {bz:.4g} mm -->\n'
        '  <!-- Origin at target center, units in meters -->\n'
        '\n'
        '  <compiler angle="radian" meshdir="."/>\n'
        '\n'
        '  <asset>\n'
        '    <texture name="cube_tex" type="2d" file="cube_atlas.png"/>\n'
        '    <material name="cube_mat" texture="cube_tex" specular="0.1" shininess="0.1"/>\n'
        '    <mesh name="cube_mesh" file="cube.obj"/>\n'
        '  </asset>\n'
        '\n'
        '  <worldbody>\n'
        '    <body name="cube" pos="0 0 0">\n'
        '      <freejoint name="cube_joint"/>\n'
        '      <geom name="cube_visual" type="mesh" mesh="cube_mesh" material="cube_mat"'
        ' contype="0" conaffinity="0" group="1" density="0"/>\n'
        f'{"".join(geom_lines)}'
        '      <site name="cube_center" pos="0 0 0" size="0.001"/>\n'
        '    </body>\n'
        '  </worldbody>\n'
        '</mujoco>\n'
    )
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"Wrote {xml_path}")


def write_voxel_mujoco_assets(
    config: CubeConfig,
    marker_faces: list[dict[str, Any]],
    source_cuboids: list[dict[str, Any]],
    occupied: set[tuple[int, int, int]],
    voxel_size: float,
    center_abs: tuple[float, float, float],
    out_dir: str,
    pixels_per_cell: int = 8,
) -> None:
    """Write MuJoCo assets for an arbitrary voxel target."""
    mj_dir = os.path.join(out_dir, "mujoco")
    os.makedirs(mj_dir, exist_ok=True)

    atlas, regions = build_voxel_texture_atlas(marker_faces, pixels_per_cell)
    atlas_path = os.path.join(mj_dir, "cube_atlas.png")
    cv2.imwrite(atlas_path, atlas)
    print(f"Wrote {atlas_path} ({atlas.shape[1]}x{atlas.shape[0]})")

    atlas_h, atlas_w = atlas.shape[:2]
    obj_path = os.path.join(mj_dir, "cube.obj")
    mtl_path = os.path.join(mj_dir, "cube.mtl")
    write_voxel_obj(marker_faces, regions, atlas_w, atlas_h, obj_path, mtl_path)

    collision_boxes = _voxel_collision_boxes(source_cuboids, occupied, voxel_size, center_abs)
    xml_path = os.path.join(mj_dir, "cube.xml")
    write_voxel_mujoco_xml(config, xml_path, collision_boxes)


# ---------------------------------------------------------------------------
# Thumbnail renderer
# ---------------------------------------------------------------------------
def _face_quad_corners(face_def: tuple, box_dims: tuple) -> np.ndarray:
    """Return 4 corner positions [TL, TR, BR, BL] for a face quad (mm)."""
    _name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
    half = [box_dims[0] / 2, box_dims[1] / 2, box_dims[2] / 2]
    face_pos = normal_sign * half[normal_ax]
    rn = right_sign * (-half[right_ax])
    rp = right_sign * (half[right_ax])
    dn = down_sign * (-half[down_ax])
    dp = down_sign * (half[down_ax])

    def _corner(rv, dv):
        c = [0.0, 0.0, 0.0]
        c[normal_ax] = face_pos
        c[right_ax] = rv
        c[down_ax] = dv
        return c

    return np.array([_corner(rn, dn), _corner(rp, dn),
                     _corner(rp, dp), _corner(rn, dp)], dtype=np.float64)


def _camera_from_angles(elev_deg: float, azim_deg: float, distance: float):
    """Compute (rvec, tvec) for a camera looking at origin from spherical coords."""
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    cam_pos = np.array([
        distance * np.cos(elev) * np.cos(azim),
        distance * np.cos(elev) * np.sin(azim),
        distance * np.sin(elev),
    ])
    fwd = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # OpenCV camera: X-right, Y-down, Z-forward
    R = np.array([right, -up, fwd], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ cam_pos).reshape(3, 1)
    return rvec, tvec, fwd


def _build_tag_centers(config: CubeConfig) -> dict[str, list[tuple[int, np.ndarray, np.ndarray]]]:
    """Return {face_name: [(tag_id, center_3d, normal_3d), ...]} for all faces."""
    result: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    mp = config.marker_pixels
    cs = config.cell_size
    id_cursor = 0

    for face_def in FACE_DEFS:
        name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
        face_rows, face_cols, down_cells, right_cells = config.face_layout(face_def)
        n_tags = face_rows * face_cols

        tag_block_w = face_cols * mp + max(0, face_cols - 1) * config.margin_cells
        tag_block_h = face_rows * mp + max(0, face_rows - 1) * config.margin_cells
        row_off = (down_cells - tag_block_h) // 2
        col_off = (right_cells - tag_block_w) // 2

        half = [config.box_dims[0] / 2, config.box_dims[1] / 2, config.box_dims[2] / 2]
        face_pos = normal_sign * half[normal_ax]
        normal = np.zeros(3)
        normal[normal_ax] = float(normal_sign)

        entries = []
        for r in range(face_rows):
            for c in range(face_cols):
                idx = r * face_cols + c
                if id_cursor + idx >= len(config.tag_ids):
                    break
                tag_id = config.tag_ids[id_cursor + idx]
                rc = row_off + r * (mp + config.margin_cells) + mp / 2
                cc = col_off + c * (mp + config.margin_cells) + mp / 2
                u = right_sign * (-half[right_ax] + cc * cs)
                v = down_sign * (-half[down_ax] + rc * cs)
                pt = np.zeros(3)
                pt[normal_ax] = face_pos
                pt[right_ax] = u
                pt[down_ax] = v
                entries.append((tag_id, pt, normal))

        result[name] = entries
        id_cursor += n_tags

    return result


def _render_cube_view(
    face_textures: dict[str, np.ndarray],
    config: CubeConfig,
    elev_deg: float,
    azim_deg: float,
    view_w: int = 400,
    view_h: int = 400,
    show_dims: bool = True,
) -> np.ndarray:
    """Render one view of the textured cube."""
    bg = np.full((view_h, view_w, 3), 240, dtype=np.uint8)

    diag = np.sqrt(sum(d ** 2 for d in config.box_dims))
    fx = fy = view_w * 1.8
    cam_matrix = np.array([[fx, 0, view_w / 2],
                           [0, fy, view_h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(5)

    rvec, tvec, fwd = _camera_from_angles(elev_deg, azim_deg, diag * 2.5)
    view_dir = -fwd  # from origin toward camera

    # Collect visible faces sorted back-to-front
    R_cam, _ = cv2.Rodrigues(rvec)
    visible: list[tuple[float, str, tuple]] = []
    for face_def in FACE_DEFS:
        name = face_def[0]
        normal = np.zeros(3)
        normal[face_def[1]] = face_def[2]
        if np.dot(normal, view_dir) > 0:
            corners = _face_quad_corners(face_def, config.box_dims)
            center_cam = (R_cam @ corners.mean(axis=0) + tvec.flatten())[2]
            visible.append((center_cam, name, corners))
    visible.sort(reverse=True)

    # Paint faces (back to front)
    for _, name, corners_3d in visible:
        projected, _ = cv2.projectPoints(corners_3d, rvec, tvec,
                                         cam_matrix, dist_coeffs)
        pts_2d = projected.reshape(-1, 2).astype(np.float32)

        tex = face_textures[name]
        th, tw = tex.shape[:2]
        src_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]],
                           dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, pts_2d)
        warped = cv2.warpPerspective(tex, M, (view_w, view_h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((th, tw), 255, dtype=np.uint8),
                                   M, (view_w, view_h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        bg = np.where(mask[:, :, np.newaxis] > 0, warped, bg)

    # Draw tag ID labels outside the cube with leader lines
    tag_centers = _build_tag_centers(config)
    visible_names = {name for _, name, _ in visible}
    label_offset = max(config.box_dims) * 0.18
    for name in visible_names:
        for tag_id, center_3d, normal_3d in tag_centers[name]:
            # Project tag center and an offset point along the face normal
            label_3d = center_3d + normal_3d * label_offset
            pts_3d = np.array([center_3d, label_3d], dtype=np.float64)
            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec,
                                        cam_matrix, dist_coeffs)
            p = proj.reshape(-1, 2).astype(int)
            tag_pt = tuple(p[0])
            lbl_pt = tuple(p[1])

            # Leader line
            cv2.line(bg, tag_pt, lbl_pt, (80, 80, 80), 1, cv2.LINE_AA)
            # Small dot on the tag
            cv2.circle(bg, tag_pt, 2, (80, 80, 80), cv2.FILLED)

            # Label with background pill
            label = str(tag_id)
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.55
            thick = 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            pad = 3
            lx, ly = lbl_pt
            cv2.rectangle(bg, (lx - tw // 2 - pad, ly - th // 2 - pad),
                          (lx + tw // 2 + pad, ly + th // 2 + pad),
                          (255, 255, 255), cv2.FILLED)
            cv2.rectangle(bg, (lx - tw // 2 - pad, ly - th // 2 - pad),
                          (lx + tw // 2 + pad, ly + th // 2 + pad),
                          (80, 80, 80), 1)
            cv2.putText(bg, label, (lx - tw // 2, ly + th // 2), font, scale,
                        (0, 0, 0), thick, cv2.LINE_AA)

    # Draw wireframe edges on top
    box = config.box_dims
    hx, hy, hz = box[0] / 2, box[1] / 2, box[2] / 2
    box_corners = np.array([
        [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
    ], dtype=np.float64)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    proj_box, _ = cv2.projectPoints(box_corners, rvec, tvec,
                                    cam_matrix, dist_coeffs)
    pts = proj_box.reshape(-1, 2).astype(int)
    for i, j in edges:
        cv2.line(bg, tuple(pts[i]), tuple(pts[j]), (180, 180, 180), 1,
                 cv2.LINE_AA)

    # Draw dimension annotations along outermost edges (first panel only)
    if show_dims:
        pts_f = proj_box.reshape(-1, 2)  # float precision
        center_2d = pts_f.mean(axis=0)
        dim_edge_candidates = {
            0: [(0,1), (3,2), (4,5), (7,6)],  # X edges
            1: [(0,3), (1,2), (4,7), (5,6)],  # Y edges
            2: [(0,4), (1,5), (2,6), (3,7)],  # Z edges
        }
        dim_color = (120, 120, 120)
        dim_font = cv2.FONT_HERSHEY_SIMPLEX
        dim_scale = 0.38
        dim_thick = 1
        gap = 15  # pixel offset from edge

        for ax_dim, cands in dim_edge_candidates.items():
            dim_mm = config.box_dims[ax_dim]
            best_dist = -1.0
            best_ij = cands[0]
            for (ci, cj) in cands:
                mid = (pts_f[ci] + pts_f[cj]) / 2.0
                d = np.linalg.norm(mid - center_2d)
                if d > best_dist:
                    best_dist = d
                    best_ij = (ci, cj)

            p1 = pts_f[best_ij[0]].copy()
            p2 = pts_f[best_ij[1]].copy()
            edge_vec = p2 - p1
            edge_len = np.linalg.norm(edge_vec)
            if edge_len < 5:
                continue
            edge_unit = edge_vec / edge_len
            perp = np.array([-edge_unit[1], edge_unit[0]])
            mid = (p1 + p2) / 2.0
            if np.dot(perp, mid - center_2d) < 0:
                perp = -perp

            q1 = p1 + perp * gap
            q2 = p2 + perp * gap
            q1_ext = p1 + perp * (gap + 4)
            q2_ext = p2 + perp * (gap + 4)

            cv2.line(bg, tuple(p1.astype(int)), tuple(q1_ext.astype(int)),
                     dim_color, 1, cv2.LINE_AA)
            cv2.line(bg, tuple(p2.astype(int)), tuple(q2_ext.astype(int)),
                     dim_color, 1, cv2.LINE_AA)
            cv2.line(bg, tuple(q1.astype(int)), tuple(q2.astype(int)),
                     dim_color, 1, cv2.LINE_AA)

            arr_len = 5.0
            for qp, direction in [(q1, edge_unit), (q2, -edge_unit)]:
                tip = qp.astype(float)
                wing1 = tip + direction * arr_len + perp * 2.5
                wing2 = tip + direction * arr_len - perp * 2.5
                arrow_pts = np.array([tip, wing1, wing2], dtype=np.int32)
                cv2.fillConvexPoly(bg, arrow_pts, dim_color, cv2.LINE_AA)

            label = f"{dim_mm:.0f}mm"
            (tw, th), _ = cv2.getTextSize(label, dim_font, dim_scale, dim_thick)
            mid_q = ((q1 + q2) / 2.0).astype(int)
            tx = mid_q[0] - tw // 2
            ty = mid_q[1] + th // 2
            cv2.rectangle(bg, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                          (240, 240, 240), cv2.FILLED)
            cv2.putText(bg, label, (tx, ty), dim_font, dim_scale,
                        dim_color, dim_thick, cv2.LINE_AA)

    # Draw RGB axes sticking out from visible positive faces only
    # (skip if the +face is facing away from camera)
    axes = [
        (0, [hx, 0, 0], [hx * 1.8, 0, 0], (0, 0, 255), "X"),
        (1, [0, hy, 0], [0, hy * 1.8, 0], (0, 255, 0), "Y"),
        (2, [0, 0, hz], [0, 0, hz * 1.8], (255, 0, 0), "Z"),
    ]
    for ax_idx, start, end, color, label in axes:
        normal = np.zeros(3)
        normal[ax_idx] = 1.0
        if np.dot(normal, view_dir) <= 0:
            continue  # positive face not visible from this angle
        pts_3d = np.array([start, end], dtype=np.float64)
        proj_ax, _ = cv2.projectPoints(pts_3d, rvec, tvec, cam_matrix, dist_coeffs)
        p = proj_ax.reshape(-1, 2).astype(int)
        cv2.line(bg, tuple(p[0]), tuple(p[1]), color, 3, cv2.LINE_AA)
        cv2.putText(bg, label, (p[1][0] + 5, p[1][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return bg


def render_cube_thumbnail(config: CubeConfig,
                          face_grids: dict[str, np.ndarray],
                          out_path: str,
                          pixels_per_cell: int = 8):
    """Render a thumbnail strip showing the cube from 3 angles."""
    face_textures = {}
    for name, grid in face_grids.items():
        tex = render_face_texture(grid, pixels_per_cell)
        face_textures[name] = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)

    top_views = [(25, 35), (25, 155), (25, 275)]
    bot_views = [(-25, 35), (-25, 155), (-25, 275)]
    top_row = np.hstack([
        _render_cube_view(face_textures, config, e, a, show_dims=(i == 0))
        for i, (e, a) in enumerate(top_views)
    ])
    bot_row = np.hstack([
        _render_cube_view(face_textures, config, e, a, show_dims=False)
        for e, a in bot_views
    ])
    views = np.vstack([top_row, bot_row])

    # Build text info panel at the bottom
    bx, by, bz = config.box_dims
    cs = config.cell_size
    info_lines = [
        f"Box: {bx:.4g} x {by:.4g} x {bz:.4g} mm"
        f"    Grid: {config.grid_x}x{config.grid_y}x{config.grid_z}"
        f"    Dict: {config.dict_name}",
        f"Tag: {config.tag_size_mm:.4g} mm ({config.marker_pixels}x"
        f"{config.marker_pixels} cells, cell={cs:.4g} mm)"
        f"    Margin: {config.margin_cells} cell ({config.margin_cells * cs:.4g} mm)"
        f"    Border: {config.border_cells} cell ({config.border_cells * cs:.4g} mm)",
        f"IDs: {config.tag_ids[0]}-{config.tag_ids[-1]}"
        f" ({len(config.tag_ids)} tags)",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    line_h = 20
    pad = 8
    panel_h = pad + line_h * len(info_lines) + pad
    panel = np.full((panel_h, views.shape[1], 3), 240, dtype=np.uint8)
    for i, line in enumerate(info_lines):
        y = pad + line_h * (i + 1) - 4
        cv2.putText(panel, line, (pad, y), font, scale, (60, 60, 60),
                    thick, cv2.LINE_AA)

    thumbnail = np.vstack([views, panel])
    cv2.imwrite(out_path, thumbnail)
    print(f"Wrote {out_path} ({thumbnail.shape[1]}x{thumbnail.shape[0]})")


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------
class CubeMeshBuilder:
    def __init__(self):
        self.vertices: list[tuple[float, float, float]] = []
        self.triangles: list[tuple[int, int, int, bool]] = []
        self._vmap: dict[tuple[int, int, int], int] = {}

    def _add_vertex(self, x: float, y: float, z: float) -> int:
        key = (round(x * 10000), round(y * 10000), round(z * 10000))
        if key in self._vmap:
            return self._vmap[key]
        idx = len(self.vertices)
        self.vertices.append((x, y, z))
        self._vmap[key] = idx
        return idx

    def add_face(self, face_def: tuple, grid: np.ndarray, box_dims: tuple, cell_size: float):
        """Add one face of the cube to the mesh."""
        _name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
        right_half = box_dims[right_ax] / 2.0
        down_half = box_dims[down_ax] / 2.0
        face_pos = normal_sign * box_dims[normal_ax] / 2.0

        down_cells, right_cells = grid.shape

        for row in range(down_cells):
            for col in range(right_cells):
                is_painted = not bool(grid[row, col])  # paint white cells; base = black

                u0 = right_sign * (-right_half + col * cell_size)
                u1 = right_sign * (-right_half + (col + 1) * cell_size)
                v0 = down_sign * (-down_half + row * cell_size)
                v1 = down_sign * (-down_half + (row + 1) * cell_size)

                def _xyz(u: float, v: float) -> tuple[float, float, float]:
                    c = [0.0, 0.0, 0.0]
                    c[normal_ax] = face_pos
                    c[right_ax] = u
                    c[down_ax] = v
                    return (c[0], c[1], c[2])

                p00 = self._add_vertex(*_xyz(u0, v0))
                p10 = self._add_vertex(*_xyz(u1, v0))
                p01 = self._add_vertex(*_xyz(u0, v1))
                p11 = self._add_vertex(*_xyz(u1, v1))

                self.triangles.append((p00, p10, p11, is_painted))
                self.triangles.append((p00, p11, p01, is_painted))


# ---------------------------------------------------------------------------
# 3MF writer
# ---------------------------------------------------------------------------
def _fmt(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.6g}"


class ThreeMFWriter:
    def __init__(self, config: CubeConfig):
        self.config = config

    def write(self, vertices: list, triangles: list, path: str):
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", self._content_types())
            zf.writestr("_rels/.rels", self._rels())
            zf.writestr("3D/_rels/3dmodel.model.rels", self._model_rels())
            zf.writestr("3D/3dmodel.model", self._assembly())
            zf.writestr("3D/Objects/object_1.model", self._object_model(vertices, triangles))
            zf.writestr("Metadata/model_settings.config", self._model_settings())
            zf.writestr("Metadata/project_settings.config", self._project_settings())

        size_kb = os.path.getsize(path) / 1024
        print(f"Wrote {path} ({size_kb:.1f} KB)")
        print(f"  Vertices: {len(vertices)}, Triangles: {len(triangles)}")

    def _content_types(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
            '</Types>\n'
        )

    def _rels(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Target="/3D/3dmodel.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>\n'
        )

    def _model_rels(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            '  <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>\n'
        )

    def _assembly(self) -> str:
        bx, by, bz = self.config.box_dims
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"\n'
            '       xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"\n'
            '       unit="millimeter" xml:lang="en-US" requiredextensions="p"\n'
            '       xmlns:BambuStudio="http://schemas.bambulab.com/package/2021">\n'
            '  <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
            f'  <metadata name="Application">{BAMBU_STUDIO_APPLICATION}</metadata>\n'
            '  <metadata name="Title">AprilCube generated target</metadata>\n'
            '  <resources>\n'
            f'    <object id="2" p:UUID="{uuid4()}" type="model">\n'
            '      <components>\n'
            f'        <component p:path="/3D/Objects/object_1.model" objectid="1" '
            f'p:UUID="{uuid4()}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
            '      </components>\n'
            '    </object>\n'
            '  </resources>\n'
            f'  <build p:UUID="{uuid4()}">\n'
            f'    <item objectid="2" p:UUID="{uuid4()}" '
            f'transform="1 0 0 0 1 0 0 0 1 128 128 {_fmt(bz / 2)}" printable="1"/>\n'
            '  </build>\n'
            '</model>\n'
        )

    def _object_model(self, vertices: list, triangles: list) -> str:
        lines: list[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            '<model unit="millimeter" xml:lang="en-US"'
            ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
            ' xmlns:BambuStudio="http://schemas.bambulab.com/package/2021"'
            ' xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"'
            ' requiredextensions="p">'
        )
        lines.append('  <metadata name="BambuStudio:3mfVersion">1</metadata>')
        lines.append(f'  <metadata name="Application">{BAMBU_STUDIO_APPLICATION}</metadata>')
        lines.append('  <metadata name="Title">AprilCube generated target</metadata>')
        lines.append("  <resources>")
        lines.append(f'    <object id="1" p:UUID="{uuid4()}" type="model">')
        lines.append("      <mesh>")
        lines.append("        <vertices>")
        for x, y, z in vertices:
            lines.append(f'          <vertex x="{_fmt(x)}" y="{_fmt(y)}" z="{_fmt(z)}"/>')
        lines.append("        </vertices>")
        lines.append("        <triangles>")
        for v1, v2, v3, painted in triangles:
            if painted:
                lines.append(f'          <triangle v1="{v1}" v2="{v2}" v3="{v3}" paint_color="8"/>')
            else:
                lines.append(f'          <triangle v1="{v1}" v2="{v2}" v3="{v3}"/>')
        lines.append("        </triangles>")
        lines.append("      </mesh>")
        lines.append("    </object>")
        lines.append("  </resources>")
        lines.append("  <build/>")
        lines.append("</model>")
        return "\n".join(lines)

    def _model_settings(self) -> str:
        bx, by, bz = self.config.box_dims
        hx, hy, hz = _fmt(bx / 2), _fmt(by / 2), _fmt(bz / 2)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<config>\n"
            '  <object id="2">\n'
            '    <metadata key="name" value="aruco_cube"/>\n'
            f'    <metadata key="extruder" value="{self.config.extruder}"/>\n'
            '    <part id="1" subtype="normal_part">\n'
            '      <metadata key="name" value="aruco_cube_body"/>\n'
            '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
            '      <metadata key="source_object_id" value="0"/>\n'
            '      <metadata key="source_volume_id" value="0"/>\n'
            f'      <metadata key="source_offset_x" value="{hx}"/>\n'
            f'      <metadata key="source_offset_y" value="{hy}"/>\n'
            f'      <metadata key="source_offset_z" value="{hz}"/>\n'
            '      <mesh_stat edges_fixed="0" degenerate_facets="0" facets_removed="0"'
            ' facets_reversed="0" backwards_edges="0"/>\n'
            "    </part>\n"
            "  </object>\n"
            "  <plate>\n"
            '    <metadata key="plater_id" value="1"/>\n'
            '    <metadata key="plater_name" value=""/>\n'
            '    <metadata key="locked" value="false"/>\n'
            "    <model_instance>\n"
            '      <metadata key="object_id" value="2"/>\n'
            '      <metadata key="instance_id" value="0"/>\n'
            '      <metadata key="identify_id" value="1"/>\n'
            "    </model_instance>\n"
            "  </plate>\n"
            "  <assemble>\n"
            f'    <assemble_item object_id="2" instance_id="0"'
            f' transform="1 0 0 0 1 0 0 0 1 0 0 {hz}" offset="0 0 0"/>\n'
            "  </assemble>\n"
            "</config>\n"
        )

    def _project_settings(self) -> str:
        # Bambu Studio 2.x warns when a Bambu-tagged 3MF has no project config.
        return (
            json.dumps(
                {
                    "version": BAMBU_STUDIO_VERSION,
                    "printer_technology": "FFF",
                    "filament_colour": ["#000000", "#FFFFFF"],
                    "filament_type": ["PLA", "PLA"],
                    "nozzle_diameter": ["0.4"],
                },
                indent=4,
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_ids(s: Any, needed: int) -> list[int]:
    if s is None:
        return list(range(needed))
    if isinstance(s, dict):
        if "ids" in s:
            return parse_ids(s["ids"], needed)
        if "range" in s:
            return parse_ids(s["range"], needed)
        start = int(s.get("start", 0))
        count = int(s.get("count", needed))
        step = int(s.get("step", 1))
        return list(range(start, start + count * step, step))
    if isinstance(s, (list, tuple)):
        return [int(v) for v in s]
    if isinstance(s, int):
        return [s]
    ids: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return ids


def parse_grid(s: str | list[int] | tuple[int, ...]) -> tuple[int, int, int]:
    """Parse grid string into (grid_x, grid_y, grid_z) tag counts.

    Accepts WxHxD (3D) or RxC (2D, backward compat):
      RxC with C>=R → (C, R, R)  i.e. long axis X
      RxC with R>C  → (C, C, R)  i.e. long axis Z
    """
    if isinstance(s, (list, tuple)):
        parts = [int(v) for v in s]
    else:
        parts = [int(p) for p in str(s).lower().split("x")]
    if len(parts) == 2:
        R, C = parts
        if C >= R:
            return C, R, R
        else:
            return C, C, R
    elif len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Grid must be WxHxD or RxC, got: {s}")


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML field '{label}' must be a mapping")
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _maybe_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _load_yaml_mapping(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML generation specs require PyYAML. Install with `pip install pyyaml`."
        ) from exc

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML generation spec must be a mapping at the top level")
    return data


def load_generation_spec(path: str | os.PathLike[str]) -> GenerationSpec:
    """Load an extensible generation spec from YAML.

    Supported today:

    ``shape.type: cuboid`` with ``shape.grid``.

    Reserved for the next step:

    ``shape.type: voxel_grid`` or ``voxel_cuboids``.  Those shapes can share
    marker, material, and output settings while using a different geometry
    backend.
    """
    data = _load_yaml_mapping(path)
    raw_shape = _first_present(data.get("shape"), data.get("target"))
    shape = {"type": raw_shape} if isinstance(raw_shape, str) else _as_mapping(raw_shape, "shape")
    marker = _as_mapping(_first_present(data.get("marker"), data.get("markers")), "marker")
    size = _as_mapping(data.get("size"), "size")
    layout = _as_mapping(data.get("layout"), "layout")
    material = _as_mapping(
        _first_present(data.get("material"), data.get("print"), data.get("fabrication")),
        "material",
    )

    shape_type = str(_first_present(shape.get("type"), data.get("type"), "cuboid")).lower()
    if shape_type in {"cube", "box"}:
        shape_type = "cuboid"
    if shape_type not in {"cuboid", "voxel_grid", "voxel_cuboids"}:
        raise ValueError(
            f"Unsupported shape.type '{shape_type}'. Expected cuboid, voxel_grid, or voxel_cuboids."
        )
    return GenerationSpec(
        output=_first_present(data.get("output"), data.get("out_dir")),
        dict_name=_first_present(data.get("dictionary"), data.get("dict"), marker.get("dictionary"), marker.get("dict")),
        grid=_first_present(shape.get("grid"), data.get("grid")),
        ids=_first_present(marker.get("ids"), data.get("ids"), marker.get("id_range"), data.get("id_range")),
        tag_size_mm=_maybe_float(_first_present(
            size.get("tag_size_mm"), size.get("tag_size"), marker.get("tag_size_mm"),
            data.get("tag_size_mm"), data.get("tag_size"),
        )),
        cell_size_mm=_maybe_float(_first_present(
            size.get("cell_size_mm"), size.get("cell_size"), data.get("cell_size_mm"), data.get("cell_size"),
        )),
        margin_cells=_maybe_int(_first_present(
            layout.get("margin_cells"), layout.get("margin_cell"), data.get("margin_cells"), data.get("margin_cell"),
        )),
        border_cells=_maybe_int(_first_present(
            layout.get("border_cells"), layout.get("border_cell"), data.get("border_cells"), data.get("border_cell"),
        )),
        extruder=_maybe_int(_first_present(material.get("extruder"), data.get("extruder"))),
        invert=_maybe_bool(_first_present(marker.get("invert"), material.get("invert"), data.get("invert"))),
        shape_type=shape_type,
        shape=shape,
        source_path=str(path),
    )


def apply_cli_overrides(spec: GenerationSpec, args: argparse.Namespace) -> GenerationSpec:
    """Apply explicit CLI values on top of YAML/default spec values."""
    if args.output is not None:
        spec.output = args.output
    if args.dict is not None:
        spec.dict_name = args.dict
    if args.grid is not None:
        spec.grid = args.grid
    if args.ids is not None:
        spec.ids = args.ids
    if args.tag_size is not None:
        spec.tag_size_mm = args.tag_size
        spec.cell_size_mm = None
    if args.cell_size is not None:
        spec.cell_size_mm = args.cell_size
        spec.tag_size_mm = None
    if args.margin_cell is not None:
        spec.margin_cells = args.margin_cell
    if args.border_cell is not None:
        spec.border_cells = args.border_cell
    if args.extruder is not None:
        spec.extruder = args.extruder
    if args.invert is not None:
        spec.invert = args.invert
    return spec


def _vec3i(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a 3-element list")
    return int(value[0]), int(value[1]), int(value[2])


def _vec3f(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a 3-element list")
    return float(value[0]), float(value[1]), float(value[2])


def _normal_from_face_def(face_def: tuple) -> tuple[float, float, float]:
    normal = [0.0, 0.0, 0.0]
    normal[face_def[1]] = float(face_def[2])
    return tuple(normal)


def _parse_voxel_occupancy(shape: dict[str, Any]) -> tuple[set[tuple[int, int, int]], list[dict[str, Any]]]:
    shape_type = str(shape.get("type", "voxel_cuboids")).lower()
    occupied: set[tuple[int, int, int]] = set()
    source_cuboids: list[dict[str, Any]] = []

    if shape_type == "voxel_cuboids":
        cuboids = shape.get("cuboids")
        if not isinstance(cuboids, list) or not cuboids:
            raise ValueError("shape.cuboids must be a non-empty list for voxel_cuboids")
        for idx, cuboid in enumerate(cuboids):
            if not isinstance(cuboid, dict):
                raise ValueError(f"shape.cuboids[{idx}] must be a mapping")
            origin = _vec3i(cuboid.get("origin", [0, 0, 0]), f"shape.cuboids[{idx}].origin")
            size = _vec3i(cuboid.get("size"), f"shape.cuboids[{idx}].size")
            if any(v <= 0 for v in size):
                raise ValueError(f"shape.cuboids[{idx}].size values must be positive")
            source_cuboids.append({
                "name": cuboid.get("name", f"cuboid_{idx}"),
                "origin": list(origin),
                "size": list(size),
            })
            for x in range(origin[0], origin[0] + size[0]):
                for y in range(origin[1], origin[1] + size[1]):
                    for z in range(origin[2], origin[2] + size[2]):
                        occupied.add((x, y, z))
    elif shape_type == "voxel_grid":
        voxels = shape.get("voxels")
        if not isinstance(voxels, list) or not voxels:
            raise ValueError("shape.voxels must be a non-empty list for voxel_grid")
        for idx, voxel in enumerate(voxels):
            occupied.add(_vec3i(voxel, f"shape.voxels[{idx}]"))
    else:
        raise ValueError(f"Unsupported voxel shape type '{shape_type}'")

    if not occupied:
        raise ValueError("voxel shape contains no occupied voxels")
    return occupied, source_cuboids


def _validate_tag_ids(tag_ids: list[int], tag_gen: TagPatternGenerator, dict_name: str) -> None:
    for tid in tag_ids:
        if tid < 0 or tid >= tag_gen.max_id:
            raise ValueError(f"tag ID {tid} out of range [0, {tag_gen.max_id}) for {dict_name}")
    if len(set(tag_ids)) < len(tag_ids):
        print("Warning: duplicate tag IDs", file=sys.stderr)


def _axis_bounds_for_voxel(
    voxel: tuple[int, int, int],
    axis: int,
    voxel_size: float,
    center_abs: tuple[float, float, float],
) -> tuple[float, float]:
    lo = voxel[axis] * voxel_size - center_abs[axis]
    hi = (voxel[axis] + 1) * voxel_size - center_abs[axis]
    return lo, hi


def _axis_at(
    voxel: tuple[int, int, int],
    axis: int,
    axis_sign: int,
    step: float,
    voxel_size: float,
    cell_size: float,
    center_abs: tuple[float, float, float],
) -> float:
    lo, hi = _axis_bounds_for_voxel(voxel, axis, voxel_size, center_abs)
    if axis_sign > 0:
        return lo + step * cell_size
    return hi - step * cell_size


def _voxel_face_point(
    voxel: tuple[int, int, int],
    face_def: tuple,
    row: float,
    col: float,
    voxel_size: float,
    cell_size: float,
    center_abs: tuple[float, float, float],
) -> np.ndarray:
    _name, normal_ax, normal_sign, right_ax, right_sign, down_ax, down_sign = face_def
    pt = np.zeros(3, dtype=np.float64)
    if normal_sign > 0:
        pt[normal_ax] = (voxel[normal_ax] + 1) * voxel_size - center_abs[normal_ax]
    else:
        pt[normal_ax] = voxel[normal_ax] * voxel_size - center_abs[normal_ax]
    pt[right_ax] = _axis_at(voxel, right_ax, right_sign, col, voxel_size, cell_size, center_abs)
    pt[down_ax] = _axis_at(voxel, down_ax, down_sign, row, voxel_size, cell_size, center_abs)
    return pt


def _add_voxel_face(
    builder: CubeMeshBuilder,
    face_def: tuple,
    voxel: tuple[int, int, int],
    grid: np.ndarray,
    voxel_size: float,
    cell_size: float,
    center_abs: tuple[float, float, float],
) -> None:
    down_cells, right_cells = grid.shape

    for row in range(down_cells):
        for col in range(right_cells):
            is_painted = not bool(grid[row, col])

            p00_xyz = _voxel_face_point(voxel, face_def, row, col, voxel_size, cell_size, center_abs)
            p10_xyz = _voxel_face_point(voxel, face_def, row, col + 1, voxel_size, cell_size, center_abs)
            p01_xyz = _voxel_face_point(voxel, face_def, row + 1, col, voxel_size, cell_size, center_abs)
            p11_xyz = _voxel_face_point(voxel, face_def, row + 1, col + 1, voxel_size, cell_size, center_abs)

            p00 = builder._add_vertex(*p00_xyz)
            p10 = builder._add_vertex(*p10_xyz)
            p01 = builder._add_vertex(*p01_xyz)
            p11 = builder._add_vertex(*p11_xyz)

            builder.triangles.append((p00, p10, p11, is_painted))
            builder.triangles.append((p00, p11, p01, is_painted))


def _voxel_face_corners(
    voxel: tuple[int, int, int],
    face_def: tuple,
    face_cells: int,
    voxel_size: float,
    cell_size: float,
    center_abs: tuple[float, float, float],
) -> np.ndarray:
    return np.array([
        _voxel_face_point(voxel, face_def, 0, 0, voxel_size, cell_size, center_abs),
        _voxel_face_point(voxel, face_def, 0, face_cells, voxel_size, cell_size, center_abs),
        _voxel_face_point(voxel, face_def, face_cells, face_cells, voxel_size, cell_size, center_abs),
        _voxel_face_point(voxel, face_def, face_cells, 0, voxel_size, cell_size, center_abs),
    ], dtype=np.float64)


def _marker_corners_on_voxel_face(
    voxel: tuple[int, int, int],
    face_def: tuple,
    face_cells: int,
    marker_pixels: int,
    voxel_size: float,
    cell_size: float,
    center_abs: tuple[float, float, float],
) -> np.ndarray:
    row_off = (face_cells - marker_pixels) // 2
    col_off = (face_cells - marker_pixels) // 2
    return np.array([
        _voxel_face_point(voxel, face_def, row_off, col_off + marker_pixels, voxel_size, cell_size, center_abs),
        _voxel_face_point(voxel, face_def, row_off, col_off, voxel_size, cell_size, center_abs),
        _voxel_face_point(voxel, face_def, row_off + marker_pixels, col_off, voxel_size, cell_size, center_abs),
        _voxel_face_point(voxel, face_def, row_off + marker_pixels, col_off + marker_pixels, voxel_size, cell_size, center_abs),
    ], dtype=np.float64)


def _render_voxel_view(
    marker_faces: list[dict[str, Any]],
    box_dims: tuple[float, float, float],
    elev_deg: float,
    azim_deg: float,
    view_w: int = 420,
    view_h: int = 420,
    show_ids: bool = True,
    background_color: int = 240,
) -> np.ndarray:
    bg = np.full((view_h, view_w, 3), background_color, dtype=np.uint8)
    diag = np.sqrt(sum(d ** 2 for d in box_dims))
    fx = fy = view_w * 1.8
    cam_matrix = np.array([[fx, 0, view_w / 2],
                           [0, fy, view_h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(5)
    rvec, tvec, fwd = _camera_from_angles(elev_deg, azim_deg, diag * 2.8)
    view_dir = -fwd
    R_cam, _ = cv2.Rodrigues(rvec)

    visible = []
    for face in marker_faces:
        normal = np.array(face["normal"], dtype=np.float64)
        if np.dot(normal, view_dir) <= 0:
            continue
        corners = np.array(face["face_corners_mm"], dtype=np.float64)
        center_cam = (R_cam @ corners.mean(axis=0) + tvec.flatten())[2]
        visible.append((center_cam, face))
    visible.sort(reverse=True)

    for _, face in visible:
        corners_3d = np.array(face["face_corners_mm"], dtype=np.float64)
        projected, _ = cv2.projectPoints(corners_3d, rvec, tvec, cam_matrix, dist_coeffs)
        pts_2d = projected.reshape(-1, 2).astype(np.float32)
        tex = cv2.cvtColor(render_face_texture(face["grid"], pixels_per_cell=10), cv2.COLOR_GRAY2BGR)
        th, tw = tex.shape[:2]
        src_pts = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, pts_2d)
        warped = cv2.warpPerspective(tex, M, (view_w, view_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((th, tw), 255, dtype=np.uint8), M, (view_w, view_h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bg = np.where(mask[:, :, np.newaxis] > 0, warped, bg)

        if show_ids:
            center_3d = np.array(face["corners_mm"], dtype=np.float64).mean(axis=0)
            label_3d = center_3d + np.array(face["normal"], dtype=np.float64) * (max(box_dims) * 0.08)
            label_pts, _ = cv2.projectPoints(np.array([center_3d, label_3d]), rvec, tvec, cam_matrix, dist_coeffs)
            p = label_pts.reshape(-1, 2).astype(int)
            cv2.line(bg, tuple(p[0]), tuple(p[1]), (80, 80, 80), 1, cv2.LINE_AA)
            cv2.circle(bg, tuple(p[0]), 2, (80, 80, 80), cv2.FILLED)
            label = str(face["id"])
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.5
            thick = 1
            (tw_text, th_text), _ = cv2.getTextSize(label, font, scale, thick)
            lx, ly = p[1]
            cv2.rectangle(bg, (lx - tw_text // 2 - 3, ly - th_text // 2 - 3),
                          (lx + tw_text // 2 + 3, ly + th_text // 2 + 3), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(bg, (lx - tw_text // 2 - 3, ly - th_text // 2 - 3),
                          (lx + tw_text // 2 + 3, ly + th_text // 2 + 3), (80, 80, 80), 1)
            cv2.putText(bg, label, (lx - tw_text // 2, ly + th_text // 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    return bg


def render_voxel_thumbnail(
    marker_faces: list[dict[str, Any]],
    box_dims: tuple[float, float, float],
    out_path: str,
    info_lines: list[str],
) -> None:
    top_views = [(25, 35), (25, 155), (25, 275)]
    bot_views = [(-25, 35), (-25, 155), (-25, 275)]
    top_row = np.hstack([
        _render_voxel_view(marker_faces, box_dims, e, a)
        for e, a in top_views
    ])
    bot_row = np.hstack([
        _render_voxel_view(marker_faces, box_dims, e, a)
        for e, a in bot_views
    ])
    views = np.vstack([top_row, bot_row])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thick = 1
    line_h = 20
    pad = 8
    panel_h = pad + line_h * len(info_lines) + pad
    panel = np.full((panel_h, views.shape[1], 3), 240, dtype=np.uint8)
    for i, line in enumerate(info_lines):
        y = pad + line_h * (i + 1) - 4
        cv2.putText(panel, line, (pad, y), font, scale, (60, 60, 60), thick, cv2.LINE_AA)
    thumbnail = np.vstack([views, panel])
    cv2.imwrite(out_path, thumbnail)
    print(f"Wrote {out_path} ({thumbnail.shape[1]}x{thumbnail.shape[0]})")


def write_voxel_readme(config_data: dict, out_dir: str) -> None:
    target = config_data["target"]
    face_counts: dict[str, int] = {}
    for marker in config_data["markers"]:
        face_counts[marker["face"]] = face_counts.get(marker["face"], 0) + 1
    face_lines = [
        f"| {name} | {count} |"
        for name, count in sorted(face_counts.items())
    ]
    md = f"""# AprilCube Voxel Target

![Target preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Shape type | `{target['type']}` |
| Voxel size | {target['voxel_size_mm']:.4g} mm |
| Occupied voxels | {target['occupied_voxels']} |
| Box dimensions | {' x '.join(f'{v:.4g}' for v in config_data['box_dims'])} mm |
| Dictionary | `{config_data['dict']}` |
| Tag size | {config_data['tag_size_mm']:.4g} mm |
| Cell size | {config_data['cell_size_mm']:.4g} mm |
| Marker count | {len(config_data['markers'])} |

## Exposed Face Counts

| Face direction | Markers |
|----------------|---------|
{chr(10).join(face_lines)}

## Files

| File | Description |
|------|-------------|
| `cube.3mf` | Multi-color 3MF target for Bambu Studio |
| `config.json` | Detector config with explicit marker corner coordinates |
| `thumbnail.png` | Preview rendering of exposed voxel-face markers |
| `mujoco/cube.xml` | MuJoCo MJCF model |
| `mujoco/cube.obj` | Wavefront OBJ mesh with UV-mapped marker faces |
| `mujoco/cube.mtl` | OBJ material file |
| `mujoco/cube_atlas.png` | Texture atlas |

## Config JSON

```json
{json.dumps(config_data, indent=2)}
```
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(md)
    print(f"Wrote {readme_path}")


def write_readme(config: CubeConfig, config_data: dict,
                 face_tag_map: dict, out_dir: str):
    """Write a README.md summarizing the generated cube."""
    bx, by, bz = config.box_dims
    cs = config.cell_size
    grid_str = f"{config.grid_x}x{config.grid_y}x{config.grid_z}"
    margin_mm = config.margin_cells * cs
    border_mm = config.border_cells * cs

    face_lines = []
    for name, ids in face_tag_map.items():
        face_lines.append(f"| {name} | {', '.join(str(i) for i in ids)} |")

    md = f"""# ArUco Cube — {grid_str}

![Cube preview](thumbnail.png)

## Parameters

| Parameter | Value |
|-----------|-------|
| Dictionary | `{config.dict_name}` |
| Grid | {grid_str} (X x Y x Z tags) |
| Box dimensions | {bx:.4g} x {by:.4g} x {bz:.4g} mm |
| Tag size | {config.tag_size_mm:.4g} mm ({config.marker_pixels}x{config.marker_pixels} cells) |
| Cell size | {cs:.4g} mm |
| Margin | {config.margin_cells} cell ({margin_mm:.4g} mm) |
| Border | {config.border_cells} cell ({border_mm:.4g} mm) |
| Total tags | {len(config.tag_ids)} |
| Tag IDs | {config.tag_ids[0]}–{config.tag_ids[-1]} |

## Face Layout

| Face | Tag IDs |
|------|---------|
{chr(10).join(face_lines)}

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
{json.dumps(config_data, indent=2)}
```

## Regenerate

```bash
aprilcube generate --grid {grid_str} --dict {config.dict_name} --tag-size {config.tag_size_mm:.4g} --margin-cell {config.margin_cells} --border-cell {config.border_cells} -o {os.path.basename(out_dir)}
```
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(md)
    print(f"Wrote {readme_path}")


def generate_voxel_target(
    spec: GenerationSpec,
    dict_name: str,
    dict_id: int,
    out_dir: str,
    margin_cells: int,
    border_cells: int,
    extruder: int,
    invert: bool,
) -> None:
    shape = spec.shape or {}
    voxel_size_value = _first_present(shape.get("voxel_size_mm"), shape.get("voxel_size"))
    if voxel_size_value is None:
        raise ValueError("voxel targets require shape.voxel_size_mm")
    voxel_size = float(voxel_size_value)
    if voxel_size <= 0:
        raise ValueError("shape.voxel_size_mm must be positive")

    occupied, source_cuboids = _parse_voxel_occupancy(shape)
    tag_gen = TagPatternGenerator(dict_id)
    marker_pixels = tag_gen.marker_pixels

    if spec.tag_size_mm is not None and spec.cell_size_mm is not None:
        raise ValueError("Specify only one of tag_size_mm/tag_size or cell_size_mm/cell_size")
    if spec.cell_size_mm is not None:
        cell_size = float(spec.cell_size_mm)
        tag_size = cell_size * marker_pixels
    elif spec.tag_size_mm is not None:
        tag_size = float(spec.tag_size_mm)
        cell_size = tag_size / marker_pixels
    else:
        face_cells_default = marker_pixels + 2 * border_cells
        cell_size = voxel_size / face_cells_default
        tag_size = cell_size * marker_pixels

    if cell_size <= 0:
        raise ValueError("cell size must be positive")

    face_cells_float = voxel_size / cell_size
    face_cells = int(round(face_cells_float))
    if abs(face_cells - face_cells_float) > 1e-6:
        raise ValueError(
            "voxel_size_mm must be an integer multiple of cell_size_mm. "
            f"Got voxel_size={voxel_size:g}, cell_size={cell_size:g}."
        )
    if face_cells < marker_pixels + 2 * border_cells:
        raise ValueError(
            f"voxel face has {face_cells} cells, but one marker needs at least "
            f"{marker_pixels + 2 * border_cells} cells with the configured border"
        )

    exposed: list[tuple[tuple[int, int, int], tuple]] = []
    for voxel in sorted(occupied):
        for face_def in FACE_DEFS:
            _name, normal_ax, normal_sign, *_rest = face_def
            neighbor = list(voxel)
            neighbor[normal_ax] += normal_sign
            if tuple(neighbor) not in occupied:
                exposed.append((voxel, face_def))

    needed = len(exposed)
    tag_ids = parse_ids(spec.ids, needed)
    if len(tag_ids) < needed:
        raise ValueError(f"need {needed} tag IDs, got {len(tag_ids)}")
    tag_ids = tag_ids[:needed]
    _validate_tag_ids(tag_ids, tag_gen, dict_name)

    xs = [v[0] for v in occupied]
    ys = [v[1] for v in occupied]
    zs = [v[2] for v in occupied]
    min_idx = (min(xs), min(ys), min(zs))
    max_idx = (max(xs), max(ys), max(zs))
    extent = (
        max_idx[0] - min_idx[0] + 1,
        max_idx[1] - min_idx[1] + 1,
        max_idx[2] - min_idx[2] + 1,
    )
    box_dims = tuple(float(e * voxel_size) for e in extent)
    center_abs = tuple((min_idx[i] + max_idx[i] + 1) * voxel_size / 2.0 for i in range(3))

    print(f"Shape: {spec.shape_type}")
    print(f"Dictionary: {dict_name} ({marker_pixels}×{marker_pixels} cells/tag)")
    print(f"Voxels: {len(occupied)} occupied, {needed} exposed marker faces")
    print(f"Voxel: {voxel_size:.4g} mm, Face grid: {face_cells}×{face_cells} cells")
    print(f"Cell: {cell_size:.4g} mm, Tag: {tag_size:.4g} mm")
    print(f"Margin: {margin_cells} cells ({margin_cells * cell_size:.4g} mm), Border: {border_cells} cells ({border_cells * cell_size:.4g} mm)")
    print(f"Box: {box_dims[0]:.4g} × {box_dims[1]:.4g} × {box_dims[2]:.4g} mm")
    print(f"Total tags: {needed}, IDs: {tag_ids[0]}–{tag_ids[-1]}")

    builder = CubeMeshBuilder()
    marker_records: list[dict[str, Any]] = []
    patterns = [tag_gen.generate(tid) for tid in tag_ids]

    for idx, ((voxel, face_def), tag_id, pattern) in enumerate(zip(exposed, tag_ids, patterns)):
        grid = build_face_grid(
            [pattern], 1, 1, face_cells, face_cells,
            marker_pixels, margin_cells, invert,
        )
        _add_voxel_face(builder, face_def, voxel, grid, voxel_size, cell_size, center_abs)

        marker_corners = _marker_corners_on_voxel_face(
            voxel, face_def, face_cells, marker_pixels,
            voxel_size, cell_size, center_abs,
        )
        face_corners = _voxel_face_corners(
            voxel, face_def, face_cells,
            voxel_size, cell_size, center_abs,
        )
        normal = _normal_from_face_def(face_def)
        marker_records.append({
            "id": int(tag_id),
            "face": face_def[0],
            "voxel": list(voxel),
            "normal": list(normal),
            "corners_mm": marker_corners.tolist(),
            "face_corners_mm": face_corners.tolist(),
            "grid": grid,
        })

    edges: dict[tuple[int, int], int] = {}
    for v1, v2, v3, _ in builder.triangles:
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            edge = (min(a, b), max(a, b))
            edges[edge] = edges.get(edge, 0) + 1
    non_manifold = sum(1 for c in edges.values() if c != 2)
    if non_manifold:
        print(f"Warning: {non_manifold} non-manifold edges detected", file=sys.stderr)

    os.makedirs(out_dir, exist_ok=True)
    dummy_config = CubeConfig(
        grid_x=extent[0], grid_y=extent[1], grid_z=extent[2],
        dict_id=dict_id, dict_name=dict_name,
        tag_ids=tag_ids,
        tag_size_mm=tag_size,
        margin_cells=margin_cells, border_cells=border_cells,
        cell_size_mm=cell_size,
        extruder=extruder, invert=invert,
        target_type=spec.shape_type,
    )
    dummy_config.compute()
    dummy_config.box_dims = box_dims
    dummy_config.x_cells = face_cells * extent[0]
    dummy_config.y_cells = face_cells * extent[1]
    dummy_config.z_cells = face_cells * extent[2]

    threemf_path = os.path.join(out_dir, "cube.3mf")
    writer = ThreeMFWriter(dummy_config)
    writer.write(builder.vertices, builder.triangles, threemf_path)

    config_markers = [
        {k: v for k, v in record.items() if k != "grid"}
        for record in marker_records
    ]
    config_data = {
        "schema_version": 2,
        "target": {
            "type": spec.shape_type,
            "voxel_size_mm": voxel_size,
            "occupied_voxels": len(occupied),
            "extent": list(extent),
            "origin_index": list(min_idx),
            "cuboids": source_cuboids,
        },
        "dict": dict_name,
        "grid": f"{extent[0]}x{extent[1]}x{extent[2]}",
        "tag_ids": tag_ids,
        "markers": config_markers,
        "tag_size_mm": tag_size,
        "cell_size_mm": cell_size,
        "margin_cells": margin_cells,
        "border_cells": border_cells,
        "marker_pixels": marker_pixels,
        "box_dims": list(box_dims),
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Wrote {config_path}")

    write_voxel_mujoco_assets(
        dummy_config, marker_records, source_cuboids, occupied,
        voxel_size, center_abs, out_dir,
    )

    thumb_path = os.path.join(out_dir, "thumbnail.png")
    info_lines = [
        f"Voxel target: {len(occupied)} voxels, {needed} marker faces, dict={dict_name}",
        f"Box: {box_dims[0]:.4g} x {box_dims[1]:.4g} x {box_dims[2]:.4g} mm    Voxel: {voxel_size:.4g} mm",
        f"Tag: {tag_size:.4g} mm ({marker_pixels}x{marker_pixels} cells, cell={cell_size:.4g} mm)    IDs: {tag_ids[0]}-{tag_ids[-1]}",
    ]
    render_voxel_thumbnail(marker_records, box_dims, thumb_path, info_lines)
    write_voxel_readme(config_data, out_dir)
    print("Done!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate a 3MF cuboid with ArUco/AprilTag markers for multi-color 3D printing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  %(prog)s --grid 1x1x1 --dict 4x4_50 --ids 0-5 --tag-size 30\n"
        "  %(prog)s --grid 2x2x2 --dict 4x4_50 --ids 0-23 --tag-size 30\n"
        "  %(prog)s --grid 5x4x1 --dict 4x4_50 --tag-size 20          # flat box\n"
        "  %(prog)s --grid 2x2 --dict 4x4_50 --ids 0-23 --tag-size 30 # 2D shorthand\n"
        "  %(prog)s target.yaml                                      # YAML spec\n",
    )
    parser.add_argument("spec", nargs="?", help="YAML generation spec")
    parser.add_argument("-c", "--config", help="YAML generation spec")
    parser.add_argument("-o", "--output", default=None,
                        help="Output directory (will contain cube.3mf + config.json)")
    parser.add_argument("-d", "--dict", default=None, choices=sorted(DICT_MAP.keys()))
    parser.add_argument("-g", "--grid", default=None,
                        help="Tags per dimension: WxHxD or RxC shorthand (default: 1x1x1)")
    parser.add_argument("-t", "--ids", default=None, help="Tag IDs: range (0-23) or comma-separated")
    size_grp = parser.add_mutually_exclusive_group()
    size_grp.add_argument("--tag-size", type=float, default=None, help="Tag size in mm (default: 30)")
    size_grp.add_argument("--cell-size", type=float, default=None, help="Cell size in mm (tag = cell × marker_pixels)")
    parser.add_argument("--margin-cell", type=int, default=None, help="Margin between tags in cells (default: 1)")
    parser.add_argument("--border-cell", type=int, default=None, help="Outer border in cells (default: 1)")
    parser.add_argument("--extruder", type=int, default=None, help="Bambu Studio extruder (default: 1)")
    parser.add_argument("--invert", action="store_true", default=None, help="Invert colors")

    args = parser.parse_args()

    if args.spec and args.config:
        print("Error: pass either a positional YAML spec or --config, not both", file=sys.stderr)
        sys.exit(1)

    yaml_path = args.config or args.spec
    try:
        spec = load_generation_spec(yaml_path) if yaml_path else GenerationSpec()
        spec = apply_cli_overrides(spec, args)
        shape_type = spec.shape_type

        dict_name = spec.dict_name or "4x4_50"
        if dict_name not in DICT_MAP:
            raise ValueError(f"Unknown dictionary '{dict_name}'")

        output = spec.output or "aruco_cube"
        margin_cells = spec.margin_cells if spec.margin_cells is not None else 1
        border_cells = spec.border_cells if spec.border_cells is not None else 1
        extruder = spec.extruder if spec.extruder is not None else 1
        invert = bool(spec.invert) if spec.invert is not None else False

        if shape_type == "cuboid":
            grid_value = spec.grid or "1x1x1"
            grid_x, grid_y, grid_z = parse_grid(grid_value)
            if spec.tag_size_mm is not None and spec.cell_size_mm is not None:
                raise ValueError("Specify only one of tag_size_mm/tag_size or cell_size_mm/cell_size")
            tag_size = spec.tag_size_mm if spec.tag_size_mm is not None else (
                0.0 if spec.cell_size_mm is not None else 30.0
            )
            cell_size = spec.cell_size_mm if spec.cell_size_mm is not None else 0.0
    except (OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    dict_id = DICT_MAP[dict_name]

    if spec.shape_type in {"voxel_cuboids", "voxel_grid"}:
        try:
            if spec.source_path:
                print(f"Spec: {spec.source_path}")
            generate_voxel_target(
                spec, dict_name, dict_id, output,
                margin_cells, border_cells, extruder, invert,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return
    if spec.shape_type != "cuboid":
        print(f"Error: unsupported shape.type '{spec.shape_type}'", file=sys.stderr)
        sys.exit(1)


    # Build config (tag_ids set later after we know total_tags)
    config = CubeConfig(
        grid_x=grid_x, grid_y=grid_y, grid_z=grid_z,
        dict_id=dict_id, dict_name=dict_name,
        tag_ids=[],
        tag_size_mm=tag_size,
        margin_cells=margin_cells, border_cells=border_cells,
        cell_size_mm=cell_size,
        extruder=extruder, invert=invert,
    )
    config.compute()

    needed = config.total_tags()
    tag_ids = parse_ids(spec.ids, needed)
    if len(tag_ids) < needed:
        print(f"Error: need {needed} tag IDs, got {len(tag_ids)}", file=sys.stderr)
        sys.exit(1)
    tag_ids = tag_ids[:needed]
    config.tag_ids = tag_ids

    # Validate
    tag_gen = TagPatternGenerator(dict_id)
    for tid in tag_ids:
        if tid < 0 or tid >= tag_gen.max_id:
            print(f"Error: tag ID {tid} out of range [0, {tag_gen.max_id}) for {dict_name}", file=sys.stderr)
            sys.exit(1)
    if len(set(tag_ids)) < len(tag_ids):
        print("Warning: duplicate tag IDs", file=sys.stderr)

    bx, by, bz = config.box_dims
    margin_mm = config.margin_cells * config.cell_size
    border_mm = config.border_cells * config.cell_size
    if spec.source_path:
        print(f"Spec: {spec.source_path}")
    print(f"Shape: {spec.shape_type}")
    print(f"Dictionary: {dict_name} ({tag_gen.marker_pixels}×{tag_gen.marker_pixels} cells/tag)")
    print(f"Grid: {grid_x}×{grid_y}×{grid_z} (X×Y×Z tags)")
    print(f"Cell: {config.cell_size:.4g} mm, Tag: {config.tag_size_mm:.4g} mm")
    print(f"Margin: {config.margin_cells} cells ({margin_mm:.4g} mm), Border: {config.border_cells} cells ({border_mm:.4g} mm)")
    print(f"Box: {bx:.4g} × {by:.4g} × {bz:.4g} mm")

    # Show per-face layout
    for fd in FACE_DEFS:
        fr, fc, dc, rc = config.face_layout(fd)
        print(f"  {fd[0]:3s}: {fr}×{fc} tags ({rc}×{dc} cells)")

    print(f"Total tags: {needed}, IDs: {tag_ids[0]}–{tag_ids[-1]}")

    # Generate tag patterns
    patterns = [tag_gen.generate(tid) for tid in tag_ids]

    # Build mesh — assign tag IDs sequentially across faces
    builder = CubeMeshBuilder()
    face_tag_map = {}  # face_name -> list of tag IDs
    face_grids = {}    # face_name -> boolean grid (for MuJoCo textures)
    id_cursor = 0
    for face_def in FACE_DEFS:
        fr, fc, dc, rc = config.face_layout(face_def)
        n = fr * fc
        face_tag_map[face_def[0]] = tag_ids[id_cursor:id_cursor + n]
        face_patterns = patterns[id_cursor:id_cursor + n]
        id_cursor += n
        grid = build_face_grid(
            face_patterns, fr, fc, dc, rc,
            config.marker_pixels, config.margin_cells, config.invert,
        )
        face_grids[face_def[0]] = grid
        builder.add_face(face_def, grid, config.box_dims, config.cell_size)

    # Validate mesh
    edges: dict[tuple[int, int], int] = {}
    for v1, v2, v3, _ in builder.triangles:
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            edge = (min(a, b), max(a, b))
            edges[edge] = edges.get(edge, 0) + 1
    non_manifold = sum(1 for c in edges.values() if c != 2)
    if non_manifold:
        print(f"Warning: {non_manifold} non-manifold edges detected", file=sys.stderr)

    # Write output directory
    out_dir = output
    os.makedirs(out_dir, exist_ok=True)

    threemf_path = os.path.join(out_dir, "cube.3mf")
    writer = ThreeMFWriter(config)
    writer.write(builder.vertices, builder.triangles, threemf_path)

    # Write config.json for the detection pipeline
    grid_str = f"{grid_x}x{grid_y}x{grid_z}"
    config_data = {
        "schema_version": 1,
        "target": {
            "type": spec.shape_type,
            "grid": grid_str,
        },
        "dict": dict_name,
        "grid": grid_str,
        "tag_ids": tag_ids,
        "faces": face_tag_map,
        "tag_size_mm": config.tag_size_mm,
        "cell_size_mm": config.cell_size,
        "margin_cells": config.margin_cells,
        "border_cells": config.border_cells,
        "marker_pixels": config.marker_pixels,
        "box_dims": list(config.box_dims),
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Wrote {config_path}")

    # Write MuJoCo assets (OBJ + atlas texture + MJCF XML)
    write_mujoco_assets(config, face_grids, out_dir)

    # Render thumbnail
    thumb_path = os.path.join(out_dir, "thumbnail.png")
    render_cube_thumbnail(config, face_grids, thumb_path)

    # Write per-directory README.md
    write_readme(config, config_data, face_tag_map, out_dir)

    print("Done!")


if __name__ == "__main__":
    main()
