import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from aprilcube import generate
from aprilcube.detect import build_tag_corner_map, load_cube_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DENSE_T_SPEC = REPO_ROOT / "examples" / "t_shape_target_xlarge_dense.yaml"


def _mesh_vertices_from_3mf(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        model_xml = archive.read("3D/Objects/object_1.model")
    root = ET.fromstring(model_xml)
    namespace = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    vertices = root.findall(".//m:vertex", namespace)
    return np.asarray([
        [float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
        for vertex in vertices
    ])


def test_yaml_preserves_half_cell_border():
    spec = generate.load_generation_spec(DENSE_T_SPEC)

    assert spec.border_cells == 0.5


@pytest.mark.parametrize("border", [-0.5, 0.25, float("inf")])
def test_border_must_be_nonnegative_half_cell_increment(border):
    config = generate.CubeConfig(
        grid_x=1,
        grid_y=1,
        grid_z=1,
        dict_id=generate.DICT_MAP["4x4_50"],
        dict_name="4x4_50",
        tag_ids=list(range(6)),
        tag_size_mm=22.5,
        border_cells=border,
    )

    with pytest.raises(ValueError, match="border_cells"):
        config.compute()


def test_half_cell_face_raster_is_centered_and_integer_border_is_unchanged():
    pattern = np.ones((6, 6), dtype=bool)

    half_grid = generate.build_face_grid(
        [pattern], 1, 1, 7, 7, 6, 1, False, layout_scale=2,
    )
    assert half_grid.shape == (14, 14)
    assert half_grid[1:13, 1:13].all()
    assert not half_grid[0, :].any()
    assert not half_grid[-1, :].any()
    assert not half_grid[:, 0].any()
    assert not half_grid[:, -1].any()

    integer_grid = generate.build_face_grid(
        [pattern], 1, 1, 8, 8, 6, 1, False,
    )
    assert integer_grid.shape == (8, 8)
    assert integer_grid[1:7, 1:7].all()


def test_cli_half_cell_border_regular_cuboid_matches_detector(tmp_path, monkeypatch):
    out_dir = tmp_path / "half_border_cube"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aprilcube generate",
            "--grid", "1x1x1",
            "--dict", "4x4_50",
            "--ids", "0-5",
            "--tag-size", "22.5",
            "--border-cell", "0.5",
            "-o", str(out_dir),
        ],
    )

    generate.main()

    data = json.loads((out_dir / "config.json").read_text())
    assert data["border_cells"] == 0.5
    assert data["box_dims"] == pytest.approx([26.25, 26.25, 26.25])

    config, _face_ids = load_cube_config(str(out_dir / "config.json"))
    corners_by_id = build_tag_corner_map(config)
    for corners in corners_by_id.values():
        assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(22.5)
        assert np.linalg.norm(corners[2] - corners[1]) == pytest.approx(22.5)
        assert sorted(np.abs(corners.mean(axis=0))) == pytest.approx([0.0, 0.0, 13.125])


def test_dense_t_half_cell_border_generation(tmp_path, monkeypatch):
    out_dir = tmp_path / "dense_t"
    monkeypatch.setattr(
        sys,
        "argv",
        ["aprilcube generate", str(DENSE_T_SPEC), "-o", str(out_dir)],
    )

    generate.main()

    data = json.loads((out_dir / "config.json").read_text())
    assert data["target"]["voxel_size_mm"] == pytest.approx(26.25)
    assert data["target"]["extent"] == [6, 2, 8]
    assert data["target"]["occupied_voxels"] == 48
    assert data["box_dims"] == pytest.approx([157.5, 52.5, 210.0])
    assert data["tag_size_mm"] == pytest.approx(22.5)
    assert data["cell_size_mm"] == pytest.approx(3.75)
    assert data["border_cells"] == 0.5
    assert len(data["markers"]) == 104
    assert len(set(data["tag_ids"])) == 104

    for marker in data["markers"]:
        tag = np.asarray(marker["corners_mm"], dtype=np.float64)
        face = np.asarray(marker["face_corners_mm"], dtype=np.float64)
        assert tag.mean(axis=0) == pytest.approx(face.mean(axis=0))
        assert np.linalg.norm(tag[1] - tag[0]) == pytest.approx(22.5)
        assert np.linalg.norm(tag[2] - tag[1]) == pytest.approx(22.5)
        assert np.linalg.norm(face[1] - face[0]) == pytest.approx(26.25)
        assert np.linalg.norm(face[2] - face[1]) == pytest.approx(26.25)

    config, _face_ids = load_cube_config(str(out_dir / "config.json"))
    assert config.border_cells == 0.5
    assert np.allclose(
        build_tag_corner_map(config)[0],
        np.asarray(data["markers"][0]["corners_mm"]),
    )

    vertices = _mesh_vertices_from_3mf(out_dir / "cube.3mf")
    assert vertices.min(axis=0) == pytest.approx([-78.75, -26.25, -105.0])
    assert vertices.max(axis=0) == pytest.approx([78.75, 26.25, 105.0])
    vertex_set = {tuple(np.round(vertex, 6)) for vertex in vertices}
    for corner in data["markers"][0]["corners_mm"]:
        assert tuple(np.round(corner, 6)) in vertex_set

    atlas = cv2.imread(str(out_dir / "mujoco" / "cube_atlas.png"), cv2.IMREAD_GRAYSCALE)
    first_tile = atlas[:112, :112]
    black = np.argwhere(first_tile < 128)
    assert black.min(axis=0).tolist() == [8, 8]
    assert black.max(axis=0).tolist() == [103, 103]
