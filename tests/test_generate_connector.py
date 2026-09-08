import json
import sys
import textwrap
import zipfile
import xml.etree.ElementTree as ET

import pytest

from aprilcube import generate


def _mesh_bounds_from_3mf(path):
    with zipfile.ZipFile(path) as zf:
        model_xml = zf.read("3D/Objects/object_1.model")
    root = ET.fromstring(model_xml)
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    vertices = root.findall(".//m:vertex", ns)
    coords = [
        (
            float(vertex.attrib["x"]),
            float(vertex.attrib["y"]),
            float(vertex.attrib["z"]),
        )
        for vertex in vertices
    ]
    mins = tuple(min(coord[idx] for coord in coords) for idx in range(3))
    maxs = tuple(max(coord[idx] for coord in coords) for idx in range(3))
    return mins, maxs


def test_generate_end_effector_connector_extends_from_positive_z(tmp_path, monkeypatch):
    out_dir = tmp_path / "mounted_cube"
    rod_length = 20.0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aprilcube generate",
            "--grid",
            "1x1x1",
            "--dict",
            "4x4_50",
            "--tag-size",
            "12",
            "--end-effector-connector",
            "--connector-rod-length",
            str(rod_length),
            "-o",
            str(out_dir),
        ],
    )
    generate.main()

    config = json.loads((out_dir / "config.json").read_text())
    attachment = config["attachment"]
    top_z = config["box_dims"][2] / 2.0

    assert attachment["type"] == "end_effector_connector"
    assert attachment["axis"] == "+Z"
    assert attachment["connector_stl"] == generate.DEFAULT_CONNECTOR_STL
    assert attachment["connector_rotation_degrees"] == [180.0, 0.0, 0.0]
    assert attachment["rod_length_mm"] == rod_length
    assert attachment["rod_radius_mm"] == 10.0
    assert attachment["surface_z_mm"] == pytest.approx(top_z)
    assert attachment["rod_start_z_mm"] == pytest.approx(top_z)
    assert attachment["rod_end_z_mm"] == pytest.approx(top_z + rod_length)

    mesh_min, mesh_max = _mesh_bounds_from_3mf(out_dir / "cube.3mf")
    assert mesh_min[2] == pytest.approx(-top_z)
    assert mesh_max[2] == pytest.approx(top_z + rod_length + 15.0)

    obj_text = (out_dir / "mujoco" / "cube.obj").read_text()
    assert "usemtl connector_material" in obj_text


def test_connector_stl_vertices_are_flipped_upside_down():
    box_dims = (20.0, 20.0, 20.0)
    attachment = generate.AttachmentConfig(
        enabled=True,
        rod_length_mm=20.0,
    )

    vertices, _faces, _metadata = generate.build_end_effector_attachment_mesh(
        box_dims, attachment,
    )
    stl_path = generate._resolve_connector_stl_path(None)
    stl_vertices, _stl_faces = generate._load_stl_mesh(stl_path)

    rotated_stl = [(x, -y, -z) for x, y, z in stl_vertices]
    stl_min, stl_max = generate._mesh_bounds(rotated_stl)
    stl_center_x = (stl_min[0] + stl_max[0]) / 2.0
    stl_center_y = (stl_min[1] + stl_max[1]) / 2.0
    z_offset = box_dims[2] / 2.0 + attachment.rod_length_mm - stl_min[2]
    expected = (
        rotated_stl[0][0] - stl_center_x,
        rotated_stl[0][1] - stl_center_y,
        rotated_stl[0][2] + z_offset,
    )
    connector_start = 2 + 2 * generate.DEFAULT_CONNECTOR_SEGMENTS

    assert vertices[connector_start] == pytest.approx(expected)


def test_yaml_attachment_spec_parses(tmp_path):
    spec_path = tmp_path / "target.yaml"
    spec_path.write_text(
        textwrap.dedent(
            """
            output: mounted_cube
            shape:
              type: cuboid
              grid: [1, 1, 1]
            dictionary: 4x4_50
            attachment:
              type: end_effector_connector
              rod_length_mm: 35
              rod_radius_mm: 8
            """
        )
    )

    spec = generate.load_generation_spec(spec_path)

    assert spec.attachment is not None
    assert spec.attachment.enabled is True
    assert spec.attachment.rod_length_mm == 35
    assert spec.attachment.rod_radius_mm == 8
