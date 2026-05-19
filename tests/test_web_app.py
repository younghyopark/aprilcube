import json
import re
import shutil
import subprocess

import pytest

from aprilcube.generate import DICT_MAP, TagPatternGenerator
from aprilcube.web.app import designer_path


def test_static_designer_assets_are_standalone():
    html = designer_path().read_text(encoding="utf-8")

    assert 'href="./styles.css"' in html
    assert 'src="./app.js"' in html
    assert 'src="./marker-data.js"' in html
    assert "/api/export" not in html
    assert "/api/dictionaries" not in html
    assert "Download YAML" in html
    assert "Generate Printable Files" in html
    assert "output-3mf" in html


def test_marker_data_matches_python_generator_for_samples():
    data = _load_marker_data()

    for dict_name, ids in {
        "apriltag_16h5": [0, 5, 29],
        "4x4_50": [0, 17, 49],
        "apriltag_36h11": [0, 42, 586],
    }.items():
        gen = TagPatternGenerator(DICT_MAP[dict_name])
        payload = data[dict_name]
        assert payload["marker_pixels"] == gen.marker_pixels
        assert payload["max_ids"] == gen.max_id

        bits = _unpack_bits(payload["data"], payload["bit_count"])
        marker_bits = gen.marker_pixels * gen.marker_pixels
        for marker_id in ids:
            start = marker_id * marker_bits
            expected = [bool(v) for v in gen.generate(marker_id).reshape(-1)]
            actual = bits[start : start + marker_bits]
            assert actual == expected


def test_web_cli_no_open_prints_designer_path():
    result = subprocess.run(
        ["aprilcube", "web", "--no-open"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == str(designer_path().resolve())


def test_client_javascript_syntax_is_valid():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    subprocess.run(
        [node, "--check", str(designer_path().with_name("app.js"))],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_marker_data():
    marker_js = designer_path().with_name("marker-data.js").read_text(encoding="utf-8")
    match = re.search(r"window\.APRILCUBE_MARKERS = (.*);\s*$", marker_js)
    assert match is not None
    return json.loads(match.group(1))


def _unpack_bits(data_b64, bit_count):
    import base64

    raw = base64.b64decode(data_b64)
    bits = []
    for index in range(bit_count):
        byte = raw[index >> 3]
        bits.append(bool(byte & (1 << (7 - (index & 7)))))
    return bits
