"""Parity tests for the Rust-backed MJCF facade (UD #5243).

These tests are skipped when the ``upstream_urdf`` Rust extension is
not importable or does not expose the MJCF entry points, which is the
expected state on CI runners that have not yet been provisioned with
the wheel.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

upstream_urdf = pytest.importorskip("upstream_urdf")
if not hasattr(upstream_urdf, "parse_mjcf"):
    pytest.skip(
        "installed upstream_urdf lacks MJCF entry points",
        allow_module_level=True,
    )

from model_generation.converters._mjcf_rust_facade import (  # noqa: E402
    HAVE_RUST,
    parse_mjcf_to_dict,
    write_mjcf_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# A representative spread that covers the 80% case the Rust parser
# models semantically (compiler/option/asset/body/joint/geom/inertial/
# actuator). Missing files are skipped per-fixture.
GOLDEN_MJCFS = [
    REPO_ROOT / "src/engines/physics_engines/mujoco/models/simple_pendulum.xml",
    REPO_ROOT / "rust_core/upstream-urdf/tests/fixtures/simple_pendulum.xml",
    REPO_ROOT / "rust_core/upstream-urdf/tests/fixtures/two_link_arm.xml",
]


def _existing_goldens() -> list[Path]:
    return [p for p in GOLDEN_MJCFS if p.exists()]


@pytest.mark.parametrize("mjcf_path", _existing_goldens(), ids=lambda p: p.name)
def test_rust_mjcf_round_trip(mjcf_path: Path) -> None:
    """Parse → write → parse: structural equality through the Python facade."""
    assert HAVE_RUST, "upstream_urdf wheel with MJCF support should be installed"
    xml = mjcf_path.read_text()
    ast = parse_mjcf_to_dict(xml)
    rendered = write_mjcf_from_dict(ast)
    ast2 = parse_mjcf_to_dict(rendered)
    assert ast == ast2, f"Round-trip mismatch for {mjcf_path}"


def test_minimal_mjcf_shape() -> None:
    """The dict shape matches what `_mjcf_rust_facade` documents."""
    xml = (
        '<mujoco model="m"><worldbody><body name="b" pos="1 2 3"/></worldbody></mujoco>'
    )
    ast = parse_mjcf_to_dict(xml)
    assert ast["model"] == "m"
    assert len(ast["worldbody"]["bodies"]) == 1
    assert ast["worldbody"]["bodies"][0]["name"] == "b"
    assert ast["worldbody"]["bodies"][0]["pos"] == [1.0, 2.0, 3.0]


def test_write_then_parse_is_valid_xml() -> None:
    """Writer output is parseable by the standard library."""
    xml = (
        '<mujoco model="m"><worldbody><body name="b" pos="0 0 0"/></worldbody></mujoco>'
    )
    ast = parse_mjcf_to_dict(xml)
    rendered = write_mjcf_from_dict(ast)
    # The Rust writer prepends an XML declaration — make sure it parses.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(rendered)
    assert root.tag == "mujoco"


def test_facade_handles_missing_optional_fields() -> None:
    """Round-tripping an AST with only required fields succeeds."""
    minimal = {
        "model": "x",
        "worldbody": {
            "bodies": [
                {
                    "name": "b1",
                    "pos": [0.0, 0.0, 0.0],
                    "geoms": [{"type_": "sphere", "size": [0.1]}],
                }
            ]
        },
    }
    rendered = write_mjcf_from_dict(minimal)
    reparsed = parse_mjcf_to_dict(rendered)
    assert reparsed["model"] == "x"
    assert reparsed["worldbody"]["bodies"][0]["name"] == "b1"


def test_extras_preserved_through_round_trip() -> None:
    """Unmodelled sections like <sensor> survive a round-trip via RawSection."""
    xml = (
        '<mujoco model="m">'
        "<worldbody><body name='b' pos='0 0 0'/></worldbody>"
        "<sensor><accelerometer name='a' site='b'/></sensor>"
        "</mujoco>"
    )
    ast = parse_mjcf_to_dict(xml)
    rendered = write_mjcf_from_dict(ast)
    assert "sensor" in rendered
    assert "accelerometer" in rendered
    # And the round-trip is structurally stable.
    ast2 = parse_mjcf_to_dict(rendered)
    assert ast == ast2


def test_json_round_trip_is_stable() -> None:
    """Serialising/deserialising the AST through json is lossless."""
    xml = Path(
        REPO_ROOT / "rust_core/upstream-urdf/tests/fixtures/two_link_arm.xml"
    ).read_text()
    ast = parse_mjcf_to_dict(xml)
    blob = json.dumps(ast)
    ast_back = json.loads(blob)
    assert ast == ast_back
