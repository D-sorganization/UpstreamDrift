"""Typed-port extension of the attachment manifest schema (CMB-8, #11659)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.model_generation.editor.attachment_ports import (
    PortPolarity,
    PortType,
)
from src.tools.model_explorer.attachment_manifest import (
    attachment_sidecar_path,
    load_attachment_manifest,
)

pytestmark = pytest.mark.unit

SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "src/tools/model_explorer/attachment_manifest.schema.json"
)


def _load(tmp_path: Path, point: dict[str, object]):
    model = tmp_path / "m.urdf"
    model.write_text('<robot name="r"><link name="hand"/></robot>', encoding="utf-8")
    base = {"name": "p", "link_name": "hand", "role": "tool-mount"}
    attachment_sidecar_path(model).write_text(
        json.dumps({"schema_version": 1, "attachment_points": [{**base, **point}]}),
        encoding="utf-8",
    )
    return load_attachment_manifest(model)


def test_typed_point_round_trips(tmp_path: Path) -> None:
    result = _load(
        tmp_path,
        {
            "port_type": "grip",
            "polarity": "socket",
            "tags": ["left"],
            "max_payload_kg": 2.0,
        },
    )
    point = result.attachment_points[0]
    assert result.warnings == ()
    assert point.is_typed
    assert point.port_type is PortType.GRIP and point.polarity is PortPolarity.SOCKET
    typed = point.typed_port()
    assert typed is not None and typed.side == "left" and typed.max_payload_kg == 2.0
    payload = point.to_dict()
    assert payload["port_type"] == "grip" and payload["polarity"] == "socket"


def test_legacy_untyped_point_still_loads(tmp_path: Path) -> None:
    point = _load(tmp_path, {}).attachment_points[0]
    assert not point.is_typed and point.typed_port() is None
    assert "port_type" not in point.to_dict()


@pytest.mark.parametrize(
    "extra",
    [
        {"port_type": "flange", "polarity": "socket"},
        {"port_type": "grip", "polarity": "female"},
        {"port_type": "grip"},
        {"polarity": "plug"},
    ],
)
def test_bad_typing_warns_and_degrades_to_untyped(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    result = _load(tmp_path, extra)
    assert result.warnings
    assert not result.attachment_points[0].is_typed


def test_schema_declares_the_typed_fields() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    props = schema["properties"]["attachment_points"]["items"]["properties"]
    assert set(props["port_type"]["enum"]) == {t.value for t in PortType}
    assert set(props["polarity"]["enum"]) == {p.value for p in PortPolarity}
