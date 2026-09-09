"""Exercise the public shaft input wire through the selected Tools provider."""

from __future__ import annotations

import importlib
import json
from hashlib import sha256
from pathlib import Path

import pytest

from tests.shared_contracts.test_tools_provider_contracts import (
    _assert_from_tools,
    _fresh_provider_import,
)

_SOURCE_BYTES = b"UpstreamDrift synthetic shaft interchange contract; not measured"
_FIXTURE = Path(__file__).parent / "fixtures" / "impact_shaft_v1.json"
pytestmark = pytest.mark.integration


def test_shaft_wire_preserves_coupled_inputs_and_unqualified_evidence() -> None:
    with _fresh_provider_import("golf_club"):
        provider = importlib.import_module("golf_club.shaft_model_data")
        _assert_from_tools(Path(provider.__file__))
        assert provider.DISTRIBUTED_SHAFT_FORMAT == "golf_club.distributed_shaft/1"
        model = provider.shaft_model_from_json(_FIXTURE.read_text(encoding="utf-8"))
        canonical = provider.shaft_model_to_json(model)
        assert (
            provider.shaft_model_to_json(provider.shaft_model_from_json(canonical))
            == canonical
        )
        assert (
            provider.shaft_model_digest(model)
            == sha256(canonical.encode("utf-8")).hexdigest()
        )
        section = model.sections[0]
        elastic, inertia = section.elastic, section.inertia
        sample = inertia.samples[0]
        body = sample.body
        assert elastic.stiffness[0][4] == 3.0
        assert body.mass_kg == 0.08
        assert model.validation_status == "unqualified"
        digest = sha256(_SOURCE_BYTES).hexdigest()
        assert provider.verify_shaft_source_bytes(model, {digest: _SOURCE_BYTES}) == (
            digest,
        )
        assert model.validation_status == "unqualified"
        source = model.sources[0]
        assert source.calibration_sha256 is None
        with pytest.raises(ValueError, match="digest mismatch"):
            provider.verify_shaft_source_bytes(model, {digest: b"altered source"})


@pytest.mark.parametrize("change", ["format", "missing", "coercion", "qualification"])
def test_shaft_consumer_refuses_unsupported_or_promoted_inputs(change: str) -> None:
    payload = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    if change == "format":
        payload["format"] = "golf_club.distributed_shaft/2"
    elif change == "missing":
        del payload["sources"][0]["calibration_sha256"]
    elif change == "coercion":
        payload["sections"][0]["length_m"] = "0.6"
    else:
        payload["validation_status"] = "physically-validated"
    with _fresh_provider_import("golf_club"):
        provider = importlib.import_module("golf_club.shaft_model_data")
        with pytest.raises((TypeError, ValueError)):
            provider.shaft_model_from_json(json.dumps(payload))


def test_paired_theme_keeps_resolved_tokens_and_custom_extension() -> None:
    with _fresh_provider_import("theme"):
        provider = importlib.import_module("shared.python.theme.api")
        _assert_from_tools(Path(provider.__file__))
        required = (
            "bg",
            "group_bg",
            "input_bg",
            "border",
            "text",
            "text_secondary",
            "label",
            "focus",
            "accent",
            "title_bg",
            "title_border",
            "table_header",
            "table_alt",
            "button_hover",
        )
        colors = provider.ThemeColors(
            **dict.fromkeys(required, "#202020"),
            is_dark=True,
            custom_impact_trace="#123456",
        )
        resolved = colors.as_dict()
        assert resolved["bg_base"] == "#202020"
        assert resolved["primary"] == "#202020"
        assert resolved["custom_impact_trace"] == "#123456"
        assert resolved["is_dark"] is True
        resolved["bg"] = "#FFFFFF"
        assert colors.as_dict()["bg"] == "#202020"
