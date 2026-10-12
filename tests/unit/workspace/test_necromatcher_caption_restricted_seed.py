"""Restricted-seed captions consume admitted provenance, never authenticate it."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import numpy as np

from restriction_fixture import restricted_case
from src.shared.python.workspace import necromatcher_caption as caption
from src.shared.python.workspace.necromatcher_fit_records import (
    build_native_fit_payload,
)

pytestmark = pytest.mark.unit


def restricted_fit() -> dict[str, Any]:
    """Metadata specimen only; public Library authentication is tested separately."""
    return {
        "provenance": {
            "operation": "restrict_initialization",
            "request_options": {
                "operation": "restrict_initialization",
                "initialization_source": "restricted_spline",
                "config": {"initialization_policy": "strict"},
            },
            "spline_interval_restriction": {"schema": "canonical-owner-specimen"},
            "spline_restriction_prior": {"policy": "selected_first_parent_pose"},
        },
        "evidence": {
            "original_fit": {
                "optimizer_ran": False,
                "converged": False,
                "initialization": None,
            }
        },
    }


def test_restriction_status_is_distinct_from_authored_projection() -> None:
    assert caption.restricted_seed_status(restricted_fit()) is True
    assert caption.authored_seed_status(restricted_fit()) is False
    assert caption.restricted_seed_status({}) is False


@pytest.mark.parametrize(
    "where,key,value",
    [
        ("original", "optimizer_ran", True),
        ("original", "optimizer_ran", 0),
        ("original", "converged", 0),
        (
            "original",
            "initialization",
            {"policy": "authored_range_project_zero_slopes"},
        ),
        ("provenance", "operation", "fit"),
        ("provenance", "spline_interval_restriction", None),
        ("provenance", "spline_restriction_prior", None),
        ("request", "operation", "fit"),
        ("request", "initialization_source", "preserved_spline"),
        ("config", "initialization_policy", "authored_range_project_zero_slopes"),
    ],
)
def test_contradictory_restricted_metadata_rejects(
    where: str, key: str, value: Any
) -> None:
    fit = restricted_fit()
    targets = {
        "original": fit["evidence"]["original_fit"],
        "provenance": fit["provenance"],
        "request": fit["provenance"]["request_options"],
        "config": fit["provenance"]["request_options"]["config"],
    }
    targets[where][key] = value
    with pytest.raises(ValueError):
        caption.restricted_seed_status(fit)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_restricted_flag_requires_native_bool(value: Any) -> None:
    with pytest.raises(ValueError):
        caption.CaptionFrame(0, Fraction(0), None, 0, restricted_seed=value)


def test_visible_restricted_label_and_mutually_exclusive_flags() -> None:
    frame = caption.CaptionFrame(
        190, Fraction(8008, 375), 20.0, 13, True, 0.35, restricted_seed=True
    )
    layout = caption.caption_layout((1280, 720), frame, caption.CaptionOverlayOptions())
    assert layout.lines[0].text == "UNOPTIMIZED RESTRICTED RESEARCH SEED"
    assert "Camera/Anatomy Unqualified" in layout.lines[1].text
    assert layout.rectangle[3] <= 144
    with pytest.raises(ValueError):
        caption.CaptionFrame(
            0, Fraction(0), None, 0, authored_seed=True, restricted_seed=True
        )


def manifest() -> dict[str, Any]:
    options = caption.CaptionOverlayOptions()
    frame = caption.CaptionFrame(0, Fraction(0), None, 0, restricted_seed=True)
    return {
        "image_size": [1280, 720],
        "caption_overlay": caption.caption_provenance(options),
        "restricted_initialization_seed": True,
        "frames": [
            {
                "frame_index": 0,
                "frame": {
                    "pts_ticks": 0,
                    "timebase_numerator": 1,
                    "timebase_denominator": 30,
                },
                "matched_rms_pixels": None,
                "matched_marker_count": 0,
                "caption_overlay": caption.caption_layout(
                    (1280, 720), frame, options
                ).to_record(),
            }
        ],
    }


@pytest.mark.parametrize("tamper", ["missing", "integer", "authored", "layout", "fit"])
def test_manifest_claim_cannot_override_admitted_fit(tamper: str) -> None:
    record = manifest()
    fit = restricted_fit()
    options = caption.CaptionOverlayOptions()
    caption.validate_caption_manifest(record, options, fit)
    if tamper == "missing":
        del record["restricted_initialization_seed"]
    elif tamper == "integer":
        record["restricted_initialization_seed"] = 1
    elif tamper == "authored":
        record["authored_initialization_seed"] = True
    elif tamper == "layout":
        record["frames"][0]["caption_overlay"]["lines"][0]["text"] = "Accepted"
    else:
        fit = {}
    with pytest.raises(ValueError):
        caption.validate_caption_manifest(record, options, fit)


def test_restricted_declaration_requires_authenticated_fit() -> None:
    with pytest.raises(ValueError):
        caption.validate_caption_manifest(manifest(), caption.CaptionOverlayOptions())


@pytest.mark.parametrize("tamper", [None, "parent", "prior", "receipt"])
def test_public_library_admits_seed_or_rejects_tampered_lineage(
    fit_case: Any, tmp_path: Path, tamper: str | None
) -> None:
    case = restricted_case(fit_case, tmp_path)
    record = build_native_fit_payload(
        case["request"],
        case["source"],
        case["result"],
        case["dense"],
        case["stamp"],
        0.0,
    )
    if tamper == "parent":
        record["provenance"]["warm_start_fit_hash"] = "sha256:" + "b" * 64
    elif tamper == "prior":
        record["provenance"]["spline_restriction_prior"]["frame_index"] = 1
    elif tamper == "receipt":
        record["provenance"]["spline_interval_restriction"]["restricted_start"][
            "spline_coefficients"
        ][0] += 0.01
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    if tamper:
        with pytest.raises(ValueError):
            case["library"].add_fit("caption-seed", "practice", path)
    else:
        case["library"].add_fit("caption-seed", "practice", path)
        loaded = case["library"].load_fit("caption-seed")
        assert caption.restricted_seed_status(loaded) is True


def test_existing_seven_argument_defaults_retain_layout() -> None:
    for authored in (False, True):
        old = caption.CaptionFrame(
            190, Fraction(8008, 375), 20.0, 13, True, 0.35, authored
        )
        explicit = caption.CaptionFrame(
            190, Fraction(8008, 375), 20.0, 13, True, 0.35, authored, False
        )
        assert caption.caption_layout(
            (1280, 720), old, caption.CaptionOverlayOptions()
        ) == (
            caption.caption_layout(
                (1280, 720), explicit, caption.CaptionOverlayOptions()
            )
        )


@pytest.mark.parametrize(
    "authored,pixel_sha,layout_sha",
    [
        (
            False,
            "3968f2ccdebbfbe6d2363981ae9b7885af9f57a77da409dee45c81f4d63c1ce5",
            "8c6c83f57a8b460cb05d65ab7f4acd907ef08b9fde7df778655e7e358e03163b",
        ),
        (
            True,
            "7792e5c296afffab51d3e9d833d19767152c0f66e43e5994bc262fd9cb07565f",
            "fcf8ae5c03f847b4fa98bb68369e2342254c71de5f4d155082e623ff1bc875af",
        ),
    ],
)
def test_prechange_authored_and_ordinary_pixels_are_exact(
    authored: bool, pixel_sha: str, layout_sha: str
) -> None:
    frame = caption.CaptionFrame(
        190, Fraction(8008, 375), 20.0, 13, True, 0.35, authored
    )
    layout = caption.caption_layout((1280, 720), frame, caption.CaptionOverlayOptions())
    image = np.full((720, 1280, 3), 127, dtype=np.uint8)
    caption.draw_caption(image, layout)
    assert hashlib.sha256(image.tobytes()).hexdigest() == pixel_sha
    assert (
        hashlib.sha256(
            json.dumps(layout.to_record(), sort_keys=True).encode()
        ).hexdigest()
        == layout_sha
    )
