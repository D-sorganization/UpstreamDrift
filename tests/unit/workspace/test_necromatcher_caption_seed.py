"""Seed status is derived from canonical authored fit provenance, not display options."""

from copy import deepcopy
from fractions import Fraction

import pytest

pytestmark = pytest.mark.unit


def seed_fit():
    return {
        "provenance": {
            "operation": "author_initialization",
            "request_options": {
                "operation": "author_initialization",
                "config": {
                    "initialization_policy": "authored_range_project_zero_slopes"
                },
            },
        },
        "evidence": {
            "original_fit": {
                "optimizer_ran": False,
                "converged": False,
                "initialization": {"policy": "authored_range_project_zero_slopes"},
            }
        },
    }


def test_seed_label_requires_canonical_authored_operation_and_no_optimizer():
    from src.shared.python.workspace.necromatcher_caption import authored_seed_status

    assert authored_seed_status(seed_fit()) is True
    assert authored_seed_status({}) is False
    fit = seed_fit()
    fit["provenance"]["operation"] = "fit"
    assert authored_seed_status(fit) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_ran", True),
        ("optimizer_ran", 0),
        ("converged", True),
        ("converged", 0),
        ("initialization", None),
    ],
)
def test_ambiguous_authored_provenance_fails_closed(field, value):
    from src.shared.python.workspace.necromatcher_caption import authored_seed_status

    fit = seed_fit()
    fit["evidence"]["original_fit"][field] = value
    with pytest.raises(ValueError):
        authored_seed_status(fit)


def test_seed_layout_is_conspicuous_and_retains_qualifications():
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
    )

    frame = CaptionFrame(190, Fraction(8008, 375), 20.0, 13, True, 0.35, True)
    layout = caption_layout((1280, 720), frame, CaptionOverlayOptions())
    text = " ".join(line.text for line in layout.lines)
    assert "UNOPTIMIZED AUTHORED RESEARCH SEED" in text
    for phrase in ("Camera/Anatomy Unqualified", "Physical Time Unknown", "Surfaces"):
        assert phrase in text
    assert layout.rectangle[3] <= 144
    with pytest.raises(ValueError):
        CaptionFrame(0, Fraction(0), None, 0, authored_seed=1)


def test_forged_seed_manifest_cannot_override_canonical_fit():
    from src.shared.python.workspace.necromatcher_caption import (
        CaptionFrame,
        CaptionOverlayOptions,
        caption_layout,
        caption_provenance,
        validate_caption_manifest,
    )

    opts = CaptionOverlayOptions()
    frame = CaptionFrame(0, Fraction(0), None, 0, authored_seed=True)
    manifest = {
        "image_size": [1280, 720],
        "caption_overlay": caption_provenance(opts),
        "authored_initialization_seed": True,
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
                "caption_overlay": caption_layout((1280, 720), frame, opts).to_record(),
            }
        ],
    }
    validate_caption_manifest(manifest, opts, seed_fit())
    with pytest.raises(ValueError):
        validate_caption_manifest(manifest, opts, {})
    with pytest.raises(ValueError):
        validate_caption_manifest(manifest, opts)
    bad = deepcopy(manifest)
    bad["authored_initialization_seed"] = 1
    with pytest.raises(ValueError):
        validate_caption_manifest(bad, opts, seed_fit())


@pytest.mark.parametrize(
    "malformed",
    [
        [],
        {"provenance": []},
        {"provenance": {"operation": "author_initialization"}, "evidence": []},
    ],
)
def test_malformed_fit_metadata_rejects_with_contract_error(malformed):
    from src.shared.python.workspace.necromatcher_caption import authored_seed_status

    with pytest.raises(ValueError):
        authored_seed_status(malformed)
