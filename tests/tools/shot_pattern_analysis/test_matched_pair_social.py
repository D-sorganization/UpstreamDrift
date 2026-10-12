"""Contract checks for condition-pair social figures using canonical density."""

import inspect
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.tools.shot_pattern_analysis import matched_pair_social as social

pytestmark = pytest.mark.unit


def _context():
    return social.PairSocialContext(
        condition_labels=("SD 1°", "SD 2°"),
        scenario="Lean 5° · Curve ×1 · Combined 2",
        social_club="Driver",
        target_point_m=(180.0, 0.0),
        total_n=10,
    )


def test_context_is_immutable_and_validation_precedence_is_preserved(
    monkeypatch, tmp_path
):
    context = _context()
    with pytest.raises(FrozenInstanceError):
        context.total_n = 11  # type: ignore[misc]
    monkeypatch.setattr(social, "write_academic_dispersion_figure", lambda *a, **k: {})
    cases = [
        ({"condition_labels": ("", ""), "scenario": ""}, "condition labels"),
        ({"scenario": "", "source_protocols": ("", "")}, "scenario and social club"),
        ({"delivery_noise": "", "source_protocols": ("", "")}, "delivery noise"),
        ({"source_protocols": ("", ""), "target_modes": ("", "")}, "source_protocols"),
        ({"target_modes": ("", ""), "target_point_m": (True, 0.0)}, "target_modes"),
        ({"target_point_m": (True, 0.0), "total_n": 1}, "target_point_m"),
        ({"total_n": 1}, "total_n"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            social.write_pair_social_density(
                {}, tmp_path / "invalid", context=replace(context, **changes)
            )


def test_pair_social_writer_fits_architecture_budget():
    writer = social.write_pair_social_density
    assert len(inspect.getsourcelines(writer)[0]) <= 100
    assert len(inspect.signature(writer).parameters) <= 8


def _points():
    return {
        f"{pattern} · {condition}": np.array([(1.0, 2.0), (3.0, 4.0)])
        for pattern in ("Straight", "Draw", "Fade")
        for condition in ("SD 1°", "SD 2°")
    }


def _render(points, tmp_path):
    return social.write_pair_social_density(
        points,
        tmp_path / "comparison",
        context=social.PairSocialContext(
            condition_labels=("SD 1°", "SD 2°"),
            scenario="Lean 5° · Curve ×1 · Combined 2",
            social_club="Driver",
            target_point_m=(180.0, 0.0),
            total_n=10,
        ),
    )


def test_social_pairs_reuse_canonical_density_without_changing_coordinates(
    monkeypatch, tmp_path
):
    calls = []

    def capture(points, stem, **kwargs):
        calls.append((points, stem, kwargs))
        return {kind: Path(f"{stem}.{kind}") for kind in ("png", "pdf", "svg")}

    monkeypatch.setattr(social, "write_academic_dispersion_figure", capture)
    points = _points()
    outputs = _render(points, tmp_path)
    assert len(outputs) == len(set(outputs.values())) == 9
    for pattern, (plotted, _, kwargs) in zip(
        ("Straight", "Draw", "Fade"), calls, strict=True
    ):
        assert tuple(plotted) == ("SD 1°", "SD 2°")
        for condition in plotted:
            np.testing.assert_array_equal(
                plotted[condition], points[f"{pattern} · {condition}"]
            )
        assert kwargs["options"].presentation == "social"
        assert kwargs["target_point_m"] == (180.0, 0.0)
        assert pattern in kwargs["scenario"]
        metadata = kwargs["metadata"]
        assert metadata["valid_shots_by_pattern"] == {"SD 1°": 2, "SD 2°": 2}
        assert metadata["total_shots_by_pattern"] == {"SD 1°": 10, "SD 2°": 10}
        assert "All Six" in metadata["Population"]
        assert "Within This Figure" in metadata["Scales"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "unequal", "nan"])
def test_invalid_six_population_data_fails_before_any_export(
    monkeypatch, tmp_path, mutation
):
    calls = []
    monkeypatch.setattr(
        social, "write_academic_dispersion_figure", lambda *a, **k: calls.append(a)
    )
    points = _points()
    if mutation == "missing":
        points.pop("Straight · SD 1°")
    elif mutation == "extra":
        points["Unknown"] = points["Straight · SD 1°"]
    elif mutation == "unequal":
        points["Straight · SD 1°"] = np.array([(1.0, 2.0)])
    else:
        points["Straight · SD 1°"][0, 0] = np.nan
    with pytest.raises(ValueError):
        _render(points, tmp_path)
    assert not calls


def test_noise_details_use_wrapped_metadata_not_single_line_subtitle(
    monkeypatch, tmp_path
):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return {"png": tmp_path / "plot.png"}

    monkeypatch.setattr(social, "write_academic_dispersion_figure", fake_render)
    details = "Path SD 0.5° · Speed SD 2 mph · Strike SD 0.25 in on Each Axis"
    social.write_pair_social_density(
        _points(),
        tmp_path / "comparison",
        context=social.PairSocialContext(
            condition_labels=("SD 1°", "SD 2°"),
            scenario="Lean 5° · Curve ×1",
            social_club="Driver",
            target_point_m=(180, 0),
            total_n=10,
            delivery_noise=details,
        ),
    )
    assert len(calls) == 3
    assert all(call["metadata"]["Delivery Noise"] == details for call in calls)
    assert all(details not in call["scenario"] for call in calls)


def test_pair_social_density_includes_matched_target_and_ordered_protocols(
    monkeypatch, tmp_path
):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return {"png": tmp_path / "plot.png"}

    monkeypatch.setattr(social, "write_academic_dispersion_figure", fake_render)
    social.write_pair_social_density(
        _points(),
        tmp_path / "comparison",
        context=social.PairSocialContext(
            condition_labels=("SD 1°", "SD 2°"),
            scenario="Lean 5° · Curve ×1",
            social_club="Driver",
            target_point_m=(180.0, 0.0),
            total_n=10,
            source_protocols=("pw-primary-face-bounds/1", "pw-control-face-bounds/1"),
            target_modes=("Within-Lean Nominal Target", "Fixed-Target Control"),
        ),
    )
    assert len(calls) == 3
    for call in calls:
        meta = call["metadata"]
        assert "Matched Target" in meta
        assert (
            meta["Matched Target"]
            == "180.00 m downrange (SD 1°: Within-Lean Nominal Target · SD 2°: Fixed-Target Control)"
        )
        assert "Protocol" in meta
        assert (
            meta["Protocol"]
            == "SD 1°: pw-primary-face-bounds/1 · SD 2°: pw-control-face-bounds/1"
        )


def test_pair_social_density_supports_nonzero_lateral_target(monkeypatch, tmp_path):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return {"png": tmp_path / "plot.png"}

    monkeypatch.setattr(social, "write_academic_dispersion_figure", fake_render)
    social.write_pair_social_density(
        _points(),
        tmp_path / "comparison",
        context=social.PairSocialContext(
            condition_labels=("SD 1°", "SD 2°"),
            scenario="Lean 5° · Curve ×1",
            social_club="Driver",
            target_point_m=(180.0, 3.5),
            total_n=10,
            source_protocols=("pw-primary-face-bounds/1", "pw-control-face-bounds/1"),
            target_modes=("Within-Lean Nominal Target", "Fixed-Target Control"),
        ),
    )
    assert len(calls) == 3
    for call in calls:
        meta = call["metadata"]
        assert "Matched Target" in meta
        assert (
            meta["Matched Target"]
            == "180.00 m downrange, 3.50 m lateral (SD 1°: Within-Lean Nominal Target · SD 2°: Fixed-Target Control)"
        )


def test_pair_social_density_default_without_source_protocols_still_includes_target(
    monkeypatch, tmp_path
):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return {"png": tmp_path / "plot.png"}

    monkeypatch.setattr(social, "write_academic_dispersion_figure", fake_render)
    social.write_pair_social_density(
        _points(),
        tmp_path / "comparison",
        context=social.PairSocialContext(
            condition_labels=("SD 1°", "SD 2°"),
            scenario="Lean 5° · Curve ×1",
            social_club="Driver",
            target_point_m=(180.0, 0.0),
            total_n=10,
        ),
    )
    assert len(calls) == 3
    for call in calls:
        meta = call["metadata"]
        assert "Matched Target" in meta
        assert meta["Matched Target"] == "180.00 m downrange (Common Target)"
        assert "Protocol" not in meta


@pytest.mark.parametrize(
    "invalid_protocols",
    [
        ("only_one",),
        ("one", "two", "three"),
        ("", "valid"),
        ("valid", "   "),
        (123, "valid"),
    ],
)
def test_pair_social_density_invalid_protocols_raises(
    monkeypatch, tmp_path, invalid_protocols
):
    monkeypatch.setattr(social, "write_academic_dispersion_figure", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="source_protocols"):
        social.write_pair_social_density(
            _points(),
            tmp_path / "comparison",
            context=social.PairSocialContext(
                condition_labels=("SD 1°", "SD 2°"),
                scenario="Lean 5° · Curve ×1",
                social_club="Driver",
                target_point_m=(180.0, 0.0),
                total_n=10,
                source_protocols=invalid_protocols,  # type: ignore[arg-type]
            ),
        )


@pytest.mark.parametrize(
    "invalid_modes",
    [
        ("only_one",),
        ("one", "two", "three"),
        ("", "valid"),
        ("valid", "   "),
        (123, "valid"),
    ],
)
def test_pair_social_density_invalid_target_modes_raises(
    monkeypatch, tmp_path, invalid_modes
):
    monkeypatch.setattr(social, "write_academic_dispersion_figure", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="target_modes"):
        social.write_pair_social_density(
            _points(),
            tmp_path / "comparison",
            context=social.PairSocialContext(
                condition_labels=("SD 1°", "SD 2°"),
                scenario="Lean 5° · Curve ×1",
                social_club="Driver",
                target_point_m=(180.0, 0.0),
                total_n=10,
                target_modes=invalid_modes,  # type: ignore[arg-type]
            ),
        )


@pytest.mark.parametrize(
    "invalid_target",
    [
        (180.0,),
        (180.0, 0.0, 1.0),
        (float("nan"), 0.0),
        (180.0, float("inf")),
        (True, 0.0),
        (180.0, False),
        ("180", 0.0),
    ],
)
def test_pair_social_density_invalid_target_point_raises(
    monkeypatch, tmp_path, invalid_target
):
    monkeypatch.setattr(social, "write_academic_dispersion_figure", lambda *a, **kw: {})
    with pytest.raises(ValueError, match="target_point_m"):
        social.write_pair_social_density(
            _points(),
            tmp_path / "comparison",
            context=social.PairSocialContext(
                condition_labels=("SD 1°", "SD 2°"),
                scenario="Lean 5° · Curve ×1",
                social_club="Driver",
                target_point_m=invalid_target,  # type: ignore[arg-type]
                total_n=10,
            ),
        )
