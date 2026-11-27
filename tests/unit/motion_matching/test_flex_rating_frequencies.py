"""Review of ``SHAFT_FLEX_DATA`` cpm values against cited sources (#15150 C19).

Two independent driver tables are recorded in the fixture with their clamp and
length conventions: Hireko Golf (graphite driver means and ranges, 45 in,
butt-clamped assembled club) and Steadfast Golf (45.5 in, clamp not stated).
The former table (R 260, S 280, X 300 cpm) sat 14, 22 and 29 cpm above the
Hireko means and 19, 22 and 27 cpm above the Steadfast values, so the settings
file ``src/config/shaft_flex_frequencies.json`` now holds the mid-range of the
two sources (A 230, R 244, S 258, X 272 cpm). Ladies is the Hireko L mean
(tabulated at 44 in) moved to 45 in with the cited length slope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.club_configurations import (  # noqa: E501
    SHAFT_FLEX_DATA,
    SHAFT_FLEX_SETTINGS,
    flex_cpm_at_length,
    load_shaft_flex_settings,
)

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "shaft_flex"
    / "driver_frequency_by_flex.json"
)
SOURCE = json.loads(FIXTURE.read_text(encoding="utf-8"))
FLEXES = sorted(SOURCE["repo_flex_to_source_flex"])
# Spread between the two independent sources is up to 5 cpm, and the flex steps
# are 10-14 cpm; 10 cpm keeps each flex unambiguously on its own step.
MEAN_TOLERANCE_CPM = 10.0


def _row(repo_flex: str) -> dict[str, float]:
    return SOURCE["graphite_driver"][SOURCE["repo_flex_to_source_flex"][repo_flex]]


def _hireko_mean_at_45_in(repo_flex: str) -> float:
    """Hireko mean moved to 45 in (only L flex was tabulated at 44 in)."""
    mean = _row(repo_flex)["mean"]
    if SOURCE["repo_flex_to_source_flex"][repo_flex] == "L":
        mean -= SHAFT_FLEX_SETTINGS["length_slope_cpm_per_in"] * (45.0 - 44.0)
    return mean


def test_fixture_records_citation_and_convention() -> None:
    for key in ("citation", "url", "accessed", "convention", "slope_assumption"):
        assert SOURCE[key]
    assert SOURCE["second_source"]["url"] != SOURCE["url"]
    assert SOURCE["second_source"]["convention"]["club_length_in"] == 45.5
    assert SOURCE["resolution_cpm"] == 1


def test_settings_file_records_sources_and_reference_length() -> None:
    assert SHAFT_FLEX_SETTINGS["reference_club_length_in"] == 45.0
    for key in ("flex_cpm", "length_slope_cpm_per_in", "reference_club_length_in"):
        assert SHAFT_FLEX_SETTINGS["sources"][key]
    assert "hirekogolf.com" in SHAFT_FLEX_SETTINGS["sources"]["flex_cpm"]
    assert "steadfastgolf.com" in SHAFT_FLEX_SETTINGS["sources"]["flex_cpm"]


def test_shaft_flex_data_is_driven_by_the_settings_file() -> None:
    for flex, cpm in SHAFT_FLEX_SETTINGS["flex_cpm"].items():
        assert SHAFT_FLEX_DATA[flex]["stiffness_cpm"] == cpm


@pytest.mark.parametrize("flex", FLEXES)
def test_repo_frequency_inside_published_range(flex: str) -> None:
    row = _row(flex)
    assert row["low"] <= SHAFT_FLEX_DATA[flex]["stiffness_cpm"] <= row["high"]


def test_repo_frequencies_increase_with_flex_like_source() -> None:
    order = ["Ladies", "Senior", "Regular", "Stiff", "X-Stiff"]
    repo = [SHAFT_FLEX_DATA[f]["stiffness_cpm"] for f in order]
    src = [_row(f)["mean"] for f in order]
    assert repo == sorted(repo) and src == sorted(src)


@pytest.mark.parametrize("flex", FLEXES)
def test_repo_frequency_near_hireko_mean(flex: str) -> None:
    gap = abs(SHAFT_FLEX_DATA[flex]["stiffness_cpm"] - _hireko_mean_at_45_in(flex))
    assert gap <= MEAN_TOLERANCE_CPM


@pytest.mark.parametrize("flex", ["Senior", "Regular", "Stiff", "X-Stiff"])
def test_repo_frequency_near_second_source(flex: str) -> None:
    second = SOURCE["second_source"]["driver_cpm"]
    cited = second[SOURCE["repo_flex_to_source_flex"][flex]]
    assert abs(SHAFT_FLEX_DATA[flex]["stiffness_cpm"] - cited) <= MEAN_TOLERANCE_CPM


def test_length_slope_inside_cited_range() -> None:
    low, high = SHAFT_FLEX_SETTINGS["length_slope_range_cpm_per_in"]
    assert (low, high) == (8.6, 10.0)
    assert low <= SHAFT_FLEX_SETTINGS["length_slope_cpm_per_in"] <= high


def test_longer_club_reads_softer_by_the_slope() -> None:
    slope = SHAFT_FLEX_SETTINGS["length_slope_cpm_per_in"]
    assert flex_cpm_at_length("Stiff", 45.0) == pytest.approx(258.0)
    assert flex_cpm_at_length("Stiff", 46.0) == pytest.approx(258.0 - slope)
    assert flex_cpm_at_length("Stiff", 44.0) == pytest.approx(258.0 + slope)


def test_flex_cpm_at_length_contract() -> None:
    with pytest.raises(ValueError, match="unknown flex"):
        flex_cpm_at_length("Noodle", 45.0)
    with pytest.raises(ValueError, match="length_in must be positive"):
        flex_cpm_at_length("Stiff", 0.0)


def test_loader_rejects_non_increasing_or_non_positive_values(tmp_path: Path) -> None:
    base = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "src/config/shaft_flex_frequencies.json"
        ).read_text(encoding="utf-8")
    )
    bad_order = {**base, "flex_cpm": {"Ladies": 250, "Senior": 240}}
    bad_value = {**base, "flex_cpm": {"Ladies": 0, "Senior": 240}}
    bad_slope = {**base, "length_slope_cpm_per_in": 0.0}
    for name, doc, message in (
        ("order", bad_order, "increase strictly"),
        ("value", bad_value, "must be positive"),
        ("slope", bad_slope, "length_slope_cpm_per_in"),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_shaft_flex_settings(path)
