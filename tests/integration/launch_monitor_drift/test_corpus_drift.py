"""ADR-0046 G0.1 drift gate: private corpus, UD stack vs vendored Tools stack.

ADR-0048 row 3 of the five ``needs-decision`` rows, and the row the ADR's
three-way taxonomy has no bucket for. Both stacks export ``load_private_corpus``
and both read the *same physical dataset* — same ``LAUNCH_MONITOR_DATA_ROOT``
environment variable, same ``data/authority/database/shot_corpus_parquet``
partition tree — but they read it for opposite reasons, and neither is a subset
of the other.

    UD    ``launch_monitor/corpus.py`` (197) — canonicalises the source-native
          imperial columns into the ADR-0031 schema and derives shot identity.
          Validates nothing about the corpus itself.
    Tools ``rate_of_closure/launch_monitor_private_corpus.py`` (106) —
          validates the manifest digest, schema version, retained-row cap,
          row count and source-partition set, then hands back the *native*
          columns untouched.

This module exercises both loaders against one synthetic two-source corpus
built in a temporary directory, and asserts the complementary guarantees as
executable claims rather than prose.

AGREE — asserted exactly
    * Path resolution: ``corpus_dataset_path(root)`` and
      ``resolve_private_corpus_path(root)`` return the identical
      ``Path`` for the same checkout root, and both honour the same
      ``LAUNCH_MONITOR_DATA_ROOT`` environment variable.
    * Both return all 4 rows across both ``source_id`` partitions, and both
      return exactly 15 columns.
    * Both fail closed when no root is supplied and the environment variable is
      unset (UD ``FileNotFoundError``, Tools ``ValueError``).

DIFFER — documented and pinned below
    D28. **The output schemas are almost disjoint.** Of 15 columns each, only
         ``club`` and ``smash_factor`` are shared: 13 UD-only names against 13
         Tools-only names. A caller cannot substitute one loader for the other.
    D29. **Unit canonicalisation exists only in UD**, with the ADR-0031 factors
         mph -> m/s 0.44704, yd -> m 0.9144, deg -> rad 0.017453292519943295 and
         rpm -> rad/s 0.10471975511965977. Tools returns the native imperial
         values. UD additionally drops ``apex_native`` (its unit varies by
         source, so it cannot be converted safely) and Tools passes it through.
    D30. **Manifest validation exists only in Tools**, and UD accepts every
         corpus Tools refuses. Five fail-closed checks — missing manifest,
         unsupported ``schema_version``, a ``total_rows`` above the 300,000-row
         desktop cap, a row-count mismatch, and a source-set mismatch — reject
         the load in Tools and are accepted silently by UD, which returns the
         same 4 rows in all five cases. Tools alone reports the
         content-addressed ``manifest_sha256`` and the privacy-safe
         ``source_name`` label built from it.
    D31. **Selection pushdown exists only in UD.** UD takes a ``source_id``
         allowlist and a canonical-metric allowlist, pushes both into the
         Parquet reader, and raises on an unknown value in either. Tools reads
         the whole corpus every time and has no selection parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.shared.python.launch_monitor.corpus import (  # noqa: E402
    CORPUS_COLUMN_MAP,
    corpus_dataset_path,
)
from src.shared.python.launch_monitor.corpus import (  # noqa: E402
    load_private_corpus as ud_load_private_corpus,
)
from src.shared.python.launch_monitor.schema import METRICS  # noqa: E402
from tests.integration.launch_monitor_drift.conftest import (  # noqa: E402
    require_vendored_tools_stack,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

require_vendored_tools_stack()

from rate_of_closure.launch_monitor_linked_scatter import (  # noqa: E402
    MAX_RETAINED_ROWS,
)
from rate_of_closure.launch_monitor_private_corpus import (  # noqa: E402
    CORPUS_RELATIVE_PATH,
    PRIVATE_DATA_ENV,
    resolve_private_corpus_path,
)
from rate_of_closure.launch_monitor_private_corpus import (  # noqa: E402
    load_private_corpus as tools_load_private_corpus,
)

EXPECTED_ROW_COUNT = 4
EXPECTED_COLUMN_COUNT = 15
EXPECTED_SOURCE_COUNT = 2
EXPECTED_DESKTOP_ROW_CAP = 300_000

# D28 pins.
SHARED_COLUMNS = {"club", "smash_factor"}
UD_ONLY_COLUMNS = {
    "ball_speed",
    "carry_distance",
    "club_speed",
    "flight_time",
    "lateral_carry",
    "launch_angle",
    "monitor_vendor",
    "observation_kind",
    "session_id",
    "shot_id",
    "source_row",
    "spin_rate",
    "total_distance",
}
TOOLS_ONLY_COLUMNS = {
    "apex_native",
    "ball_speed_mph",
    "carry_yd",
    "club_speed_mph",
    "file",
    "flight_time_s",
    "lateral_carry_yd",
    "launch_angle_deg",
    "monitor",
    "row_index",
    "source_id",
    "spin_rate_rpm",
    "total_yd",
}

# D29 pins: the ADR-0031 canonicalisation factors UD applies and Tools does not.
CANONICAL_UNIT_FACTORS = {
    "mph": 0.44704,
    "yd": 0.9144,
    "deg": 0.017453292519943295,
    "rpm": 0.10471975511965977,
    "s": 1.0,
    "1": 1.0,
}

CORPUS_ROWS = pd.DataFrame(
    {
        "monitor": ["TrackMan", "TrackMan", "FlightScope Mevo+", "FlightScope Mevo+"],
        "file": ["a.csv", "a.csv", "b.csv", "b.csv"],
        "row_index": [0, 1, 0, 1],
        "club": ["Driver", "Driver", "7 Iron", "7 Iron"],
        "club_speed_mph": [100.0, 101.0, 80.0, 81.0],
        "ball_speed_mph": [150.0, 151.5, 110.0, 111.0],
        "smash_factor": [1.5, 1.5, 1.375, 1.37],
        "launch_angle_deg": [12.0, 12.5, 18.0, 18.5],
        "spin_rate_rpm": [2700.0, 2750.0, 6500.0, 6550.0],
        "carry_yd": [250.0, 252.0, 165.0, 166.0],
        "total_yd": [270.0, 272.0, 172.0, 173.0],
        "lateral_carry_yd": [3.0, -2.0, 1.0, -1.5],
        "flight_time_s": [6.1, 6.2, 5.0, 5.1],
        "apex_native": [95.0, 96.0, 28.0, 29.0],
    }
)
CORPUS_PARTITIONS = {
    "synthetic_trackman": CORPUS_ROWS.iloc[:2],
    "synthetic_mevo": CORPUS_ROWS.iloc[2:],
}


def _default_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sources": {
            name: {"rows": len(group)} for name, group in CORPUS_PARTITIONS.items()
        },
        "total_rows": len(CORPUS_ROWS),
    }


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _build_checkout(
    root: Path,
    *,
    manifest: dict[str, Any] | None = None,
    write_manifest: bool = True,
) -> Path:
    """Materialise a hive-partitioned corpus both loaders can read."""
    dataset = root / CORPUS_RELATIVE_PATH
    for source_id, group in CORPUS_PARTITIONS.items():
        partition = dataset / f"source_id={source_id}"
        partition.mkdir(parents=True, exist_ok=True)
        group.to_parquet(partition / "part-0.parquet", index=False)
    if write_manifest:
        (dataset / "_MANIFEST.json").write_bytes(
            _manifest_bytes(manifest if manifest is not None else _default_manifest())
        )
    return root


@pytest.fixture(scope="module")
def checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid two-source private-corpus checkout."""
    return _build_checkout(tmp_path_factory.mktemp("adr0046_corpus") / "checkout")


@pytest.fixture(scope="module")
def ud_frame(checkout: Path) -> pd.DataFrame:
    return ud_load_private_corpus(checkout)


@pytest.fixture(scope="module")
def tools_corpus(checkout: Path):
    return tools_load_private_corpus(checkout)


def test_both_stacks_resolve_the_same_physical_path(checkout: Path) -> None:
    """AGREE: one env var, one relative path, one resolved directory."""
    assert PRIVATE_DATA_ENV == "LAUNCH_MONITOR_DATA_ROOT"
    assert Path("data/authority/database/shot_corpus_parquet") == (CORPUS_RELATIVE_PATH)
    ud_path = corpus_dataset_path(checkout)
    tools_path = resolve_private_corpus_path(checkout)
    assert ud_path == tools_path
    assert ud_path == (checkout / CORPUS_RELATIVE_PATH).resolve()


def test_both_stacks_read_the_same_rows(ud_frame, tools_corpus) -> None:
    """AGREE: same partition set, same row count, same column count."""
    assert len(ud_frame) == EXPECTED_ROW_COUNT
    assert len(tools_corpus.frame) == EXPECTED_ROW_COUNT
    assert len(ud_frame.columns) == EXPECTED_COLUMN_COUNT
    assert len(tools_corpus.frame.columns) == EXPECTED_COLUMN_COUNT
    assert tools_corpus.source_count == EXPECTED_SOURCE_COUNT
    # UD renames the corpus ``source_id`` to ``session_id``; the values match.
    assert set(ud_frame["session_id"]) == set(tools_corpus.frame["source_id"])
    assert set(ud_frame["session_id"]) == set(CORPUS_PARTITIONS)


def test_both_stacks_fail_closed_without_a_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGREE: no explicit root and no environment variable means no data."""
    monkeypatch.delenv(PRIVATE_DATA_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="authority is unavailable"):
        ud_load_private_corpus()
    with pytest.raises(ValueError, match="Select the private authority root"):
        tools_load_private_corpus()


def test_both_stacks_honour_the_environment_variable(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGREE: the same env var selects the corpus for both loaders."""
    monkeypatch.setenv(PRIVATE_DATA_ENV, str(checkout))
    assert len(ud_load_private_corpus()) == EXPECTED_ROW_COUNT
    assert len(tools_load_private_corpus().frame) == EXPECTED_ROW_COUNT


def test_divergence_d28_output_schemas_are_almost_disjoint(
    ud_frame, tools_corpus
) -> None:
    """DIFFER (D28): 2 of 15 column names are shared; 13 are exclusive each."""
    ud_columns = set(ud_frame.columns)
    tools_columns = set(tools_corpus.frame.columns)

    assert ud_columns & tools_columns == SHARED_COLUMNS
    assert ud_columns - tools_columns == UD_ONLY_COLUMNS
    assert tools_columns - ud_columns == TOOLS_ONLY_COLUMNS
    assert len(UD_ONLY_COLUMNS) == len(TOOLS_ONLY_COLUMNS) == 13

    # UD derives a stable 20-hex-character shot identity Tools does not have.
    assert ud_frame["shot_id"].nunique() == EXPECTED_ROW_COUNT
    assert all(len(value) == 20 for value in ud_frame["shot_id"])
    assert set(ud_frame["observation_kind"]) == {"shot"}


def test_divergence_d29_only_ud_canonicalises_units(ud_frame, tools_corpus) -> None:
    """DIFFER (D29): the ADR-0031 conversion happens on one side only."""
    ud_sorted = ud_frame.sort_values(["session_id", "source_row"]).reset_index(
        drop=True
    )
    tools_sorted = tools_corpus.frame.sort_values(
        ["source_id", "row_index"]
    ).reset_index(drop=True)

    checked = 0
    for native, (canonical, unit) in sorted(CORPUS_COLUMN_MAP.items()):
        if native not in CORPUS_ROWS.columns:
            continue
        native_value = float(tools_sorted.loc[0, native])
        canonical_value = float(ud_sorted.loc[0, canonical])
        assert canonical_value == pytest.approx(
            native_value * CANONICAL_UNIT_FACTORS[unit], rel=1e-12
        )
        # The factor is 1.0 exactly when the native unit is already canonical.
        assert (CANONICAL_UNIT_FACTORS[unit] == 1.0) == (
            METRICS[canonical].canonical_unit == unit
        )
        checked += 1
    assert checked == 9

    # Tools keeps the native imperial values verbatim.
    assert float(tools_sorted.loc[0, "carry_yd"]) == 165.0
    assert float(ud_sorted.loc[0, "carry_distance"]) == pytest.approx(
        150.876, rel=1e-12
    )

    # ``apex_native`` has no safe conversion, so UD drops it and Tools keeps it.
    assert "apex_native" not in ud_frame.columns
    assert "apex_native" in tools_corpus.frame.columns


@pytest.mark.parametrize(
    ("label", "manifest", "write_manifest", "match"),
    [
        ("missing_manifest", None, False, "manifest not found"),
        (
            "unsupported_schema",
            {"schema_version": 2, "sources": {}, "total_rows": EXPECTED_ROW_COUNT},
            True,
            "manifest schema is unsupported",
        ),
        (
            "row_cap_exceeded",
            {
                "schema_version": 1,
                "sources": dict.fromkeys(CORPUS_PARTITIONS, {}),
                "total_rows": EXPECTED_DESKTOP_ROW_CAP + 1,
            },
            True,
            "outside the desktop retained-",
        ),
        (
            "row_count_mismatch",
            {
                "schema_version": 1,
                "sources": dict.fromkeys(CORPUS_PARTITIONS, {}),
                "total_rows": EXPECTED_ROW_COUNT - 1,
            },
            True,
            "row count mismatch",
        ),
        (
            "source_set_mismatch",
            {
                "schema_version": 1,
                "sources": {"other": {}},
                "total_rows": EXPECTED_ROW_COUNT,
            },
            True,
            "source IDs do not match",
        ),
    ],
)
def test_divergence_d30_manifest_validation_exists_only_in_tools(
    tmp_path: Path,
    label: str,
    manifest: dict[str, Any] | None,
    write_manifest: bool,
    match: str,
) -> None:
    """DIFFER (D30): Tools refuses five corpora UD loads without comment."""
    root = _build_checkout(
        tmp_path / label, manifest=manifest, write_manifest=write_manifest
    )

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        tools_load_private_corpus(root)

    accepted = ud_load_private_corpus(root)
    assert len(accepted) == EXPECTED_ROW_COUNT


def test_divergence_d30_only_tools_reports_the_manifest_digest(
    checkout: Path, tools_corpus
) -> None:
    """DIFFER (D30): a content-addressed corpus identity, on one side only."""
    assert MAX_RETAINED_ROWS == EXPECTED_DESKTOP_ROW_CAP
    manifest_path = checkout / CORPUS_RELATIVE_PATH / "_MANIFEST.json"
    expected = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert tools_corpus.manifest_sha256 == expected
    assert len(tools_corpus.manifest_sha256) == 64
    assert tools_corpus.source_name == (
        f"Private Corpus ({EXPECTED_SOURCE_COUNT} sources; manifest {expected[:12]}...)"
    )
    # The digest is over the manifest bytes, not the data; the corpus rows are
    # never hashed by either loader.
    assert expected == hashlib.sha256(_manifest_bytes(_default_manifest())).hexdigest()


def test_divergence_d31_selection_pushdown_exists_only_in_ud(
    checkout: Path,
) -> None:
    """DIFFER (D31): a source and metric allowlist, on one side only."""
    pruned = ud_load_private_corpus(
        checkout, sources=["synthetic_mevo"], metrics=["carry_distance"]
    )
    assert len(pruned) == 2
    assert set(pruned["session_id"]) == {"synthetic_mevo"}
    assert "carry_distance" in pruned.columns
    assert "ball_speed" not in pruned.columns

    with pytest.raises(ValueError, match="Unknown corpus sources requested"):
        ud_load_private_corpus(checkout, sources=["not_a_source"])
    with pytest.raises(ValueError, match="Unknown corpus metrics requested"):
        ud_load_private_corpus(checkout, metrics=["not_a_metric"])

    # Tools' signature carries a root only: no selection is expressible.
    assert tools_load_private_corpus.__code__.co_varnames[
        : tools_load_private_corpus.__code__.co_argcount
    ] == ("root",)
    assert len(tools_load_private_corpus(checkout).frame) == EXPECTED_ROW_COUNT
