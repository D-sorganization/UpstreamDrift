"""Qt-free import-review helpers for the Launch Monitor Analytics Sessions tab.

Extracted from ``src/tools/launch_monitor_analytics/widgets.py``'s
``ImportMappingDialog`` (issue #11987) so the header-reading, auto-mapping and
``ImportOptions``-building logic that precedes ``import_session`` can be
exercised and reused without a PyQt6 dependency — by the dialog itself and by
the web API's Sessions-tab-parity routes
(``src/api/routes/launch_monitor_analytics_sessions.py``). ``ImportMappingDialog``
now delegates to these functions instead of duplicating their logic.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.tools.launch_monitor_model import (
    IDENTITY_COLUMNS,
    METRICS,
    PROFILES,
    ColumnMapping,
    ImportOptions,
)

RETAIN_ONLY = "(retain only)"
MEASUREMENT_STATUSES: tuple[str, ...] = (
    "reported",
    "measured",
    "estimated",
    "derived",
    "unknown",
)
_VALID_MULTIPLIERS: tuple[float, ...] = (1.0, -1.0)


def read_headers(path: Path) -> list[str]:
    """Return the source file's column headers without reading full rows."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=5, sep=None, engine="python")
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, nrows=5, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, nrows=5)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    return [str(column) for column in frame.columns]


def mapping_targets() -> list[str]:
    """Return the selectable per-header mapping targets, retain-only first."""
    return [RETAIN_ONLY, *IDENTITY_COLUMNS, "date", "time", *METRICS]


def auto_mappings(profile_id: str, headers: list[str]) -> dict[str, str]:
    """Return ``profile_id``'s automatic source-column-to-target mapping.

    Raises:
        ValueError: If ``profile_id`` is not a known import profile.
    """
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown import profile: {profile_id}")
    return {
        item.source_column: item.target_column
        for item in PROFILES[profile_id].mappings_for(headers)
    }


def build_import_options(
    *,
    profile_id: str,
    rows: Iterable[tuple[str, str, str, float, str]],
    session_name: str,
    default_session_name: str,
    player: str,
    monitor_model: str,
    software_version: str,
) -> ImportOptions:
    """Build :class:`ImportOptions` from reviewed per-header mapping rows.

    ``rows`` holds ``(header, target, unit_text, multiplier, status)`` tuples
    — exactly the per-row widget state ``ImportMappingDialog.import_options()``
    reads from its mapping table. A row whose ``target`` is :data:`RETAIN_ONLY`
    is skipped. A blank ``unit_text`` becomes ``None``; a blank ``session_name``
    falls back to ``default_session_name``; blank ``player``, ``monitor_model``
    and ``software_version`` become ``None``.

    Raises:
        ValueError: If ``profile_id`` is unknown, a row's ``target`` is not a
            valid mapping target, ``multiplier`` is not ``1.0`` or ``-1.0``, or
            ``status`` is not one of :data:`MEASUREMENT_STATUSES`.
    """
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown import profile: {profile_id}")
    valid_targets = set(mapping_targets())
    mappings: list[ColumnMapping] = []
    for header, target, unit_text, multiplier, status in rows:
        if target == RETAIN_ONLY:
            continue
        if target not in valid_targets:
            raise ValueError(f"Unknown target column: {target}")
        if multiplier not in _VALID_MULTIPLIERS:
            raise ValueError("multiplier must be 1.0 or -1.0")
        if status not in MEASUREMENT_STATUSES:
            raise ValueError(
                f"measurement_status must be one of {', '.join(MEASUREMENT_STATUSES)}"
            )
        unit = unit_text.strip() or None
        mappings.append(ColumnMapping(header, target, unit, multiplier, status))
    return ImportOptions(
        profile_id=profile_id,
        mappings=tuple(mappings),
        session_name=session_name.strip() or default_session_name,
        player=player.strip() or None,
        monitor_model=monitor_model.strip() or None,
        software_version=software_version.strip() or None,
    )
