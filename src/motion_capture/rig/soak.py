"""Soak test summary aggregation and exit code calculation for multi-take recordings (#9613).

A soak session records N back-to-back takes to assess hardware reliability and bus stability.
This module defines the machine-readable summary schema and pure helpers to aggregate take results
and compute the overall session exit code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import require

from .bundle import RECORDINGS_FILE, RecordingsIndex
from .session import CaptureOutcome, SessionManifest

SOAK_SUMMARY_SCHEMA = "upstreamdrift.rig.soak_summary"
SOAK_SUMMARY_VERSION = 1
SOAK_SUMMARY_FILE = "soak_summary.json"

EXIT_BY_OUTCOME: dict[CaptureOutcome, int] = {
    CaptureOutcome.SUPPORTED: 0,
    CaptureOutcome.DEGRADED: 1,
    CaptureOutcome.BLOCKED: 1,
    CaptureOutcome.UNAVAILABLE: 2,
}


_RECORDING_FIELDS = ("view", "identity", "returncode", "frames", "duration_s", "bytes")


def _recording_dict(entry: Any) -> dict[str, Any]:
    """Project a recording (mapping or object) onto exactly the soak-summary fields."""
    if isinstance(entry, dict):
        return {field: entry.get(field) for field in _RECORDING_FIELDS}
    return {field: getattr(entry, field, None) for field in _RECORDING_FIELDS}


@dataclass(frozen=True)
class TakeSummary:
    """Summary of one take in a soak session.

    Invariants:
    - ``take`` is 1-based index (>= 1)
    - ``dir`` is non-empty directory name
    - ``outcome`` is non-empty string outcome matching :class:`CaptureOutcome`
    - ``recordings`` is a tuple of recording dictionaries
    """

    take: int
    dir: str
    outcome: str
    recordings: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        require(self.take >= 1, "take must be >= 1", self.take)
        require(bool(self.dir), "dir must not be empty", self.dir)
        require(bool(self.outcome), "outcome must not be empty", self.outcome)
        normalized = tuple(_recording_dict(r) for r in self.recordings)
        object.__setattr__(self, "recordings", normalized)

    @classmethod
    def from_bundle(
        cls,
        take: int,
        out_dir: Path,
        manifest: SessionManifest,
    ) -> TakeSummary:
        """Construct a :class:`TakeSummary` from a completed take bundle directory.

        Preconditions:
        - ``take`` >= 1
        - ``out_dir`` exists and contains ``recordings.json``
        - ``manifest`` is the session manifest for this take
        """
        require(take >= 1, "take must be >= 1", take)
        require(out_dir.is_dir(), "out_dir must be a directory", str(out_dir))
        index_path = out_dir / RECORDINGS_FILE
        require(
            index_path.is_file(),
            "recordings file must exist in bundle",
            str(index_path),
        )
        index = RecordingsIndex.model_validate_json(
            index_path.read_text(encoding="utf-8")
        )
        recordings = tuple(_recording_dict(e) for e in index.recordings)
        return cls(
            take=take,
            dir=out_dir.name,
            outcome=manifest.outcome.value,
            recordings=recordings,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert take summary to dictionary matching the soak summary schema."""
        return {
            "take": self.take,
            "dir": self.dir,
            "outcome": self.outcome,
            "recordings": list(self.recordings),
        }


def soak_exit_code(takes: Sequence[TakeSummary]) -> int:
    """Determine the soak run exit code from the worst take's outcome.

    Exit code mapping rule:
    Exit codes map from take outcomes via :data:`EXIT_BY_OUTCOME`:
    - ``CaptureOutcome.SUPPORTED`` ("supported") -> 0
    - ``CaptureOutcome.DEGRADED`` ("degraded") -> 1
    - ``CaptureOutcome.BLOCKED`` ("blocked") -> 1
    - ``CaptureOutcome.UNAVAILABLE`` ("unavailable") -> 2

    The soak exit code is the worst (maximum numeric) exit code among all takes.
    - If all takes are ``supported``, exit code is 0.
    - If any take is ``degraded`` or ``blocked`` (and none ``unavailable``), exit code is 1.
    - If any take is ``unavailable``, exit code is 2.
    Precondition: at least one take was recorded (an empty soak is not a success).
    """
    require(bool(takes), "soak_exit_code needs at least one take", len(takes))
    return max(EXIT_BY_OUTCOME[CaptureOutcome(t.outcome)] for t in takes)


def build_soak_summary(
    takes: Sequence[TakeSummary],
    repeat: int | None = None,
) -> dict[str, Any]:
    """Build a machine-readable soak summary dictionary.

    Schema:
    {
      "schema": "upstreamdrift.rig.soak_summary",
      "version": 1,
      "repeat": N,
      "takes": [
        {
          "take": k,
          "dir": "take_01",
          "outcome": "<manifest.outcome.value>",
          "recordings": [{"view", "identity", "returncode", "frames", "duration_s", "bytes"}, ...]
        },
        ...
      ],
      "ok": <bool: every take has the fully successful outcome>
    }

    A take with any 0-frame recording is never ok (existing outcome logic marks it
    ``blocked`` or ``unavailable``; in addition, any 0-frame recording forces ``ok=False``).

    Precondition: ``repeat`` (or len(takes) if omitted) must be >= 1.
    Postcondition: returns schema-compliant soak summary dictionary.
    """
    total_repeat = len(takes) if repeat is None else repeat
    require(total_repeat >= 1, "repeat must be >= 1", total_repeat)
    require(
        len(takes) <= total_repeat,
        f"number of takes ({len(takes)}) cannot exceed repeat count ({total_repeat})",
    )

    is_ok = bool(takes) and all(
        t.outcome == CaptureOutcome.SUPPORTED.value
        and all(r["frames"] != 0 for r in t.recordings)
        for t in takes
    )

    return {
        "schema": SOAK_SUMMARY_SCHEMA,
        "version": SOAK_SUMMARY_VERSION,
        "repeat": total_repeat,
        "takes": [t.to_dict() for t in takes],
        "ok": is_ok,
    }
