"""Data model and filtering engine for the Matched Swing Browser (MS-80, #10353).

Provides the lineage spine and search indexing over ``reports/matched_swing_ledger.json``,
reusing :class:`~src.shared.python.workspace.results_browser.ResultFilter`
to resolve folded-in issue #8824 and establish the contract consumed by #10521 (ORG-13).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.ledger import (
    default_ledger_path,
    find_repo_root,
)
from src.shared.python.motion_matching.ledger_schema import (
    Ledger,
    LedgerRow,
)
from src.shared.python.workspace.results_browser import ResultFilter

logger = get_logger(__name__)

__all__ = [
    "MatchedSwingBrowserModel",
    "MatchedSwingFilter",
]


@dataclass(frozen=True)
class MatchedSwingFilter:
    """Filter criteria for matched swing ledger views."""

    engine: str | None = None
    capture: str | None = None
    lane: str | None = None
    verdict: str | None = None
    text: str | None = None

    def to_result_filter(self) -> ResultFilter:
        """Convert to canonical ResultFilter lineage structure (#8824 / #10521)."""
        return ResultFilter(
            backend=self.engine,
            text=self.text,
        )


class MatchedSwingBrowserModel:
    """Business logic and query engine for browsing matched swing receipts."""

    def __init__(self, repo_root: Path | None = None) -> None:
        self._repo_root = (repo_root or find_repo_root()).resolve()

    @property
    def repo_root(self) -> Path:
        """Root directory of the repository workspace."""
        return self._repo_root

    @precondition(
        lambda self, ledger_path=None: (
            ledger_path is None or isinstance(ledger_path, (Path, str))
        )
    )
    @postcondition(lambda result: isinstance(result, list))
    def load_ledger(self, ledger_path: Path | str | None = None) -> list[LedgerRow]:
        """Load and parse ledger rows from disk."""
        target = (
            Path(ledger_path) if ledger_path else default_ledger_path(self._repo_root)
        )
        if not target.is_absolute():
            target = (self._repo_root / target).resolve()

        if not target.is_file():
            logger.warning("Ledger file not found at %s", target)
            return []

        try:
            content = target.read_text(encoding="utf-8")
            data = json.loads(content)
            ledger = Ledger.model_validate(data)
            return ledger.rows
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("Failed to parse ledger from %s: %s", target, exc)
            raise ValueError(f"Corrupt or invalid ledger at {target}: {exc}") from exc

    @precondition(
        lambda self, rows, criteria: (
            isinstance(rows, list) and isinstance(criteria, MatchedSwingFilter)
        )
    )
    @postcondition(lambda result: isinstance(result, list))
    def filter_rows(
        self, rows: list[LedgerRow], criteria: MatchedSwingFilter
    ) -> list[LedgerRow]:
        """Filter ledger rows matching all specified criteria."""
        out: list[LedgerRow] = []
        q_text = (criteria.text or "").strip().lower()

        for row in rows:
            if not self._row_matches_criteria(row, criteria, q_text):
                continue
            out.append(row)
        return out

    def _row_matches_criteria(
        self, row: LedgerRow, criteria: MatchedSwingFilter, q_text: str
    ) -> bool:
        """Evaluate whether a single row matches the filter criteria."""
        if criteria.engine and row.engine.lower() != criteria.engine.lower():
            return False

        if criteria.capture and (row.capture or "").lower() != criteria.capture.lower():
            return False

        if criteria.lane and row.lane.lower() != criteria.lane.lower():
            return False

        if criteria.verdict:
            row_verdict = self.extract_verdict_string(row)
            if row_verdict.upper() != criteria.verdict.upper():
                return False

        if q_text:
            searchable = (
                f"{row.receipt_path} {row.engine} {row.lane} {row.capture or ''} "
                f"{row.candidate_sha or ''} {row.reason or ''}"
            ).lower()
            if q_text not in searchable:
                return False

        return True

    @staticmethod
    def extract_verdict_string(row: LedgerRow) -> str:
        """Determine human-readable acceptance verdict for a row."""
        if not row.acceptance:
            return "UNCLASSIFIED"
        status = row.acceptance.get("status")
        if status:
            return str(status).upper()
        if row.acceptance.get("is_physically_accepted"):
            return "PASSED"
        return "REJECTED"

    @staticmethod
    def format_metric(value: float | None, unit: str = "mm") -> str:
        """Format a quantitative metric cleanly with units."""
        if value is None or math.isnan(value):
            return "—"
        if unit == "mm":
            return f"{value * 1000.0:.2f} mm"
        if unit == "deg":
            return f"{math.degrees(value):.2f}°"
        return f"{value:.4f} {unit}"

    def resolve_artifact_path(self, row: LedgerRow, artifact_type: str) -> Path | None:
        """Resolve absolute path on disk to an artifact associated with a row."""
        rel_str: str | None = None
        if artifact_type == "gif":
            rel_str = row.artefacts.gif
        elif artifact_type == "npz":
            rel_str = row.artefacts.npz
        elif artifact_type == "receipt":
            rel_str = row.receipt_path
        elif artifact_type == "parity":
            receipt_file = (self._repo_root / row.receipt_path).resolve()
            candidate_parity = receipt_file.parent / "parity_vs_mujoco.json"
            if candidate_parity.is_file():
                return candidate_parity
            return None

        if not rel_str:
            return None

        target = (self._repo_root / rel_str).resolve()
        return target if target.is_file() else None

    @staticmethod
    def get_unique_engines(rows: list[LedgerRow]) -> list[str]:
        """Extract sorted list of distinct engine names present in rows."""
        return sorted({row.engine for row in rows if row.engine})

    @staticmethod
    def get_unique_captures(rows: list[LedgerRow]) -> list[str]:
        """Extract sorted list of distinct capture types present in rows."""
        return sorted({row.capture for row in rows if row.capture})

    @staticmethod
    def get_unique_lanes(rows: list[LedgerRow]) -> list[str]:
        """Extract sorted list of distinct execution lanes present in rows."""
        return sorted({row.lane for row in rows if row.lane})
