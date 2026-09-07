"""Feature Parity Registry Loader — single source of truth for feature parity.

This module loads the machine-readable feature-parity registry
(``feature_parity.json``) that tracks every user-facing feature across the
canonical PyQt6 desktop app and the Tauri/React web app (PyQt6 is the model;
see epic #7462 and issue #7445).

Design by Contract:
    Preconditions:
        - Registry file must exist at the expected path
        - Registry must be valid JSON conforming to the schema
        - Every ``gap`` entry must carry a positive issue number
        - Every ``exempt`` entry must carry a non-empty reason
    Postconditions:
        - All returned entries have valid, non-empty feature ids and statuses
        - Feature ids are unique
    Invariants:
        - Registry is immutable after loading (frozen dataclasses)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

CONFIG_DIR = Path(__file__).parent
REGISTRY_PATH = CONFIG_DIR / "feature_parity.json"

# ``api_only``: the feature is an API/WebSocket endpoint with no dedicated
# surface in either shell; both shells are thin consumers (issue #8861). It
# is neither achieved parity nor a gap, and requires a ``reason``.
VALID_STATUSES = frozenset({"parity", "gap", "exempt", "api_only"})


def _is_valid_issue(value: Any) -> bool:
    """True when ``value`` is a positive-integer GitHub issue number.

    ``bool`` is a subclass of ``int`` in Python, so ``True``/``False`` would
    otherwise sneak through an ``isinstance(value, int)`` check; reject them
    explicitly.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class FeatureParityEntry:
    """A single feature-parity registry entry.

    Attributes:
        feature_id: Unique dotted identifier (e.g. ``analysis.static_plots``)
        title: Human-readable feature name
        status: One of ``parity``, ``gap``, ``exempt``
        pyqt: Repo-relative path to the PyQt6 implementation (or None)
        api: Repo-relative path to the API implementation (or None)
        web: Repo-relative path to the web implementation (or None)
        issue: Open GitHub issue number (required when status is ``gap``)
        reason: Exemption rationale (required when status is ``exempt``)
        pending_decision: True when the exemption awaits the #7460 decision
        tiles: Launcher-manifest tile ids covered by this feature
        notes: Optional free-form clarification
    """

    feature_id: str
    title: str
    status: str
    pyqt: str | None = None
    api: str | None = None
    web: str | None = None
    issue: int | None = None
    reason: str | None = None
    pending_decision: bool = False
    tiles: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, feature_id: str, data: dict[str, Any]) -> FeatureParityEntry:
        """Create an entry from a registry dict, validating the contract.

        Args:
            feature_id: The registry key for this entry
            data: Dictionary with entry properties from the registry

        Returns:
            FeatureParityEntry instance

        Raises:
            TypeError: If a field has the wrong type
            ValueError: If required fields are missing or invalid
        """
        # DbC preconditions
        if not feature_id or not feature_id.strip():
            raise ValueError("Feature id must be a non-empty string")
        if not isinstance(data, dict):
            raise TypeError(f"Entry '{feature_id}' must be an object")

        status = data.get("status")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Entry '{feature_id}' has invalid status {status!r}; "
                f"must be one of {sorted(VALID_STATUSES)}"
            )

        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"Entry '{feature_id}' must define a non-empty title")

        issue = data.get("issue")
        if status == "gap":
            if not _is_valid_issue(issue):
                raise ValueError(
                    f"Gap entry '{feature_id}' requires a positive integer "
                    f"'issue' number, got {issue!r}"
                )
        elif issue is not None and not _is_valid_issue(issue):
            raise ValueError(f"Entry '{feature_id}' has invalid issue number {issue!r}")

        reason = data.get("reason")
        if status in ("exempt", "api_only") and (
            not isinstance(reason, str) or not reason.strip()
        ):
            raise ValueError(
                f"{status.capitalize()} entry '{feature_id}' requires a non-empty 'reason'"
            )

        for path_field in ("pyqt", "api", "web"):
            value = data.get(path_field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"Entry '{feature_id}' field '{path_field}' must be a "
                    f"non-empty string or null, got {value!r}"
                )

        tiles_raw = data.get("tiles", [])
        if not isinstance(tiles_raw, list) or not all(
            isinstance(t, str) and t.strip() for t in tiles_raw
        ):
            raise ValueError(
                f"Entry '{feature_id}' field 'tiles' must be a list of "
                f"non-empty strings"
            )

        # pending_decision is exemption-scoped: it flags an exempt entry whose
        # exemption awaits the #7460 decision. A truthy value on any other
        # status is a registry authoring error.
        pending_decision = data.get("pending_decision", False)
        if pending_decision and status != "exempt":
            raise ValueError(
                f"Entry '{feature_id}' sets 'pending_decision' but status is "
                f"{status!r}; pending_decision is only valid for 'exempt' entries"
            )

        notes = data.get("notes", "")
        if not isinstance(notes, str):
            raise ValueError(
                f"Entry '{feature_id}' field 'notes' must be a string, got {notes!r}"
            )

        return cls(
            feature_id=feature_id,
            title=title.strip(),
            status=status,
            pyqt=data.get("pyqt"),
            api=data.get("api"),
            web=data.get("web"),
            issue=issue,
            reason=reason.strip() if isinstance(reason, str) else None,
            pending_decision=bool(pending_decision),
            tiles=tuple(tiles_raw),
            notes=notes,
        )

    @property
    def referenced_paths(self) -> tuple[str, ...]:
        """All non-null file paths referenced by this entry."""
        return tuple(p for p in (self.pyqt, self.api, self.web) if p)


@dataclass(frozen=True)
class FeatureParityRegistry:
    """The complete feature-parity registry.

    Invariant: entry feature ids are unique; entries are sorted by id.
    """

    version: str
    description: str
    entries: tuple[FeatureParityEntry, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, path: Path | None = None) -> FeatureParityRegistry:
        """Load the feature-parity registry from disk.

        Args:
            path: Optional override path. Defaults to REGISTRY_PATH.

        Returns:
            Loaded FeatureParityRegistry

        Raises:
            FileNotFoundError: If the registry file doesn't exist
            ValueError: If the registry format or any entry is invalid
        """
        registry_path = path or REGISTRY_PATH

        # DbC precondition
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Feature parity registry not found: {registry_path}"
            )

        with open(registry_path, encoding="utf-8") as f:
            raw = json.load(f)

        features = raw.get("features")
        if not isinstance(features, dict):
            raise ValueError("Registry missing 'features' object")

        entries = tuple(
            sorted(
                (
                    FeatureParityEntry.from_dict(feature_id, entry)
                    for feature_id, entry in features.items()
                ),
                key=lambda e: e.feature_id,
            )
        )

        registry = cls(
            version=raw.get("version", "0.0.0"),
            description=raw.get("description", ""),
            entries=entries,
        )

        # DbC postcondition: unique feature ids (dict keys guarantee this,
        # but assert the invariant explicitly for future format changes)
        ids = [e.feature_id for e in entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate feature ids in registry")

        # DbC postcondition: every launcher tile is claimed by at most one
        # entry. Without this check a tile listed by two entries would
        # silently collapse in the coverage matrix (last-write-wins),
        # hiding the duplicate claim from reviewers.
        tile_owners: dict[str, list[str]] = {}
        for entry in entries:
            for tile in entry.tiles:
                tile_owners.setdefault(tile, []).append(entry.feature_id)
        duplicate_tiles = {
            tile: owners for tile, owners in tile_owners.items() if len(owners) > 1
        }
        if duplicate_tiles:
            details = "; ".join(
                f"{tile!r} claimed by {sorted(owners)}"
                for tile, owners in sorted(duplicate_tiles.items())
            )
            raise ValueError(f"Duplicate launcher tile ids in registry: {details}")

        logger.info(
            "Loaded %d feature parity entries (v%s)", len(entries), registry.version
        )
        return registry

    def get(self, feature_id: str) -> FeatureParityEntry | None:
        """Get an entry by its feature id.

        Args:
            feature_id: The feature identifier

        Returns:
            FeatureParityEntry if found, None otherwise
        """
        if not feature_id:
            raise ValueError("feature_id must be provided")
        for entry in self.entries:
            if entry.feature_id == feature_id:
                return entry
        return None

    def by_status(self, status: str) -> list[FeatureParityEntry]:
        """Get all entries with the given status.

        Args:
            status: One of ``parity``, ``gap``, ``exempt``

        Returns:
            List of matching entries, sorted by feature id
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown parity status: {status}")
        return [e for e in self.entries if e.status == status]

    @property
    def gaps(self) -> list[FeatureParityEntry]:
        """All entries with status ``gap``."""
        return self.by_status("gap")

    @property
    def exemptions(self) -> list[FeatureParityEntry]:
        """All entries with status ``exempt``."""
        return self.by_status("exempt")

    @property
    def covered_tile_ids(self) -> frozenset[str]:
        """All launcher-manifest tile ids covered by some entry."""
        return frozenset(tile for entry in self.entries for tile in entry.tiles)
