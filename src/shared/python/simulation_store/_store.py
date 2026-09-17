"""SimulationDataStore — JSON-backed CRUD for simulation run data.

Each run is persisted as a single JSON file under::

    platformdirs.user_data_dir("upstream-drift") / "simulations" / "<run_id>.json"

Design-by-Contract invariants
------------------------------
- ``run_id`` must be a non-empty string containing only alphanumerics, hyphens,
  and underscores (prevents path-traversal attacks).
- ``save_run`` postcondition: the file exists after saving.
- ``load_run`` postcondition: returned value is a ``dict``.

Law of Demeter
--------------
All filesystem interaction is delegated to ``_RunFile``, a private helper that
owns the path derivation logic.  ``SimulationDataStore`` never reaches through
more than one layer.

Implements part of Epic #5396.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import platformdirs

from src.shared.python.contracts import ensure, require
from src.shared.python.simulation_store.replay_bundle import load_simscape_bundle


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APP_NAME = "upstream-drift"
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_ID_ERROR = (
    "run_id must be a non-empty string of alphanumerics, hyphens, or underscores "
    "(max 256 chars)"
)


def _validate_run_id(run_id: str) -> None:
    """Raise ``ValueError`` if *run_id* violates the naming contract."""
    require(
        isinstance(run_id, str) and bool(_ID_PATTERN.match(run_id)),
        _ID_ERROR,
        run_id,
    )


# ---------------------------------------------------------------------------
# Private helper — Law of Demeter boundary
# ---------------------------------------------------------------------------


class _RunFile:
    """Encapsulates path logic for a single simulation run file."""

    def __init__(self, base_dir: Path, run_id: str) -> None:
        self._path = base_dir / f"{run_id}.json"

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def write(self, data: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read(self) -> dict[str, Any]:
        text = self._path.read_text(encoding="utf-8")
        parsed: dict[str, Any] = json.loads(text)
        return parsed

    def delete(self) -> None:
        self._path.unlink()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SimulationDataStore:
    """Persistent key-value store for simulation run data.

    Args:
        base_dir: Override the default data directory (useful for testing).
            Defaults to ``platformdirs.user_data_dir("upstream-drift") / "simulations"``.

    Examples::

        store = SimulationDataStore()
        store.save_run("run_001", {"engine": "drake", "score": 0.95})
        data = store.load_run("run_001")
        runs = store.list_runs()  # ["run_001"]
        store.delete_run("run_001")

    """

    def __init__(self, base_dir: Path | None = None) -> None:
        if base_dir is None:
            self._base_dir = Path(platformdirs.user_data_dir(_APP_NAME)) / "simulations"
        else:
            self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "simulation_store_initialized base_dir=%s",
            self._base_dir,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_run(self, run_id: str, data: dict[str, Any]) -> None:
        """Persist *data* under *run_id*.

        Preconditions:
            - ``run_id`` is a non-empty alphanumeric/hyphen/underscore string.
            - ``data`` is a ``dict``.

        Postcondition:
            The backing file exists after the call.

        Args:
            run_id: Unique identifier for this simulation run.
            data: Arbitrary JSON-serialisable mapping.
        """
        _validate_run_id(run_id)
        require(isinstance(data, dict), "data must be a dict", data)

        run_file = _RunFile(self._base_dir, run_id)
        run_file.write(data)

        ensure(run_file.exists(), "save_run postcondition: backing file must exist")
        logger.info("simulation_run_saved run_id=%s", run_id)

    def load_run(self, run_id: str) -> dict[str, Any]:
        """Load and return the data stored under *run_id*.

        Preconditions:
            - ``run_id`` is a non-empty alphanumeric/hyphen/underscore string.

        Postcondition:
            Returns a ``dict``.

        Raises:
            ValueError: If *run_id* violates the naming contract.
            KeyError: If no run with *run_id* exists.

        Args:
            run_id: Identifier previously passed to ``save_run``.
        """
        _validate_run_id(run_id)

        run_file = _RunFile(self._base_dir, run_id)
        if not run_file.exists():
            raise KeyError(f"No simulation run found with run_id={run_id!r}")

        result = run_file.read()
        ensure(isinstance(result, dict), "load_run postcondition: must return dict")
        logger.debug("simulation_run_loaded run_id=%s", run_id)
        return result

    def list_runs(self) -> list[str]:
        """Return a sorted list of all stored run IDs.

        Postcondition:
            Returns a ``list[str]``.
        """
        run_ids = sorted(p.stem for p in self._base_dir.glob("*.json") if p.is_file())
        ensure(isinstance(run_ids, list), "list_runs postcondition: must return list")
        return run_ids

    def delete_run(self, run_id: str) -> None:
        """Remove the run identified by *run_id*.

        Preconditions:
            - ``run_id`` is a non-empty alphanumeric/hyphen/underscore string.

        Postcondition:
            The backing file no longer exists.

        Raises:
            ValueError: If *run_id* violates the naming contract.
            KeyError: If no run with *run_id* exists.
        """
        _validate_run_id(run_id)

        run_file = _RunFile(self._base_dir, run_id)
        if not run_file.exists():
            raise KeyError(f"No simulation run found with run_id={run_id!r}")

        run_file.delete()
        ensure(
            not run_file.exists(),
            "delete_run postcondition: backing file must not exist",
        )
        logger.info("simulation_run_deleted run_id=%s", run_id)

    def run_exists(self, run_id: str) -> bool:
        """Return ``True`` if *run_id* is present in the store.

        Preconditions:
            - ``run_id`` is a non-empty alphanumeric/hyphen/underscore string.
        """
        _validate_run_id(run_id)
        return _RunFile(self._base_dir, run_id).exists()

    # ------------------------------------------------------------------
    # Replay Bundle Catalog
    # ------------------------------------------------------------------

    def register_replay_bundle(self, manifest_path: Path | str) -> dict[str, Any]:
        """Verify and register a saved simulation replay manifest in the store.

        Preconditions:
            - ``manifest_path`` must exist and point to a readable replay manifest.

        Postconditions:
            - Returns a catalog entry dict with provenance and status.
            - Backing entry is saved in the store under ``bundle.run_id``.
        """
        p = Path(manifest_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Replay manifest not found: {p}")

        bundle = load_simscape_bundle(p)
        time_s = bundle.arrays["time_s"]
        tau_valid = bundle.arrays["tau_valid"]
        has_torque = bool(tau_valid.any())
        if tau_valid.all():
            tau_status = "available"
        elif tau_valid.any():
            tau_status = "partly_available"
        else:
            tau_status = "unavailable"

        report_path_str = ""
        if "report" in bundle.artifact_paths:
            report_path_str = str(bundle.artifact_paths["report"])

        # Check for cylinder animation GIF nearby
        animation_path_str = ""
        possible_anims = [
            p.parent.parent
            / "visuals_returned102"
            / f"{bundle.run_id.replace('-', '_')}_cylinders.gif",
            p.parent.parent
            / "visuals_returned102"
            / "simscape_returned102_cylinders.gif",
            p.parent / f"{bundle.run_id}_cylinders.gif",
        ]
        for anim in possible_anims:
            if anim.is_file():
                animation_path_str = str(anim.resolve())
                break

        entry: dict[str, Any] = {
            "run_id": bundle.run_id,
            "engine": "simscape",
            "status": bundle.status,
            "duration_s": float(time_s[-1]),
            "n_samples": int(len(time_s)),
            "manifest_path": str(p),
            "report_path": report_path_str,
            "animation_path": animation_path_str,
            "tau_status": tau_status,
            "has_torque": has_torque,
            "gates": dict(bundle.report.get("gates", {})),
            "metrics": dict(bundle.report.get("metrics", {})),
            "is_replay_catalog": True,
        }

        self.save_run(bundle.run_id, entry)
        logger.info(
            "replay_bundle_registered run_id=%s status=%s tau_status=%s",
            bundle.run_id,
            bundle.status,
            tau_status,
        )
        return entry

    def list_catalog_entries(self) -> list[dict[str, Any]]:
        """Return all catalog entries stored in the database.

        Postcondition:
            Returns a list of dicts for registered replay bundles.
        """
        entries: list[dict[str, Any]] = []
        for run_id in self.list_runs():
            try:
                run_data = self.load_run(run_id)
                if (
                    isinstance(run_data, dict)
                    and run_data.get("is_replay_catalog") is True
                ):
                    entries.append(run_data)
            except Exception:
                logger.debug("Skipping unreadable or invalid run file: %s", run_id)
        return entries

    def get_catalog_entry(self, run_id: str) -> dict[str, Any]:
        """Return the catalog entry for *run_id*.

        Raises:
            KeyError: If run_id is not found or is not a catalog entry.
        """
        run_data = self.load_run(run_id)
        if not (
            isinstance(run_data, dict) and run_data.get("is_replay_catalog") is True
        ):
            raise KeyError(f"Run {run_id!r} is not a registered replay catalog entry")
        return run_data

    @staticmethod
    def discover_known_manifests(
        search_dirs: Sequence[Path] | None = None,
    ) -> list[Path]:
        """Find verified replay manifests in repository evidence directories."""
        if search_dirs is None:
            repo_root = Path(__file__).resolve().parents[4]
            search_dirs = [
                repo_root / "docs/development/simscape_tour_matching/native_evidence",
            ]
        discovered: list[Path] = []
        for d in search_dirs:
            if d.is_dir():
                discovered.extend(sorted(d.glob("*.replay.json")))
        return discovered
