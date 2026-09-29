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
from pathlib import Path
from typing import Any

import platformdirs

from src.shared.python.contracts import ensure, require

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
        print(store.list_runs())  # ["run_001"]
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
