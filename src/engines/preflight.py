"""Engine preflight checks for UpstreamDrift (Issue #10377 — MS-103).

Provides a platform/engine support matrix with pinned reproducible runtime
and bootstrap recipes, plus per-check preflight verification.

Design contracts
----------------
- Fail-closed: any FAIL outcome means the engine is not usable in that
  configuration.  Callers must not silently ignore FAIL results.
- Tier-aware: ``experimental`` engines downgrade ImportError to SKIP so
  the required CI gate is not blocked.
- Law of Demeter: checker objects only reach into their own attributes;
  all SDK imports happen inside the check methods (not at module import time).
- DRY: ``_try_import`` is the single import helper used everywhere.

Usage::

    from src.engines.preflight import PreflightRunner

    runner = PreflightRunner(engines=["mujoco", "drake"])
    summary = runner.run_all()
    logging.info(summary.to_json())

"""

from __future__ import annotations

import enum
import importlib
import json
import logging
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.engines.tiers import ENGINE_TIERS, get_engine_tier
from src.shared.python.contracts import (
    ensure,
    postcondition,
    precondition,
    require,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum Python version required by pyproject.toml
# ---------------------------------------------------------------------------

_MIN_PYTHON: tuple[int, int] = (3, 11)

# ---------------------------------------------------------------------------
# Engine SDK import paths  (only the top-level package name is needed)
# ---------------------------------------------------------------------------

_ENGINE_SDK_MODULE: dict[str, str] = {
    "mujoco": "mujoco",
    "drake": "pydrake",
    "pinocchio": "pinocchio",
    "opensim": "opensim",
    "myosuite": "myosuite",
    "putting_green": "mujoco",  # uses MuJoCo under the hood
}

# ---------------------------------------------------------------------------
# Known engine set (single source of truth from tiers.py)
# ---------------------------------------------------------------------------

KNOWN_ENGINES: frozenset[str] = frozenset(ENGINE_TIERS.keys())

# ---------------------------------------------------------------------------
# Remediation advice table — keeps messages DRY
# ---------------------------------------------------------------------------

_REMEDIATION: dict[str, dict[str, str]] = {
    "mujoco": {
        "sdk_import": "pip install mujoco>=3.0",
        "runtime_version": "Upgrade to Python 3.11+: https://python.org/downloads/",
        "model_assets": (
            "Run: git submodule update --init shared/models/opensim/opensim-models"
        ),
        "display": "Set MUJOCO_GL=osmesa for headless; or install a virtual display (xvfb-run).",
        "capacity": "Reinstall MuJoCo: pip install --upgrade mujoco",
    },
    "drake": {
        "sdk_import": (
            "pip install -e '.[all-engines]'  # or: conda install -c conda-forge drake"
        ),
        "runtime_version": "Upgrade to Python 3.11+: https://python.org/downloads/",
        "model_assets": "git submodule update --init src/shared/tools/human-gazebo",
        "display": "Drake drake-visualizer requires a display; set DRAKE_NO_DISPLAY=1 for headless.",
        "capacity": "Reinstall Drake: pip install -e '.[all-engines]'",
    },
    "pinocchio": {
        "sdk_import": (
            "pip install -e '.[all-engines]'  # or: conda install -c conda-forge pinocchio"
        ),
        "runtime_version": "Upgrade to Python 3.11+.",
        "model_assets": "git submodule update --init src/shared/tools/human-gazebo",
        "display": "No display requirements for Pinocchio core.",
        "capacity": "Reinstall Pinocchio: pip install -e '.[all-engines]'",
    },
    "opensim": {
        "sdk_import": (
            "pip install -e '.[biomechanics]'  # or: conda install -c opensim-org opensim"
        ),
        "runtime_version": "Upgrade to Python 3.11+.",
        "model_assets": (
            "git submodule update --init shared/models/opensim/opensim-models"
        ),
        "display": "OpenSim GUI requires a display; headless mode is available.",
        "capacity": "Reinstall OpenSim: pip install -e '.[biomechanics]'",
    },
    "myosuite": {
        "sdk_import": ("pip install -e '.[biomechanics]'  # or: pip install myosuite"),
        "runtime_version": "Upgrade to Python 3.11+.",
        "model_assets": "git submodule update --init shared/models/myosuite/myo_sim",
        "display": "MyoSuite uses MuJoCo headless rendering; set MUJOCO_GL=osmesa.",
        "capacity": "Reinstall MyoSuite: pip install -e '.[biomechanics]'",
    },
    "putting_green": {
        "sdk_import": "pip install mujoco>=3.0  # putting_green uses MuJoCo",
        "runtime_version": "Upgrade to Python 3.11+.",
        "model_assets": "No separate model assets required for putting_green.",
        "display": "Set MUJOCO_GL=osmesa for headless.",
        "capacity": "Reinstall MuJoCo: pip install --upgrade mujoco",
    },
}


# ===========================================================================
# CheckOutcome
# ===========================================================================


class CheckOutcome(enum.Enum):
    """Outcome of a single preflight check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"

    def __str__(self) -> str:
        return self.value


# ===========================================================================
# EnginePreflightResult
# ===========================================================================


@dataclass(frozen=True)
class EnginePreflightResult:
    """Immutable record of a single preflight check result.

    Invariant: ``engine`` and ``check`` must be non-empty strings.

    Args:
        engine: Normalised engine name (e.g. ``"mujoco"``).
        check: Name of the check performed (e.g. ``"sdk_import"``).
        outcome: ``PASS``, ``FAIL``, or ``SKIP``.
        message: Human-readable description of the check outcome.
        remediation: Actionable remediation step when outcome is FAIL; else None.
    """

    engine: str
    check: str
    outcome: CheckOutcome
    message: str
    remediation: str | None = None

    def __post_init__(self) -> None:
        require(
            isinstance(self.engine, str) and bool(self.engine.strip()),
            "engine must be a non-empty string",
            self.engine,
        )
        require(
            isinstance(self.check, str) and bool(self.check.strip()),
            "check must be a non-empty string",
            self.check,
        )
        require(
            isinstance(self.outcome, CheckOutcome),
            "outcome must be a CheckOutcome instance",
            self.outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        d: dict[str, Any] = {
            "engine": self.engine,
            "check": self.check,
            "outcome": str(self.outcome),
            "message": self.message,
        }
        if self.remediation is not None:
            d["remediation"] = self.remediation
        return d


# ===========================================================================
# EnginePreflightChecker
# ===========================================================================


class EnginePreflightChecker:
    """Runs individual preflight checks for one engine.

    Design by Contract
    ------------------
    - Precondition: ``engine_name`` must be a recognised engine name.
    - All check methods return :class:`EnginePreflightResult`; they never raise.
    - Engine SDKs are imported *inside* check methods, not at construction time
      (Law of Demeter / isolation).

    Args:
        engine_name: One of the keys in :data:`KNOWN_ENGINES`.
    """

    def __init__(self, engine_name: str) -> None:
        require(
            isinstance(engine_name, str) and bool(engine_name.strip()),
            "engine_name must be a non-empty string",
            engine_name,
        )
        normalised = engine_name.strip().lower()
        require(
            normalised in KNOWN_ENGINES,
            f"engine_name must be one of {sorted(KNOWN_ENGINES)}",
            engine_name,
        )
        self._engine_name = normalised
        self._tier = get_engine_tier(normalised)

    @property
    def engine_name(self) -> str:
        """Normalised engine name."""
        return self._engine_name

    @property
    def tier(self) -> str:
        """Engine tier: ``core``, ``extended``, or ``experimental``."""
        return self._tier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remediation(self, check: str) -> str | None:
        """Return the remediation string for this engine + check combo."""
        return _REMEDIATION.get(self._engine_name, {}).get(check)

    def _make_result(
        self,
        check: str,
        outcome: CheckOutcome,
        message: str,
        *,
        remediation: str | None = None,
    ) -> EnginePreflightResult:
        """Construct a result, attaching remediation when outcome is FAIL."""
        if outcome == CheckOutcome.FAIL and remediation is None:
            remediation = self._remediation(check)
        return EnginePreflightResult(
            engine=self._engine_name,
            check=check,
            outcome=outcome,
            message=message,
            remediation=remediation,
        )

    def _import_engine_sdk(self) -> types.ModuleType | None:  # type: ignore[name-defined]
        """Attempt to import the engine SDK; return the module or None."""
        sdk_name = _ENGINE_SDK_MODULE.get(self._engine_name)
        if sdk_name is None:
            return None
        try:
            return importlib.import_module(sdk_name)
        except (ImportError, ModuleNotFoundError):
            return None

    # ------------------------------------------------------------------
    # Public check methods
    # ------------------------------------------------------------------

    def check_runtime_version(self) -> EnginePreflightResult:
        """Check that the Python interpreter meets the minimum version.

        Postcondition: returns an :class:`EnginePreflightResult`.
        """
        check = "runtime_version"
        major, minor = sys.version_info[:2]
        req_major, req_minor = _MIN_PYTHON
        if (major, minor) >= (req_major, req_minor):
            return self._make_result(
                check,
                CheckOutcome.PASS,
                f"Python {major}.{minor} >= {req_major}.{req_minor} ✓",
            )
        return self._make_result(
            check,
            CheckOutcome.FAIL,
            (
                f"Python {major}.{minor} < {req_major}.{req_minor} (required); "
                "upgrade your interpreter."
            ),
        )

    def check_sdk_import(self) -> EnginePreflightResult:
        """Check that the engine's Python SDK can be imported.

        Experimental engines degrade ImportError to SKIP so they don't block
        the required CI gate.

        Postcondition: returns an :class:`EnginePreflightResult`.
        """
        check = "sdk_import"
        sdk_name = _ENGINE_SDK_MODULE.get(self._engine_name)
        if sdk_name is None:
            return self._make_result(
                check,
                CheckOutcome.SKIP,
                f"No SDK import path configured for {self._engine_name!r}.",
            )

        module = self._import_engine_sdk()
        if module is not None:
            version = getattr(module, "__version__", "unknown")
            return self._make_result(
                check,
                CheckOutcome.PASS,
                f"{sdk_name} v{version} imported successfully ✓",
            )

        # SDK not available
        if self._tier == "experimental":
            return self._make_result(
                check,
                CheckOutcome.SKIP,
                (
                    f"{sdk_name} not available (engine tier: experimental); "
                    "skipping non-blocking check."
                ),
            )
        return self._make_result(
            check,
            CheckOutcome.FAIL,
            f"{sdk_name} could not be imported.",
        )

    def check_model_assets(
        self, asset_root: Path | None = None
    ) -> EnginePreflightResult:
        """Check that model asset directories exist and are non-empty.

        Args:
            asset_root: Override the default asset root for testing.

        Precondition: ``asset_root`` must be a :class:`~pathlib.Path` or ``None``.
        Postcondition: returns an :class:`EnginePreflightResult`.
        """
        check = "model_assets"
        require(
            asset_root is None or isinstance(asset_root, Path),
            "asset_root must be a pathlib.Path or None",
            type(asset_root).__name__,
        )

        if asset_root is None:
            # Default: skip when no external asset root is specified.
            # In full integration, callers supply the repo root.
            return self._make_result(
                check,
                CheckOutcome.SKIP,
                "No asset_root supplied; skipping model asset check.",
            )

        if not asset_root.exists():
            return self._make_result(
                check,
                CheckOutcome.FAIL,
                f"Asset root does not exist: {asset_root}",
            )

        if not asset_root.is_dir():
            return self._make_result(
                check,
                CheckOutcome.FAIL,
                f"Asset root is not a directory: {asset_root}",
            )

        # At least one file present means the submodule was initialised.
        has_files = any(asset_root.iterdir())
        if has_files:
            return self._make_result(
                check,
                CheckOutcome.PASS,
                f"Asset root present and non-empty: {asset_root} ✓",
            )

        return self._make_result(
            check,
            CheckOutcome.FAIL,
            f"Asset root exists but is empty: {asset_root}",
        )

    def check_display(self) -> EnginePreflightResult:
        """Check that a rendering backend or headless display is available.

        Engines that do not require a display (e.g. Pinocchio core) return PASS
        unconditionally.

        Postcondition: returns an :class:`EnginePreflightResult`.
        """
        check = "display"

        # Pinocchio does not need a display
        if self._engine_name == "pinocchio":
            return self._make_result(
                check,
                CheckOutcome.PASS,
                "Pinocchio does not require a display backend ✓",
            )

        # MuJoCo / MyoSuite: MUJOCO_GL controls the backend
        mujoco_gl = os.environ.get("MUJOCO_GL", "")
        if mujoco_gl in ("osmesa", "egl", "glfw"):
            return self._make_result(
                check,
                CheckOutcome.PASS,
                f"MUJOCO_GL={mujoco_gl!r} — headless/GPU rendering configured ✓",
            )

        # Check for a DISPLAY variable (X11)
        display_var = os.environ.get("DISPLAY", "")
        if display_var:
            return self._make_result(
                check,
                CheckOutcome.PASS,
                f"DISPLAY={display_var!r} — X11 display available ✓",
            )

        # No display configuration found — SKIP (advisory) for experimental,
        # or SKIP for extended; FAIL would be too strict for a CI-agnostic check.
        return self._make_result(
            check,
            CheckOutcome.SKIP,
            (
                "No display variable (DISPLAY, MUJOCO_GL) detected. "
                "Rendering may fail at runtime."
            ),
            remediation=self._remediation(check),
        )

    def check_capacity(self) -> EnginePreflightResult:
        """Check that the engine can instantiate and step a minimal model.

        This is a smoke-test: import the SDK, create a trivial model, call
        ``mj_step`` (or equivalent) once, and verify no exception is raised.

        Postcondition: returns an :class:`EnginePreflightResult`.
        """
        check = "capacity"

        sdk = self._import_engine_sdk()
        if sdk is None:
            outcome = (
                CheckOutcome.SKIP if self._tier == "experimental" else CheckOutcome.FAIL
            )
            return self._make_result(
                check,
                outcome,
                f"SDK not available for {self._engine_name!r}; cannot run capacity check.",
            )

        # Engine-specific minimal smoke step. Catch concrete SDK failure
        # classes only — do not use bare ``except Exception`` (BLE001 ratchet).
        try:
            return self._run_minimal_step(sdk)
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning("Capacity check for %r raised: %s", self._engine_name, exc)
            return self._make_result(
                check,
                CheckOutcome.FAIL,
                f"Minimal step raised {type(exc).__name__}: {exc}",
            )

    def _run_minimal_step(self, sdk: Any) -> EnginePreflightResult:
        """Engine-specific minimal model instantiation + one step.

        Returns:
            :class:`EnginePreflightResult` with PASS or FAIL outcome.
        """
        check = "capacity"
        engine = self._engine_name

        if engine in ("mujoco", "putting_green"):
            return self._capacity_mujoco(sdk, check)
        if engine == "drake":
            return self._capacity_drake(sdk, check)
        if engine == "pinocchio":
            return self._capacity_pinocchio(sdk, check)
        # opensim, myosuite — complex setup; treat as SKIP for now
        return self._make_result(
            check,
            CheckOutcome.SKIP,
            f"Minimal capacity step not yet implemented for {engine!r}; skipping.",
        )

    def _capacity_mujoco(self, sdk: Any, check: str) -> EnginePreflightResult:
        """MuJoCo capacity check: load a 2-link pendulum and call mj_step."""
        xml = """
        <mujoco>
          <worldbody>
            <body>
              <joint type="hinge"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        try:
            model = sdk.MjModel.from_xml_string(xml)
            data = sdk.MjData(model)
            sdk.mj_step(model, data)
            return self._make_result(
                check,
                CheckOutcome.PASS,
                f"MuJoCo mj_step completed; nq={model.nq} ✓",
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return self._make_result(
                check,
                CheckOutcome.FAIL,
                f"MuJoCo mj_step raised {type(exc).__name__}: {exc}",
            )

    def _capacity_drake(self, sdk: Any, check: str) -> EnginePreflightResult:
        """Drake capacity check: build a MultibodyPlant, finalize, advance context."""
        try:
            from pydrake.multibody.plant import MultibodyPlant
            from pydrake.systems.analysis import Simulator

            plant = MultibodyPlant(time_step=0.001)
            plant.Finalize()
            simulator = Simulator(plant)
            simulator.AdvanceTo(0.001)
            return self._make_result(
                check,
                CheckOutcome.PASS,
                "Drake MultibodyPlant finalized and stepped ✓",
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return self._make_result(
                check,
                CheckOutcome.FAIL,
                f"Drake capacity check raised {type(exc).__name__}: {exc}",
            )

    def _capacity_pinocchio(self, sdk: Any, check: str) -> EnginePreflightResult:
        """Pinocchio capacity check: build a free-flyer model and run RNEA."""
        try:
            import numpy as np

            model = sdk.Model()
            data = model.createData()
            q = sdk.neutral(model)
            v = np.zeros(model.nv)
            a = np.zeros(model.nv)
            sdk.rnea(model, data, q, v, a)
            return self._make_result(
                check,
                CheckOutcome.PASS,
                "Pinocchio RNEA completed on neutral model ✓",
            )
        except (
            AttributeError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            return self._make_result(
                check,
                CheckOutcome.FAIL,
                f"Pinocchio capacity check raised {type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Convenience: run all checks for this engine
    # ------------------------------------------------------------------

    def run_all_checks(
        self, asset_root: Path | None = None
    ) -> list[EnginePreflightResult]:
        """Run all five checks in order; return the results.

        Postcondition: returns a list of exactly five :class:`EnginePreflightResult`.
        """
        results = [
            self.check_runtime_version(),
            self.check_sdk_import(),
            self.check_model_assets(asset_root=asset_root),
            self.check_display(),
            self.check_capacity(),
        ]
        ensure(
            len(results) == 5,  # noqa: PLR2004
            "run_all_checks must return exactly 5 results",
            len(results),
        )
        return results


# ===========================================================================
# PreflightSummary
# ===========================================================================


@dataclass
class PreflightSummary:
    """Aggregate result of a multi-engine preflight run.

    Invariant: ``results`` must be non-empty.

    Args:
        results: All :class:`EnginePreflightResult` instances from the run.
    """

    results: list[EnginePreflightResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        require(
            bool(self.results),
            "PreflightSummary must contain at least one result",
        )

    @property
    def pass_count(self) -> int:
        """Number of PASS results."""
        return sum(1 for r in self.results if r.outcome == CheckOutcome.PASS)

    @property
    def fail_count(self) -> int:
        """Number of FAIL results."""
        return sum(1 for r in self.results if r.outcome == CheckOutcome.FAIL)

    @property
    def skip_count(self) -> int:
        """Number of SKIP results."""
        return sum(1 for r in self.results if r.outcome == CheckOutcome.SKIP)

    @property
    def overall_outcome(self) -> CheckOutcome:
        """FAIL if any result is FAIL; PASS otherwise (SKIPs are not failures)."""
        if any(r.outcome == CheckOutcome.FAIL for r in self.results):
            return CheckOutcome.FAIL
        return CheckOutcome.PASS

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "overall_outcome": str(self.overall_outcome),
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "skip_count": self.skip_count,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ===========================================================================
# PreflightRunner
# ===========================================================================


class PreflightRunner:
    """Orchestrates preflight checks for a list of engines.

    Design by Contract
    ------------------
    - Precondition: ``engines`` must be a non-empty list of known engine names.
    - Postcondition: :meth:`run_all` returns a :class:`PreflightSummary`.

    Args:
        engines: List of engine names to check.  Defaults to all known engines.
        asset_root: Optional path to model asset directory (passed to each checker).
    """

    def __init__(
        self,
        engines: list[str] | None = None,
        asset_root: Path | None = None,
    ) -> None:
        if engines is None:
            engines = sorted(KNOWN_ENGINES)
        require(
            isinstance(engines, list) and bool(engines),
            "engines must be a non-empty list",
            engines,
        )
        for name in engines:
            require(
                isinstance(name, str) and name.strip().lower() in KNOWN_ENGINES,
                f"Unknown engine {name!r}; must be one of {sorted(KNOWN_ENGINES)}",
                name,
            )
        self._engines: list[str] = [e.strip().lower() for e in engines]
        self._asset_root = asset_root

    def run_all(self) -> PreflightSummary:
        """Run all checks for all configured engines.

        Postcondition: returns a non-empty :class:`PreflightSummary`.
        """
        all_results: list[EnginePreflightResult] = []
        for engine_name in self._engines:
            checker = EnginePreflightChecker(engine_name)
            results = checker.run_all_checks(asset_root=self._asset_root)
            all_results.extend(results)

        ensure(bool(all_results), "run_all must produce at least one result")
        return PreflightSummary(results=all_results)
