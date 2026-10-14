"""Same-input bundle: everything an engine needs to reproduce a motion (#11607).

A bundle fixes the model (spec bytes), a closure-consistent initial state,
the per-step efforts (zero-order hold at ``dt_s``) and the integration policy.
An opt-in finer effort hold (#12279) adds ``hold_efforts`` with
``effort_holds`` rows per step; ``efforts`` keeps one row per step (each
step's first hold) for the per-step consumers, and the manifest records the
hold so every engine's replay uses the same policy.
It also carries the reference engine's states so a replay can be scored
without re-running the reference.  It is a single ``.npz`` with a JSON
manifest, so it can be shipped to hosts that have only one engine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]

SCHEMA = "same-input-bundle/v1"
POLICY: dict[str, Any] = {
    "integrator": "rk4",
    "substeps": 8,
    "effort_hold": "zero_order",
    "kkt": "exact",
    "closure_projection": "closest_point_each_step",
    "projection_tolerance": 1e-13,
    "root_efforts": "zero",
}


@dataclass(frozen=True)
class InputBundle:
    """Validated same-input bundle.

    Postconditions: ``efforts`` is (steps, nv); ``reference_q`` and
    ``reference_v`` are (steps + 1, nv) and start at ``q0``/``v0``;
    ``effort_holds`` divides the policy substeps, and ``hold_efforts`` is
    (steps * effort_holds, nv) when ``effort_holds > 1``, else ``None``.
    """

    spec_bytes: bytes
    coordinate_order: tuple[str, ...]
    dt_s: float
    q0: Array
    v0: Array
    efforts: Array
    reference_q: Array
    reference_v: Array
    reference_engine: str
    provenance: dict[str, Any] = field(default_factory=dict)
    effort_holds: int = 1
    hold_efforts: Array | None = None
    effort_hold_dt_s: float | None = None

    def __post_init__(self) -> None:
        nv = len(self.coordinate_order)
        if tuple(json.loads(self.spec_bytes)["coordinate_order"]) != (
            self.coordinate_order
        ):
            raise ValueError("coordinate_order must equal the spec coordinate order")
        if not (np.isfinite(self.dt_s) and self.dt_s > 0.0):
            raise ValueError("dt_s must be positive")
        if self.q0.shape != (nv,) or self.v0.shape != (nv,):
            raise ValueError("q0 and v0 must be (nv,)")
        if self.efforts.ndim != 2 or self.efforts.shape[1] != nv:
            raise ValueError("efforts must be (steps, nv)")
        expected = (self.efforts.shape[0] + 1, nv)
        if self.reference_q.shape != expected or self.reference_v.shape != expected:
            raise ValueError("reference states must be (steps + 1, nv)")
        if not np.array_equal(self.reference_q[0], self.q0) or not np.array_equal(
            self.reference_v[0], self.v0
        ):
            raise ValueError("reference states must start at q0, v0")
        arrays = (self.q0, self.v0, self.efforts, self.reference_q, self.reference_v)
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("bundle arrays must be finite")
        self._reconcile_effort_holds()
        self._check_hold_efforts()

    def _reconcile_effort_holds(self) -> None:
        """Sync effort_hold_dt_s and effort_holds (#12279)."""
        if self.effort_hold_dt_s is not None:
            from src.shared.python.motion_matching.same_input.integrator import (
                effort_holds_for,
            )

            computed_holds = effort_holds_for(self.effort_hold_dt_s, self.dt_s)
            if self.effort_holds != 1 and self.effort_holds != computed_holds:
                raise ValueError("effort_holds must match effort_hold_dt_s")
            object.__setattr__(self, "effort_holds", computed_holds)
        else:
            object.__setattr__(self, "effort_hold_dt_s", self.dt_s / self.effort_holds)

    def _check_hold_efforts(self) -> None:
        """``hold_efforts`` is present iff ``effort_holds > 1`` and its every
        ``effort_holds``-th row equals ``efforts`` (one hold per step)."""
        from src.shared.python.motion_matching.same_input.integrator import (
            StepPolicy,
        )

        StepPolicy(substeps=int(POLICY["substeps"]), effort_holds=self.effort_holds)
        holds, table = self.effort_holds, self.hold_efforts
        if holds == 1:
            if table is not None:
                raise ValueError("hold_efforts requires effort_holds > 1")
            return
        if table is None:
            raise ValueError("effort_holds > 1 requires hold_efforts")
        if table.shape != (self.efforts.shape[0] * holds, *self.efforts.shape[1:]):
            raise ValueError("hold_efforts must be (steps * effort_holds, nv)")
        if not np.isfinite(table).all() or not np.array_equal(
            table[::holds], self.efforts
        ):
            raise ValueError("hold_efforts must start each step with efforts")

    @property
    def spec_sha256(self) -> str:
        return hashlib.sha256(self.spec_bytes).hexdigest()

    @property
    def steps(self) -> int:
        return int(self.efforts.shape[0])

    @property
    def effort_hold_s(self) -> float:
        """Duration each recorded effort is held for in a replay."""
        return self.dt_s / self.effort_holds

    @property
    def effort_substeps(self) -> int:
        """RK4 substeps per effort hold sample (#12279)."""
        substeps = int(POLICY.get("substeps", 8))
        return substeps // self.effort_holds

    @property
    def projection_dt_s(self) -> float:
        """Closure projection interval in seconds (#12279)."""
        return self.dt_s

    def effort_hold_policy(self) -> Any:
        """The EffortHoldPolicy contract for this bundle (#12279)."""
        return EffortHoldPolicy(
            effort_hold_dt_s=float(self.effort_hold_dt_s or self.effort_hold_s),
            effort_substeps=self.effort_substeps,
            projection_dt_s=self.projection_dt_s,
        )

    def replay_efforts(self, steps: int | None = None) -> Array:
        """Efforts at the hold rate for the first ``steps`` steps (default all).

        Replay these with :meth:`step_policy`; they are ``efforts`` itself for
        a one-hold bundle.
        """
        count = self.steps if steps is None else steps
        if not 0 <= count <= self.steps:
            raise ValueError(f"steps must lie in [0, {self.steps}]")
        table = self.efforts if self.hold_efforts is None else self.hold_efforts
        return table[: count * self.effort_holds]

    def step_policy(self, **overrides: Any) -> Any:
        """The ``StepPolicy`` this bundle was recorded with (hold rate and
        substeps from the manifest); ``overrides`` set ``project`` or
        ``stop_on_failure``."""
        from src.shared.python.motion_matching.same_input.integrator import (
            StepPolicy,
        )

        return StepPolicy(
            substeps=int(POLICY["substeps"]),
            effort_holds=self.effort_holds,
            **overrides,
        )

    def manifest(self) -> dict[str, Any]:
        """JSON-serialisable description stored beside the arrays."""
        return {
            "schema": SCHEMA,
            "spec_sha256": self.spec_sha256,
            "coordinate_order": list(self.coordinate_order),
            "dt_s": self.dt_s,
            "steps": self.steps,
            "duration_s": self.steps * self.dt_s,
            "policy": {
                **POLICY,
                "effort_holds_per_step": self.effort_holds,
                "effort_hold_s": self.effort_hold_s,
                "effort_hold_dt_s": float(self.effort_hold_dt_s or self.effort_hold_s),
                "effort_substeps": self.effort_substeps,
                "projection_dt_s": self.projection_dt_s,
            },
            "reference_engine": self.reference_engine,
            "provenance": self.provenance,
        }

    def ball_impact(self) -> Any:
        """The recorded, latched ball force (``ImpactForce``) or ``None``.

        Every engine's open-loop replay applies this identical force
        (GCV-20, #11767); bundles without an impact return ``None``.
        """
        record = self.provenance.get("ball_impact")
        if record is None:
            return None
        from src.shared.python.motion_matching.impact_force import ImpactForce

        return ImpactForce.from_record(record)

    def save(self, path: Path) -> None:
        """Write the bundle as one compressed ``.npz``."""
        arrays: dict[str, Any] = {
            "manifest": np.array(json.dumps(self.manifest(), sort_keys=True)),
            "spec": np.frombuffer(self.spec_bytes, dtype=np.uint8),
            "q0": self.q0,
            "v0": self.v0,
            "efforts": self.efforts,
            "reference_q": self.reference_q,
            "reference_v": self.reference_v,
        }
        if self.hold_efforts is not None:
            arrays["hold_efforts"] = self.hold_efforts
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> InputBundle:
        """Read and validate a bundle; rejects unknown schemas and bad hashes."""
        with np.load(path, allow_pickle=False) as data:
            manifest = json.loads(str(data["manifest"]))
            if manifest.get("schema") != SCHEMA:
                raise ValueError(f"unsupported bundle schema {manifest.get('schema')}")
            spec_bytes = data["spec"].tobytes()
            bundle = cls(
                spec_bytes=spec_bytes,
                coordinate_order=tuple(manifest["coordinate_order"]),
                dt_s=float(manifest["dt_s"]),
                q0=data["q0"],
                v0=data["v0"],
                efforts=data["efforts"],
                reference_q=data["reference_q"],
                reference_v=data["reference_v"],
                reference_engine=str(manifest["reference_engine"]),
                provenance=dict(manifest.get("provenance", {})),
                effort_holds=int(
                    manifest.get("policy", {}).get("effort_holds_per_step", 1)
                ),
                hold_efforts=data.get("hold_efforts"),
                effort_hold_dt_s=(
                    float(manifest["policy"]["effort_hold_dt_s"])
                    if "policy" in manifest and "effort_hold_dt_s" in manifest["policy"]
                    else None
                ),
            )
        if bundle.spec_sha256 != manifest["spec_sha256"]:
            raise ValueError("bundle spec hash does not match its manifest")
        return bundle


DEFAULT_EFFORT_HOLD_DT_S: float = 0.001
FINE_EFFORT_HOLD_DT_S: float = 0.000125
DEFAULT_PROJECTION_DT_S: float = 0.001
DEFAULT_EFFORT_SUBSTEPS: int = 8
FINE_EFFORT_SUBSTEPS: int = 1

KNOWN_7IRON_FOOT_SLIP_1MS_MM: float = 43.0
KNOWN_7IRON_FOOT_SLIP_FINE_MM: float = 3.3
FOOT_SLIP_BOUND_MM: float = 10.0


@dataclass(frozen=True)
class EffortHoldPolicy:
    """Explicit effort hold rate policy contract (#12279).

    Parameters:
        effort_hold_dt_s: Controller zero-order hold interval in seconds.
        effort_substeps: RK4 substeps integrated under each held effort sample.
        projection_dt_s: Interval between kinematic closure projections.
    """

    effort_hold_dt_s: float = DEFAULT_EFFORT_HOLD_DT_S
    effort_substeps: int = DEFAULT_EFFORT_SUBSTEPS
    projection_dt_s: float = DEFAULT_PROJECTION_DT_S

    def __post_init__(self) -> None:
        if not (np.isfinite(self.effort_hold_dt_s) and self.effort_hold_dt_s > 0.0):
            raise ValueError("effort_hold_dt_s must be positive and finite")
        if not (np.isfinite(self.projection_dt_s) and self.projection_dt_s > 0.0):
            raise ValueError("projection_dt_s must be positive and finite")
        if isinstance(self.effort_substeps, bool) or not isinstance(
            self.effort_substeps, int
        ):
            raise TypeError("effort_substeps must be an integer")
        if self.effort_substeps < 1:
            raise ValueError("effort_substeps must be positive")

    @property
    def effort_holds_per_step(self) -> int:
        """Number of effort hold periods per closure projection step."""
        ratio = self.projection_dt_s / self.effort_hold_dt_s
        return max(1, int(round(ratio)))

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return manifest policy dictionary entries."""
        return {
            "effort_hold_dt_s": self.effort_hold_dt_s,
            "effort_substeps": self.effort_substeps,
            "projection_dt_s": self.projection_dt_s,
            "effort_holds_per_step": self.effort_holds_per_step,
            "effort_hold_s": self.effort_hold_dt_s,
        }


def resolve_effort_hold_dt_s(
    effort_hold_s: float | None = None,
    effort_hold_dt_s: float | None = None,
) -> float:
    """Resolve effort_hold_dt_s from either parameter name (#12279)."""
    if effort_hold_dt_s is not None and effort_hold_s is not None:
        if abs(effort_hold_dt_s - effort_hold_s) > 1e-12:
            raise ValueError("effort_hold_dt_s and effort_hold_s must match")
        return float(effort_hold_dt_s)
    if effort_hold_dt_s is not None:
        return float(effort_hold_dt_s)
    if effort_hold_s is not None:
        return float(effort_hold_s)
    return DEFAULT_EFFORT_HOLD_DT_S


def _extract_manifest_policy(source: Any) -> tuple[dict[str, Any], float]:
    """Extract policy dict and projection dt_s from various sources."""
    if hasattr(source, "manifest") and callable(source.manifest):
        data = source.manifest()
        return data.get("policy", {}), float(data.get("dt_s", DEFAULT_PROJECTION_DT_S))
    if isinstance(source, (str, Path)):
        with np.load(source, allow_pickle=False) as bundle_data:
            manifest_json = json.loads(str(bundle_data["manifest"]))
            return manifest_json.get("policy", {}), float(
                manifest_json.get("dt_s", DEFAULT_PROJECTION_DT_S)
            )
    if isinstance(source, dict):
        if "manifest" in source and isinstance(source["manifest"], dict):
            return _extract_manifest_policy(source["manifest"])
        if "policy" in source and isinstance(source["policy"], dict):
            return source["policy"], float(source.get("dt_s", DEFAULT_PROJECTION_DT_S))
        fallback_dt = source.get("dt_s", source.get("projection_dt_s"))
        return source, float(
            DEFAULT_PROJECTION_DT_S if fallback_dt is None else fallback_dt
        )
    raise TypeError(f"cannot read effort hold policy from {type(source).__name__}")


def read_effort_hold_policy(source: Any) -> EffortHoldPolicy:
    """Read the effort hold policy from a bundle, manifest dict, or path (#12279).

    This unified helper allows engine replays (MuJoCo, Drake, Simscape, etc.)
    to interpret the bundle's effort hold configuration consistently.
    """
    if hasattr(source, "effort_hold_policy") and callable(source.effort_hold_policy):
        return source.effort_hold_policy()  # type: ignore[no-any-return]

    policy_dict, dt_s = _extract_manifest_policy(source)

    if "effort_hold_dt_s" in policy_dict:
        effort_hold_dt_s = float(policy_dict["effort_hold_dt_s"])
    elif "effort_hold_s" in policy_dict:
        effort_hold_dt_s = float(policy_dict["effort_hold_s"])
    elif "effort_holds_per_step" in policy_dict:
        effort_hold_dt_s = dt_s / int(policy_dict["effort_holds_per_step"])
    else:
        effort_hold_dt_s = DEFAULT_EFFORT_HOLD_DT_S

    if "effort_substeps" in policy_dict:
        effort_substeps = int(policy_dict["effort_substeps"])
    elif "substeps" in policy_dict and "effort_holds_per_step" in policy_dict:
        substeps = int(policy_dict["substeps"])
        holds = int(policy_dict["effort_holds_per_step"])
        effort_substeps = max(1, substeps // holds)
    elif abs(effort_hold_dt_s - FINE_EFFORT_HOLD_DT_S) < 1e-9:
        effort_substeps = FINE_EFFORT_SUBSTEPS
    else:
        effort_substeps = DEFAULT_EFFORT_SUBSTEPS

    if "projection_dt_s" in policy_dict:
        projection_dt_s = float(policy_dict["projection_dt_s"])
    else:
        projection_dt_s = dt_s

    return EffortHoldPolicy(
        effort_hold_dt_s=effort_hold_dt_s,
        effort_substeps=effort_substeps,
        projection_dt_s=projection_dt_s,
    )
