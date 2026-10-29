"""Receipt contract for the unified cross-engine perturbation study (#13795).

Every engine entry records the same six setup fields, and
:func:`validate_receipt` fails closed when a field is missing, engines
disagree, a stub backend is counted or a deferred receipt carries numbers.
Cross-engine CVs use ``_safe_cv``: a near-zero mean is ``None`` plus a reason,
never perfect agreement (#12807). Fewer than two real engines gives a
``deferred`` receipt with no metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.shared.python.analysis.cross_engine import (
    BACKEND_REAL,
    BACKEND_STUB,
    CV_UNDEFINED_REASON,
    METRIC_KEYS,
    _safe_cv,
)
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineRunResult,
)
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TriplePendulumSpec,
)

SCHEMA = "upstreamdrift.cross_engine_perturbation_receipt.v1"
STATUS_MEASURED = "measured"
STATUS_DEFERRED = "deferred_insufficient_real_engines"
BACKEND_NOT_RUN = "not_run"
INTERPOLATION_MODES = ("zoh", "linear")
#: Fields every engine entry must record, with identical values.
SETUP_FIELDS = (
    "spec_hash",
    "gravity_m_s2",
    "dt_s",
    "integrator",
    "seed",
    "interpolation_mode",
)
MIN_COUNTED_ENGINES = 2
#: The runner cannot read g, dt or the integrator back from an engine.
SETUP_ATTESTATION = "declared_by_harness_not_read_back_from_engine"


class ReceiptContractError(ValueError):
    """Raised when a receipt violates the cross-engine receipt contract."""


def triple_pendulum_spec_hash(spec: TriplePendulumSpec) -> str:
    """SHA-256 of the canonical JSON form of ``spec``."""
    canonical = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SharedSetup:
    """The single setup every engine in one study must run (fail-closed)."""

    spec_hash: str
    gravity_m_s2: float
    dt_s: float
    integrator: str
    seed: int
    interpolation_mode: str

    def __post_init__(self) -> None:
        for name in ("gravity_m_s2", "dt_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not self.spec_hash or not self.integrator:
            raise ValueError("spec_hash and integrator must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if self.interpolation_mode not in INTERPOLATION_MODES:
            raise ValueError(f"unknown interpolation_mode: {self.interpolation_mode!r}")

    @classmethod
    def from_spec(
        cls,
        spec: TriplePendulumSpec,
        *,
        dt_s: float,
        integrator: str,
        seed: int,
        interpolation_mode: str,
    ) -> SharedSetup:
        """Take ``g`` and the hash from the one spec."""
        return cls(
            triple_pendulum_spec_hash(spec),
            float(spec.gravity_m_s2),
            dt_s,
            integrator,
            seed,
            interpolation_mode,
        )


def _cross_engine_cv(means: dict[str, dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        values = np.array([engine[metric] for engine in means.values()])
        cv = _safe_cv(float(np.mean(values)), float(np.std(values)))
        undefined = math.isnan(cv)
        summary[metric] = {
            "cv": None if undefined else cv,
            "reason": CV_UNDEFINED_REASON if undefined else None,
        }
    return summary


def build_receipt(
    setup: SharedSetup,
    backends: dict[str, str],
    results: dict[str, CrossEngineRunResult] | None = None,
    *,
    n_trials: int,
    t_end_s: float,
    noise_amplitude_nm: float = 0.1,
) -> dict[str, Any]:
    """Assemble a receipt; only real backends with results are counted."""
    if not backends:
        raise ReceiptContractError("backends must name at least one engine")
    results = results or {}
    means = {
        name: {m: getattr(results[name], f"mean_{m}") for m in METRIC_KEYS}
        for name, backend in backends.items()
        if backend == BACKEND_REAL and name in results
    }
    measured = len(means) >= MIN_COUNTED_ENGINES
    return {
        "schema": SCHEMA,
        "status": STATUS_MEASURED if measured else STATUS_DEFERRED,
        "setup": asdict(setup),
        "n_trials": n_trials,
        "t_end_s": t_end_s,
        "noise_amplitude_nm": noise_amplitude_nm,
        "engines": {
            name: {
                **asdict(setup),
                "backend": backend,
                "counted": measured and name in means,
                "means": means[name] if measured and name in means else None,
            }
            for name, backend in backends.items()
        },
        "cv_summary": _cross_engine_cv(means) if measured else None,
        "stubbed_engines": sorted(n for n, b in backends.items() if b == BACKEND_STUB),
        "setup_attestation": SETUP_ATTESTATION,
    }


def _check_measured(receipt: dict[str, Any]) -> None:
    counted = {n: e for n, e in receipt["engines"].items() if e["counted"]}
    if len(counted) < MIN_COUNTED_ENGINES:
        raise ReceiptContractError("measured receipt needs two counted engines")
    for name, entry in counted.items():
        if entry["backend"] != BACKEND_REAL:
            raise ReceiptContractError(f"stub backend counted for engine {name!r}")
    summary = receipt["cv_summary"] or {}
    if set(summary) != set(METRIC_KEYS):
        raise ReceiptContractError("cv_summary must cover every metric")
    for metric, cv in summary.items():
        if cv["cv"] is None and not cv["reason"]:
            raise ReceiptContractError(f"undefined cv for {metric} has no reason")


def validate_receipt(receipt: dict[str, Any]) -> None:
    """Raise :class:`ReceiptContractError` unless ``receipt`` honours the contract."""
    if receipt.get("schema") != SCHEMA:
        raise ReceiptContractError(f"schema must be {SCHEMA}")
    setup = receipt.get("setup", {})
    try:
        SharedSetup(**{field: setup[field] for field in SETUP_FIELDS})
    except (KeyError, ValueError) as exc:
        raise ReceiptContractError(f"invalid shared setup: {exc!r}") from exc
    engines = receipt.get("engines") or {}
    if len(engines) < MIN_COUNTED_ENGINES:
        raise ReceiptContractError("receipt must name at least two engines")
    for name, entry in engines.items():
        for field in SETUP_FIELDS:
            if field not in entry:
                raise ReceiptContractError(f"engine {name!r} does not record {field}")
            if entry[field] != setup[field]:
                raise ReceiptContractError(
                    f"engine {name!r} disagrees on {field}: {entry[field]!r}"
                )
    if receipt["status"] == STATUS_MEASURED:
        _check_measured(receipt)
    elif receipt["status"] != STATUS_DEFERRED:
        raise ReceiptContractError(f"unknown status {receipt['status']!r}")
    elif receipt["cv_summary"] is not None or any(
        e["counted"] or e["means"] is not None for e in engines.values()
    ):
        raise ReceiptContractError("deferred receipt must not carry numbers")
