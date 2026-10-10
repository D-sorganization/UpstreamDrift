"""Same-input bundle: everything an engine needs to reproduce a motion (#11607).

A bundle fixes the model (spec bytes), a closure-consistent initial state,
the per-step efforts (zero-order hold at ``dt_s``) and the integration policy.
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
    ``reference_v`` are (steps + 1, nv) and start at ``q0``/``v0``.
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

    @property
    def spec_sha256(self) -> str:
        return hashlib.sha256(self.spec_bytes).hexdigest()

    @property
    def steps(self) -> int:
        return int(self.efforts.shape[0])

    def manifest(self) -> dict[str, Any]:
        """JSON-serialisable description stored beside the arrays."""
        return {
            "schema": SCHEMA,
            "spec_sha256": self.spec_sha256,
            "coordinate_order": list(self.coordinate_order),
            "dt_s": self.dt_s,
            "steps": self.steps,
            "duration_s": self.steps * self.dt_s,
            "policy": POLICY,
            "reference_engine": self.reference_engine,
            "provenance": self.provenance,
        }

    def save(self, path: Path) -> None:
        """Write the bundle as one compressed ``.npz``."""
        np.savez_compressed(
            path,
            manifest=np.array(json.dumps(self.manifest(), sort_keys=True)),
            spec=np.frombuffer(self.spec_bytes, dtype=np.uint8),
            q0=self.q0,
            v0=self.v0,
            efforts=self.efforts,
            reference_q=self.reference_q,
            reference_v=self.reference_v,
        )

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
            )
        if bundle.spec_sha256 != manifest["spec_sha256"]:
            raise ValueError("bundle spec hash does not match its manifest")
        return bundle
