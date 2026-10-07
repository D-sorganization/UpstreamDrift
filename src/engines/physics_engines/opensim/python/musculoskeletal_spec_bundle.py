"""Reader for the same-input reference bundle used by the musculoskeletal swing.

Part of issue #11617 (epic #11605), phase 2.  The bundle (``same-input-bundle/v1``)
is a forward-dynamics-consistent MuJoCo reference of a matched swing: 44 spec
coordinates sampled at 1 ms, the joint efforts that produced them and the exact
full-body specification.  This module only reads the ``.npz`` container with
numpy, so it does not depend on the (separately reviewed) ``same_input`` package.

Conventions
-----------
* ``reference_q`` / ``reference_v`` have ``steps + 1`` rows, ``efforts`` has
  ``steps`` rows: row ``k`` of the efforts was held (zero order) over
  ``[t_k, t_{k+1}]``.
* The six root coordinates (three translations, three pelvis rotations) carry
  zero effort by policy: the ground contact supplies the root wrench.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import require

SCHEMA = "same-input-bundle/v1"
ROOT_COORDINATE_COUNT = 6
_REQUIRED_KEYS = ("manifest", "spec", "q0", "v0", "efforts", "reference_q")


@dataclass(frozen=True)
class SpecBundle:
    """Immutable view of a same-input bundle.

    Attributes:
        spec_bytes: the ``full-body-v1`` document the reference was produced with.
        coordinate_order: the 44 spec coordinate names.
        dt_s: sample interval in seconds.
        q: ``(steps + 1, nv)`` coordinates (rad or m).
        v: ``(steps + 1, nv)`` coordinate rates.
        efforts: ``(steps, nv)`` joint-conjugate generalised forces (N m or N).
        manifest: parsed manifest (provenance, policy, hashes).
        sha256: hash of the bundle file, recorded in receipts.
    """

    spec_bytes: bytes
    coordinate_order: tuple[str, ...]
    dt_s: float
    q: np.ndarray
    v: np.ndarray
    efforts: np.ndarray
    manifest: dict[str, Any]
    sha256: str

    def __post_init__(self) -> None:
        nv = len(self.coordinate_order)
        require(self.dt_s > 0.0, "dt_s must be positive")
        require(self.q.ndim == 2 and self.q.shape[1] == nv, "q must be (steps+1, nv)")
        require(self.v.shape == self.q.shape, "v must have the shape of q")
        require(
            self.efforts.shape == (self.q.shape[0] - 1, nv),
            "efforts must be (steps, nv) with steps = len(q) - 1",
        )
        for name, array in (("q", self.q), ("v", self.v), ("efforts", self.efforts)):
            require(bool(np.isfinite(array).all()), f"{name} must be finite")

    @property
    def steps(self) -> int:
        return int(self.efforts.shape[0])

    @property
    def nv(self) -> int:
        return len(self.coordinate_order)

    @property
    def capture(self) -> str:
        return str(self.manifest.get("provenance", {}).get("capture", "unknown"))

    def times(self) -> np.ndarray:
        """Sample times of ``q``/``v`` (``steps + 1`` values, seconds)."""
        return np.arange(self.q.shape[0]) * self.dt_s

    def step_accelerations(self) -> np.ndarray:
        """Mean acceleration over each held-effort step, ``(steps, nv)``.

        ``(v[k+1] - v[k]) / dt`` is the exact step mean of the plant acceleration
        under the zero-order-held effort, so it pairs with ``efforts[k]``.
        """
        return np.diff(self.v, axis=0) / self.dt_s

    def step_midpoint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Midpoint coordinates and rates of each step, ``(steps, nv)`` each."""
        return (
            0.5 * (self.q[:-1] + self.q[1:]),
            0.5 * (self.v[:-1] + self.v[1:]),
        )

    def index(self, name: str) -> int:
        """Column of coordinate ``name``; raises ``KeyError`` if unknown."""
        try:
            return self.coordinate_order.index(name)
        except ValueError:
            raise KeyError(f"unknown coordinate {name!r}") from None


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_spec_bundle(path: str | Path) -> SpecBundle:
    """Load a ``same-input-bundle/v1`` ``.npz`` file.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the schema, keys or shapes are not those of the bundle.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"bundle not found: {file}")
    with np.load(file) as data:
        missing = [k for k in _REQUIRED_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"bundle is missing keys {missing}")
        manifest = json.loads(str(data["manifest"]))
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"unsupported bundle schema {manifest.get('schema')!r}")
        spec_bytes = bytes(data["spec"])
        q = np.asarray(data["reference_q"], dtype=float)
        v = (
            np.asarray(data["reference_v"], dtype=float)
            if "reference_v" in data.files
            else np.zeros_like(q)
        )
        efforts = np.asarray(data["efforts"], dtype=float)
    spec = json.loads(spec_bytes)
    order = tuple(manifest["coordinate_order"])
    if tuple(spec["coordinate_order"]) != order:
        raise ValueError("manifest and spec disagree on the coordinate order")
    return SpecBundle(
        spec_bytes=spec_bytes,
        coordinate_order=order,
        dt_s=float(manifest["dt_s"]),
        q=q,
        v=v,
        efforts=efforts,
        manifest=manifest,
        sha256=sha256_of(file),
    )
