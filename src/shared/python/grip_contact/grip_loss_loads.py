"""Load recorded grip pad forces for offline grip-loss prediction (#15128)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from src.shared.python.grip_contact.contact_run import ContactRun, SIDES

DEFAULT_GRIP_LOAD_PROVIDER = "contact_run"


@dataclass(frozen=True)
class GripLoadSeries:
    """Recorded per-patch normal and tangential forces (world, SI)."""

    club: str
    source: str
    patch_labels: tuple[str, ...]
    time_s: np.ndarray
    normal_force_n: np.ndarray
    tangential_force_n: np.ndarray

    def __post_init__(self) -> None:
        t = np.asarray(self.time_s, dtype=float)
        if t.ndim != 1 or t.size < 2 or not np.all(np.diff(t) > 0.0):
            raise ValueError("time_s must be strictly increasing")
        n, p = t.size, len(self.patch_labels)
        for name in ("normal_force_n", "tangential_force_n"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n, p) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be finite with shape (n, pads)")


@runtime_checkable
class GripLoadProvider(Protocol):
    """Swappable source for recorded per-pad grip loads."""

    def load(self, club: str, *, evidence_root: Path) -> GripLoadSeries:
        """Return pad load history for ``club`` under ``evidence_root``."""


_LOAD_REGISTRY: dict[str, GripLoadProvider] = {}


def register_grip_load_provider(name: str, provider: GripLoadProvider) -> None:
    if not isinstance(name, str) or not name.strip():
        raise TypeError("provider name must be a non-empty string")
    if not isinstance(provider, GripLoadProvider):
        raise TypeError("provider must implement GripLoadProvider")
    _LOAD_REGISTRY[name] = provider


def resolve_grip_load_provider(name: str) -> GripLoadProvider:
    provider = _LOAD_REGISTRY.get(name)
    if provider is None:
        raise ValueError(f"unknown grip load provider {name!r}")
    return provider


@dataclass(frozen=True)
class _ContactRunProvider:
    def load(self, club: str, *, evidence_root: Path) -> GripLoadSeries:
        path = evidence_root / "contact" / f"mujoco_{club}_series.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        return loads_from_contact_run(ContactRun.load_npz(path), club=club)


def loads_from_contact_run(run: ContactRun, *, club: str) -> GripLoadSeries:
    """Build :class:`GripLoadSeries` from a committed contact run."""
    if not isinstance(run, ContactRun):
        raise TypeError(f"run must be ContactRun, got {type(run).__name__}")
    series = run.series
    time_s = np.asarray(series.time_s, dtype=float)
    n = time_s.size
    labels: list[str] = []
    normals: list[np.ndarray] = []
    tangentials: list[np.ndarray] = []

    for side in SIDES:
        role = "lead" if side == "L" else "trail"
        fn = np.asarray(run.normal_force_n[side], dtype=float)
        if fn.shape[0] != n:
            raise ValueError("normal_force series must match time samples")
        if run.tangential_force_n is not None and side in run.tangential_force_n:
            ft = np.asarray(run.tangential_force_n[side], dtype=float)
        else:
            ft = _estimate_tangential_from_hand_wrench(series, side, fn)
        for pad_idx in range(fn.shape[1]):
            labels.append(f"{role}:{pad_idx}")
            normals.append(fn[:, pad_idx])
            tangentials.append(ft[:, pad_idx])

    normal_force_n = np.column_stack(normals)
    tangential_force_n = np.column_stack(tangentials)
    return GripLoadSeries(
        club=club,
        source=str(club),
        patch_labels=tuple(labels),
        time_s=time_s,
        normal_force_n=normal_force_n,
        tangential_force_n=tangential_force_n,
    )


def _estimate_tangential_from_hand_wrench(
    series: object,
    side: str,
    normal_force_n: np.ndarray,
    *,
    friction: float | None = None,
) -> np.ndarray:
    """Allocate hand transverse demand across pads proportional to normal force."""
    from src.shared.python.grip_contact.parity import GripKineticsSeries

    if not isinstance(series, GripKineticsSeries):
        raise TypeError("series must be GripKineticsSeries")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    force = np.asarray(series.force_on_club_n[side], dtype=float)
    axis = np.asarray(series.club_rotation[:, :, 0], dtype=float)
    axial = np.einsum("ni,ni->n", force, axis)[:, None] * axis
    transverse = force - axial
    transverse_mag = np.linalg.norm(transverse, axis=1)
    fn = np.asarray(normal_force_n, dtype=float)
    fn_sum = np.maximum(np.sum(fn, axis=1), 1e-12)
    if friction is None:
        # Caller supplies friction via evaluator margin; here bound demand only.
        cap = transverse_mag
    else:
        cap = np.minimum(transverse_mag, friction * np.sum(fn, axis=1))
    return cap[:, None] * (fn / fn_sum[:, None])


register_grip_load_provider(DEFAULT_GRIP_LOAD_PROVIDER, _ContactRunProvider())

__all__ = [
    "DEFAULT_GRIP_LOAD_PROVIDER",
    "GripLoadProvider",
    "GripLoadSeries",
    "loads_from_contact_run",
    "register_grip_load_provider",
    "resolve_grip_load_provider",
]
