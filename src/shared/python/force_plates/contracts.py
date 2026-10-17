"""Vertical-only two-plate force trial contract (OF-01, epic #13061).

A vertical force plate pair measures, per foot, the normal (vertical) load and
the centre of pressure (CoP) of that load.  It does **not** measure shear
(``F_x``, ``F_y``) or the free moment about the vertical axis; those stay
unobserved and are never reported as zero.

Stance frame (metres, viewed from above, binding): origin midway between the
plate centres, ``+x`` from the left plate toward the right plate, ``+y`` toward
the ball, ``+z`` up.  The golfer faces ``+y``, so the left plate is under the
left foot for either handedness; ``handedness`` only decides which foot leads
(``RH``: left leads).  Loads are exerted by the ground on the foot, positive
upward, in N.

Source rules carried by the record (they match how a vertical plate computes
its outputs and are checked on construction):

* a foot's CoP is NaN exactly where its load is below ``cop_min_load_n``;
* the net CoP is the load-weighted CoP of the feet above that threshold, and
  NaN where neither foot is;
* both CoP components are NaN together.

Pure numpy; no engine or file imports.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FOOT_LABELS",
    "NET_LABEL",
    "NET_COP_TOL_M",
    "Handedness",
    "PlateProvenance",
    "PlateUncertainty",
    "VerticalPlateTrial",
    "loaded_net_cop",
]

Array: TypeAlias = NDArray[np.float64]
Handedness = Literal["RH", "LH"]

#: Anatomical foot labels, in stance order (left plate at ``-x``).
FOOT_LABELS: tuple[str, str] = ("left", "right")
#: Label of the combined (both-feet) CoP.
NET_LABEL = "net"
#: Tolerance on the net-CoP identity [m]: source rounding (0.01 mm in a CSV
#: export) plus float error, far below any physical CoP uncertainty.
NET_COP_TOL_M = 1e-4

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEAD = {"RH": "left", "LH": "right"}


def _finite_positive(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _readonly(value: ArrayLike, name: str) -> Array:
    """Read-only float64 copy; NaN is allowed, infinity is not."""
    arr = np.array(value, dtype=np.float64, copy=True)
    if np.isinf(arr).any():
        raise ValueError(f"{name} must not contain infinity")
    arr.flags.writeable = False
    return arr


def loaded_net_cop(
    load_n: Mapping[str, ArrayLike], cop_m: Mapping[str, ArrayLike]
) -> Array:
    """Load-weighted CoP ``(T, 2)`` of the feet whose CoP is available.

    A foot with a NaN CoP (below its threshold) carries no weight; the result is
    NaN where no foot has a CoP.  This is the source definition of the net CoP
    of a vertical plate pair.
    """
    loads = [np.asarray(load_n[k], dtype=np.float64) for k in FOOT_LABELS]
    cops = [np.asarray(cop_m[k], dtype=np.float64) for k in FOOT_LABELS]
    weights = [
        np.where(np.isnan(c[:, 0]), 0.0, w) for w, c in zip(loads, cops, strict=True)
    ]
    total = weights[0] + weights[1]
    weighted = sum(
        w[:, None] * np.nan_to_num(c) for w, c in zip(weights, cops, strict=True)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        net = weighted / total[:, None]
    net[total <= 0.0] = np.nan
    return net


@dataclass(frozen=True)
class PlateUncertainty:
    """Declared 1-sigma measurement uncertainty of a vertical plate (no defaults).

    The load uncertainty of a foot is ``hypot(force_floor_n, force_fraction * F)``
    (an absolute floor plus a fraction of the reading); the CoP uncertainty is
    ``cop_m`` in each horizontal axis.  ``source`` names where the numbers come
    from (a calibration record, a datasheet, or a declared hypothesis).

    Preconditions: ``force_floor_n`` and ``cop_m`` positive and finite,
    ``force_fraction`` finite and non-negative, ``source`` non-empty
    (``ValueError``).
    """

    force_floor_n: float
    force_fraction: float
    cop_m: float
    source: str

    def __post_init__(self) -> None:
        _finite_positive(self.force_floor_n, "force_floor_n")
        _finite_positive(self.cop_m, "cop_m")
        frac = self.force_fraction
        if not isinstance(frac, (int, float)) or not math.isfinite(frac) or frac < 0.0:
            raise ValueError(f"force_fraction must be finite and >= 0, got {frac!r}")
        _text(self.source, "source")

    def force_sigma_n(self, load_n: ArrayLike) -> Array:
        """1-sigma load uncertainty [N] for the loads ``load_n`` [N]."""
        load = np.asarray(load_n, dtype=np.float64)
        return np.hypot(self.force_floor_n, self.force_fraction * load)


@dataclass(frozen=True)
class PlateProvenance:
    """Where a trial came from.

    ``file_sha256`` maps a file role to its SHA-256 digest (lower-case hex).
    ``calibration_id`` names the calibration that produced the loads when the
    source records one.  ``qualification`` is a durable reference to a passed
    device qualification (deferred-validation record DV-13061); ``None`` means
    the measurement is **unqualified**.
    """

    device: str
    source_format: str
    source_revision: str
    file_sha256: Mapping[str, str]
    calibration_id: str | None = None
    qualification: str | None = None

    def __post_init__(self) -> None:
        for name in ("device", "source_format", "source_revision"):
            _text(getattr(self, name), name)
        digests = dict(self.file_sha256)
        for role, digest in digests.items():
            _text(role, "file_sha256 role")
            if not isinstance(digest, str) or not _SHA256.match(digest):
                raise ValueError(f"file_sha256[{role!r}] is not a sha256 hex digest")
        for name in ("calibration_id", "qualification"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        object.__setattr__(self, "file_sha256", MappingProxyType(digests))

    @property
    def qualified(self) -> bool:
        """True only when a device qualification reference is recorded."""
        return self.qualification is not None


@dataclass(frozen=True, eq=False)
class VerticalPlateTrial:
    """A vertical-only two-plate force trial (see the module docstring).

    ``load_n`` maps each of :data:`FOOT_LABELS` to ``(T,)`` loads [N];
    ``cop_m`` maps the feet and :data:`NET_LABEL` to ``(T, 2)`` stance-frame
    CoPs [m] (NaN where unavailable).  ``events_s`` holds force events (time
    from ``times_s[0]``'s clock, or ``None`` when not detected) in the source's
    own names; they are force events, not swing positions.
    ``corner_forces_n`` optionally keeps the ``(T, 8)`` raw corner channels for
    diagnostics.  ``capacity_n`` is the declared total capacity of the pair.

    Raises ``ValueError`` (``TypeError`` for wrong object types) on any broken
    invariant; arrays are stored as read-only copies.
    """

    times_s: Array
    load_n: Mapping[str, Array]
    cop_m: Mapping[str, Array]
    body_weight_n: float
    handedness: Handedness
    events_s: Mapping[str, float | None]
    cop_min_load_n: float
    capacity_n: float
    provenance: PlateProvenance
    uncertainty: PlateUncertainty
    corner_forces_n: Array | None = field(default=None)

    def __post_init__(self) -> None:
        self._check_types()
        times = self._times()
        loads = self._loads(times.shape[0])
        cops = self._cops(times.shape[0], loads)
        self._check_events(times)
        corner = self._corner(times.shape[0])
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "load_n", MappingProxyType(loads))
        object.__setattr__(self, "cop_m", MappingProxyType(cops))
        object.__setattr__(self, "events_s", MappingProxyType(dict(self.events_s)))
        object.__setattr__(self, "corner_forces_n", corner)

    # ------------------------------------------------------------ checks --
    def _check_types(self) -> None:
        if not isinstance(self.provenance, PlateProvenance):
            raise TypeError("provenance must be a PlateProvenance")
        if not isinstance(self.uncertainty, PlateUncertainty):
            raise TypeError("uncertainty must be a PlateUncertainty")
        if self.handedness not in get_args(Handedness):
            raise ValueError(
                f"handedness must be 'RH' or 'LH', got {self.handedness!r}"
            )
        _finite_positive(self.body_weight_n, "body_weight_n")
        _finite_positive(self.cop_min_load_n, "cop_min_load_n")
        _finite_positive(self.capacity_n, "capacity_n")

    def _times(self) -> Array:
        times = _readonly(self.times_s, "times_s")
        if times.ndim != 1 or times.size < 2:
            raise ValueError("times_s must be a 1-D array of at least two samples")
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be finite and strictly increasing")
        return times

    def _loads(self, n: int) -> dict[str, Array]:
        if set(self.load_n) != set(FOOT_LABELS):
            raise ValueError(f"load_n must be keyed exactly by {FOOT_LABELS}")
        loads = {k: _readonly(self.load_n[k], f"load_n[{k!r}]") for k in FOOT_LABELS}
        for key, load in loads.items():
            if load.shape != (n,):
                raise ValueError(
                    f"load_n[{key!r}] must have shape ({n},), times_s length"
                )
            if not np.isfinite(load).all() or np.any(load < 0.0):
                raise ValueError(f"load_n[{key!r}] must be finite and non-negative")
        if np.any(loads["left"] + loads["right"] > self.capacity_n):
            raise ValueError(
                f"total load exceeds the declared capacity {self.capacity_n} N"
            )
        return loads

    def _cops(self, n: int, loads: Mapping[str, Array]) -> dict[str, Array]:
        keys = (*FOOT_LABELS, NET_LABEL)
        if set(self.cop_m) != set(keys):
            raise ValueError(f"cop_m must be keyed exactly by {keys}")
        cops = {k: _readonly(self.cop_m[k], f"cop_m[{k!r}]") for k in keys}
        for key, cop in cops.items():
            if cop.shape != (n, 2):
                raise ValueError(f"cop_m[{key!r}] must have shape ({n}, 2)")
            if np.any(np.isnan(cop[:, 0]) != np.isnan(cop[:, 1])):
                raise ValueError(f"cop_m[{key!r}] components must be NaN together")
        for key in FOOT_LABELS:
            if np.any(np.isnan(cops[key][:, 0]) != (loads[key] < self.cop_min_load_n)):
                raise ValueError(
                    f"cop_m[{key!r}] must be NaN exactly where its load is below the "
                    f"{self.cop_min_load_n} N threshold"
                )
        self._check_net_cop(cops, loads)
        return cops

    def _check_net_cop(
        self, cops: Mapping[str, Array], loads: Mapping[str, Array]
    ) -> None:
        expected = loaded_net_cop(loads, cops)
        net = cops[NET_LABEL]
        if np.any(np.isnan(net[:, 0]) != np.isnan(expected[:, 0])):
            raise ValueError(
                "net CoP must be NaN exactly where no foot is above threshold"
            )
        both = ~np.isnan(expected[:, 0])
        if np.any(np.abs(net[both] - expected[both]) > NET_COP_TOL_M):
            raise ValueError("net CoP must be the load-weighted CoP of the loaded feet")

    def _check_events(self, times: Array) -> None:
        span = (float(times[0]), float(times[-1]))
        for name, value in dict(self.events_s).items():
            if not isinstance(name, str) or not name:
                raise ValueError("events_s names must be non-empty strings")
            if value is None:
                continue
            if not math.isfinite(value) or not span[0] <= value <= span[1]:
                raise ValueError(f"events_s[{name!r}] = {value!r} lies outside {span}")

    def _corner(self, n: int) -> Array | None:
        if self.corner_forces_n is None:
            return None
        corner = _readonly(self.corner_forces_n, "corner_forces_n")
        if corner.shape != (n, 8) or not np.isfinite(corner).all():
            raise ValueError(f"corner_forces_n must be finite with shape ({n}, 8)")
        return corner

    # ------------------------------------------------------------- views --
    @property
    def n_samples(self) -> int:
        """Number of samples ``T``."""
        return int(self.times_s.shape[0])

    @property
    def lead_foot(self) -> str:
        """The lead foot label (``left`` for a right-handed golfer)."""
        return _LEAD[self.handedness]

    @property
    def trail_foot(self) -> str:
        """The trail foot label."""
        return FOOT_LABELS[1 - FOOT_LABELS.index(self.lead_foot)]

    def net_load_n(self) -> Array:
        """Total vertical load of both feet [N], ``(T,)``."""
        return self.load_n["left"] + self.load_n["right"]

    def lead_fraction(self) -> Array:
        """Lead-foot share of the total load, NaN where the total is below threshold."""
        total = self.net_load_n()
        with np.errstate(invalid="ignore", divide="ignore"):
            share = self.load_n[self.lead_foot] / total
        return np.where(total < self.cop_min_load_n, np.nan, share)

    def load_sigma_n(self, foot: str) -> Array:
        """Declared 1-sigma load uncertainty of ``foot`` [N], ``(T,)``."""
        if foot not in FOOT_LABELS:
            raise ValueError(f"foot must be one of {FOOT_LABELS}, got {foot!r}")
        return self.uncertainty.force_sigma_n(self.load_n[foot])
