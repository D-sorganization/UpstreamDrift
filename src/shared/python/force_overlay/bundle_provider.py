"""Engine-agnostic overlay frame provider for same-input bundles (NV-3, #11676).

Builds :class:`ForceTorqueFrame` series from a same-input bundle without
touching any engine SDK:

* bundle efforts -> one joint-torque wrench per joint anchor (coordinates that
  share an anchor, e.g. the three hip rotations, merge into one vector);
* the shared contact law (``evaluate_contact_samples`` of any full-body model)
  -> one ground reaction force per foot body, applied at the normal-load
  weighted centre of pressure;
* the centre of mass -> the system weight, as a gravity wrench at the CoM.

Engine adapters only have to supply the two small protocols below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any, NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import joint_torque_wrench
from src.shared.python.force_overlay.series import ForceTorqueSeries
from src.shared.python.motion_matching.same_input import InputBundle

Array = NDArray[np.float64]
Vec3 = tuple[float, float, float]
_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_ANCHOR_DECIMALS = 5


class JointFrame(NamedTuple):
    """World-frame anchor and unit rotation axis of one rotational coordinate."""

    body: str
    anchor_m: Vec3
    axis_world: Vec3


@runtime_checkable
class KinematicsSource(Protocol):
    """Kinematics an engine adapter supplies for torque and weight glyphs."""

    total_mass_kg: float

    def joint_frames(
        self, coordinates: Mapping[str, float]
    ) -> Mapping[str, JointFrame]:
        """Rotational coordinates only, keyed by coordinate name."""
        ...

    def center_of_mass_m(self, coordinates: Mapping[str, float]) -> Vec3:
        """System centre of mass in the world frame."""
        ...


@runtime_checkable
class ContactSource(Protocol):
    """The shared contact law, e.g. any full-body model adapter."""

    def evaluate_contact_samples(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Mapping[str, Any]:
        """Per-sphere ``ContactSample`` values."""
        ...


def _label_part(text: str) -> str:
    return _LABEL_SAFE.sub("_", text).strip("_") or "x"


def _tuple3(vec: Sequence[float]) -> Vec3:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


class BundleOverlayProvider:
    """Time series of overlay frames for one bundle (or a replay of it).

    Preconditions: ``q``/``v`` default to the bundle reference states and must
    have shape ``(steps + 1, nv)``; ``efforts`` come from the bundle.
    Postconditions: every frame lies on the bundle time grid ``k * dt_s``.
    """

    def __init__(
        self,
        bundle: InputBundle,
        contact: ContactSource,
        kinematics: KinematicsSource,
        *,
        engine: str,
        q: Array | None = None,
        v: Array | None = None,
        torque_floor_nm: float = 0.0,
        include_weight: bool = True,
    ) -> None:
        if not isinstance(bundle, InputBundle):
            raise TypeError(
                f"bundle must be an InputBundle, got {type(bundle).__name__}"
            )
        if not engine:
            raise ValueError("engine must be a non-empty string")
        if not (math.isfinite(torque_floor_nm) and torque_floor_nm >= 0.0):
            raise ValueError("torque_floor_nm must be finite and non-negative")
        expected = bundle.reference_q.shape
        self._q = bundle.reference_q if q is None else np.asarray(q, dtype=float)
        self._v = bundle.reference_v if v is None else np.asarray(v, dtype=float)
        if self._q.shape != expected or self._v.shape != expected:
            raise ValueError(f"q and v must have shape {expected}")
        spec = json.loads(bundle.spec_bytes)
        self._bundle = bundle
        self._contact = contact
        self._kin = kinematics
        self._engine = engine
        self._floor = float(torque_floor_nm)
        self._include_weight = include_weight
        self._gravity = np.asarray(
            spec.get("gravity_m_s2", (0.0, 0.0, -9.80665)), float
        )
        self._sphere_body = {
            s["name"]: s["body"] for s in spec.get("contact", {}).get("spheres", ())
        }

    def __len__(self) -> int:
        return int(self._q.shape[0])

    def frame_at(self, index: int) -> ForceTorqueFrame:
        """Overlay frame at state ``index`` (effort ``min(index, steps - 1)``)."""
        if not 0 <= index < len(self):
            raise IndexError(f"frame index {index} outside [0, {len(self) - 1}]")
        names = self._bundle.coordinate_order
        coords = dict(zip(names, map(float, self._q[index]), strict=True))
        rates = dict(zip(names, map(float, self._v[index]), strict=True))
        efforts = self._bundle.efforts[min(index, self._bundle.steps - 1)]
        wrenches = [
            *self._contact_wrenches(coords, rates),
            *self._torque_wrenches(coords, dict(zip(names, efforts, strict=True))),
            *self._weight_wrenches(coords),
        ]
        return ForceTorqueFrame(
            time_s=float(index * self._bundle.dt_s),
            engine=self._engine,
            wrenches=tuple(wrenches),
            metadata={
                "source": "same_input_bundle",
                "spec_sha256": self._bundle.spec_sha256,
            },
        )

    def series(self, stride: int = 1) -> ForceTorqueSeries:
        """Frames at every ``stride``-th state."""
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride!r}")
        frames = tuple(self.frame_at(k) for k in range(0, len(self), stride))
        return ForceTorqueSeries(frames=frames, engine=self._engine)

    def _contact_wrenches(
        self, coords: Mapping[str, float], rates: Mapping[str, float]
    ) -> list[OverlayWrench]:
        per_body: dict[str, list[tuple[Array, Array, float]]] = {}
        samples = self._contact.evaluate_contact_samples(coords, rates)
        for name, sample in samples.items():
            force = np.asarray(sample.normal_force_n) + np.asarray(
                sample.friction_force_n
            )
            if float(np.linalg.norm(force)) <= 1e-9:
                continue
            body = self._sphere_body.get(name, name.rsplit("_", 1)[-1])
            load = float(np.linalg.norm(sample.normal_force_n)) or 1e-9
            per_body.setdefault(body, []).append(
                (np.asarray(sample.contact_point_m, float), force, load)
            )
        out = []
        for body, items in sorted(per_body.items()):
            total = np.sum([f for _, f, _ in items], axis=0)
            weights = np.array([w for _, _, w in items])
            cop = np.sum([w * p for (p, _, w) in items], axis=0) / weights.sum()
            out.append(
                OverlayWrench(
                    WrenchKind.CONTACT,
                    f"contact:grf_{_label_part(body)}",
                    body,
                    _tuple3(cop),
                    force_n=_tuple3(total),
                    source=f"{self._engine}:shared_contact_law",
                )
            )
        return out

    def _torque_wrenches(
        self, coords: Mapping[str, float], effort: Mapping[str, float]
    ) -> list[OverlayWrench]:
        groups: dict[tuple[float, ...], list[tuple[str, JointFrame]]] = {}
        for coord, frame in self._kin.joint_frames(coords).items():
            key = tuple(np.round(frame.anchor_m, _ANCHOR_DECIMALS))
            groups.setdefault(key, []).append((coord, frame))
        out: list[OverlayWrench] = []
        used: set[str] = set()
        for _, items in sorted(groups.items()):
            taus = [float(effort[c]) for c, _ in items]
            axes = np.array([f.axis_world for _, f in items], dtype=float)
            net = np.asarray(taus) @ axes
            if float(np.linalg.norm(net)) < max(self._floor, 1e-9):
                continue
            body = items[0][1].body
            label = f"actuator:{_label_part(body)}"
            n = 2
            while label in used:
                label, n = f"actuator:{_label_part(body)}_{n}", n + 1
            used.add(label)
            out.append(
                joint_torque_wrench(
                    label,
                    body,
                    taus,
                    axes,
                    items[0][1].anchor_m,
                    f"{self._engine}:bundle_efforts",
                )
            )
        return out

    def _weight_wrenches(self, coords: Mapping[str, float]) -> list[OverlayWrench]:
        if not self._include_weight:
            return []
        weight = float(self._kin.total_mass_kg) * self._gravity
        return [
            OverlayWrench(
                WrenchKind.GRAVITY,
                "gravity:com",
                "system",
                _tuple3(self._kin.center_of_mass_m(coords)),
                force_n=_tuple3(weight),
                source=f"{self._engine}:system_weight",
            )
        ]
