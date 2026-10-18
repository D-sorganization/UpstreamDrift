"""Engine-agnostic overlay frame provider for same-input bundles (NV-3, #11676).

Builds :class:`ForceTorqueFrame` series from a same-input bundle without
touching any engine SDK:

* bundle efforts -> one joint-torque wrench per joint anchor (coordinates that
  share an anchor, e.g. the three hip rotations, merge into one vector);
* the shared contact law (``evaluate_contact_samples`` of any full-body model)
  -> the ground-reaction breakdown of the shared core (GCV-1, #11707; GCV-2,
  #11708): per-foot and net force at the wrench-derived centre of pressure,
  free moment and moment about the centre of mass;
* the centre of mass -> the system weight, as a gravity wrench at the CoM.

Engine adapters only have to supply the two small protocols below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any, NamedTuple, Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.mujoco_axial_loads import (
    ExternalWrench,
    GripClosure,
)
from src.shared.python.biomechanics.grip_extraction import unavailable_analysis
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.biomechanics.ground_reaction_wrenches import (
    ground_reaction_overlay,
)
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.conversions import joint_torque_wrench
from src.shared.python.force_overlay.grip_frame import grip_wrenches_and_metadata
from src.shared.python.force_overlay.series import ForceTorqueSeries
from src.shared.python.motion_matching.same_input import InputBundle

Array: TypeAlias = NDArray[np.float64]
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


def _tuple3(vec: Sequence[float] | Array) -> Vec3:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


@runtime_checkable
class GripSource(Protocol):
    """An engine adapter that can report the hands' wrench on the club (GCV-8)."""

    def grip_analysis(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> GripAnalysis:
        """Per-hand wrenches ON THE CLUB at the given state."""
        ...


@runtime_checkable
class AxialLoadSource(Protocol):
    """An engine adapter that can report segment axial loads (tension positive)."""

    def axial_load_frame(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        time_s: float,
        external_wrenches: Sequence[ExternalWrench] | None = None,
        closure: GripClosure | None = None,
    ) -> AxialLoadFrame | None:
        """Axial loads for a prescribed motion, or ``None`` when unqualified.

        ``external_wrenches`` are the ground contact wrenches the provider also
        draws (one source of truth for the support reaction); ``None`` leaves
        the engine's own contact handling in charge.
        """
        ...


@runtime_checkable
class ClosureAxialLoadSource(AxialLoadSource, Protocol):
    """An axial-load source that closes a rigid grip loop with a physical wrench.

    ``closure_wrench`` resolves the weld multiplier from the applied efforts; its
    result is passed back to ``axial_load_frame`` as ``closure`` so the loop is
    not closed by a soft constraint's impedance (#13179).
    """

    def closure_wrench(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float] | None,
    ) -> GripClosure:
        """Closure wrench at the state, or an unavailable closure with a reason."""
        ...

    def axial_load_frame(  # type: ignore[override]
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        time_s: float,
        external_wrenches: Sequence[OverlayWrench] | None = None,
        closure: GripClosure | None = None,
    ) -> AxialLoadFrame | None:
        """As :meth:`AxialLoadSource.axial_load_frame`, with the closure wrench."""
        ...


_GRIP_FAILURES = (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError)


def sphere_bodies(spec: Mapping[str, Any]) -> dict[str, str]:
    """Contact-sphere name to spec body name, from a full-body spec document."""
    spheres = spec.get("contact", {}).get("spheres", ())
    return {sphere["name"]: sphere["body"] for sphere in spheres}


def contact_sample_wrenches(
    samples: Mapping[str, Any],
    sphere_body: Mapping[str, str],
    *,
    source: str,
    free_moments: Mapping[str, Any] | None = None,
) -> list[OverlayWrench]:
    """Loaded shared-law contact samples as ground ``CONTACT`` wrenches.

    ``samples`` maps sphere name to a ``ContactSample``; ``sphere_body`` maps
    sphere name to its spec body (default: the name's last ``_`` part).
    Postcondition: unloaded spheres (``|F| <= 1e-9 N``) are omitted, so a foot
    off the ground has no wrench rather than a zero one. ``free_moments`` (sphere
    name to world-frame moment about the contact normal, N m) is carried as the
    wrench's ``torque_nm`` so the ground free moment is not dropped (#13179);
    ``None`` or a missing sphere leaves the wrench torque-free.
    """
    ground: list[OverlayWrench] = []
    for name, sample in samples.items():
        force = np.asarray(sample.normal_force_n) + np.asarray(sample.friction_force_n)
        if float(np.linalg.norm(force)) <= 1e-9:
            continue
        ground.append(
            OverlayWrench(
                WrenchKind.CONTACT,
                f"contact:{_label_part(name)}",
                sphere_body.get(name, name.rsplit("_", 1)[-1]),
                _tuple3(np.asarray(sample.contact_point_m, float)),
                force_n=_tuple3(force),
                torque_nm=(
                    _tuple3(free_moments[name])
                    if free_moments and name in free_moments
                    else None
                ),
                source=source,
            )
        )
    return ground


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
        self._grip: GripSource | None = None
        self._axial: AxialLoadSource | None = None
        self._axial_impact: tuple[float, float] | None = None
        self._accelerations: Array | None = None
        self._grip_cache: dict[int, GripAnalysis] = {}
        self._grip_source = f"{engine}:grip"
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
        self._sphere_body = sphere_bodies(spec)

    def with_grip(
        self, grip: GripSource, source: str | None = None
    ) -> BundleOverlayProvider:
        """Add the hands' wrench on the club (GCV-10); returns ``self``.

        ``source`` labels where the extraction came from (frame metadata
        ``grip_source``).  Raises ``TypeError`` if ``grip`` has no
        ``grip_analysis``.
        """
        if not isinstance(grip, GripSource):
            raise TypeError("grip must provide grip_analysis(coords, rates, efforts)")
        self._grip = grip
        self._grip_cache.clear()
        if source:
            self._grip_source = source
        return self

    def with_axial_loads(self, axial: AxialLoadSource) -> BundleOverlayProvider:
        """Attach per-frame segment axial loads (``frame.axial_loads``); returns ``self``.

        Kinematics are exactly those of the torque/contact wrenches (the provider's
        ``q`` and ``v``); accelerations are the central finite difference of ``v``
        on the bundle time grid (one-sided at the ends), unfiltered. Raises
        ``TypeError`` if ``axial`` has no ``axial_load_frame``.
        """
        if not isinstance(axial, AxialLoadSource):
            raise TypeError("axial must provide axial_load_frame(q, v, a, time_s)")
        self._axial = axial
        self._accelerations = None
        return self

    def with_axial_impact_window(
        self, impact_time_s: float, half_width_s: float
    ) -> BundleOverlayProvider:
        """Report axial loads as unavailable around ball impact; returns ``self``.

        The ball impact impulse is not modelled, so an open-loop segment load
        there (the root residual reaches about 3 kN) must not be shown. Frames
        with ``|t - impact_time_s| <= half_width_s`` carry ``None`` for every
        segment (never zero) and ``axial_impact: unavailable`` metadata.
        Raises ValueError for a non-finite time or a negative/non-finite width.
        """
        if not math.isfinite(impact_time_s):
            raise ValueError(f"impact_time_s must be finite, got {impact_time_s!r}")
        if not math.isfinite(half_width_s) or half_width_s < 0:
            raise ValueError(
                f"half_width_s must be >= 0 and finite, got {half_width_s!r}"
            )
        self._axial_impact = (float(impact_time_s), float(half_width_s))
        return self

    def _axial_frame(
        self,
        index: int,
        coords: Mapping[str, float],
        rates: Mapping[str, float],
        ground: Sequence[OverlayWrench],
        efforts: Mapping[str, float] | None = None,
    ) -> tuple[AxialLoadFrame | None, dict[str, Any]]:
        """Axial loads at ``index`` plus the closure status metadata.

        A source with ``closure_wrench`` gets the physical grip closure for the
        state's efforts; its status (``axial_closure``: ``applied`` or
        ``unavailable``, with ``axial_closure_reason`` and the world-frame 6-D
        ``axial_closure_multiplier``) is reported so shading can say why a load
        is blank. Sources without it are called exactly as before.
        """
        if self._axial is None:
            return None, {}
        if self._accelerations is None:
            self._accelerations = np.gradient(self._v, float(self._bundle.dt_s), axis=0)
        accel = dict(
            zip(
                self._bundle.coordinate_order,
                map(float, self._accelerations[index]),
                strict=True,
            )
        )
        extra: dict[str, Any] = {}
        kwargs: dict[str, Any] = {}
        if isinstance(self._axial, ClosureAxialLoadSource):
            closure = self._axial.closure_wrench(coords, rates, efforts)
            kwargs["closure"] = closure
            if closure.available:
                extra["axial_closure"] = "applied"
                extra["axial_closure_multiplier"] = list(closure.multiplier_world or ())
            else:
                extra["axial_closure"] = "unavailable"
                extra["axial_closure_reason"] = closure.reason
        frame = self._axial.axial_load_frame(
            coords,
            rates,
            accel,
            float(index * self._bundle.dt_s),
            external_wrenches=cast(Sequence[ExternalWrench], ground),
            **kwargs,
        )
        if frame is not None and self._in_impact_window(frame.time_s):
            frame = AxialLoadFrame(
                frame.time_s,
                dict.fromkeys(frame.values_n),
                f"{frame.source}; unavailable: ball impact impulse is not modelled",
            )
            extra["axial_impact"] = "unavailable"
        return frame, extra

    def _in_impact_window(self, time_s: float) -> bool:
        if self._axial_impact is None:
            return False
        centre, half = self._axial_impact
        return abs(time_s - centre) <= half + 1e-12

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
        named_efforts = dict(zip(names, efforts, strict=True))
        ground = self._ground_wrenches(coords, rates)
        wrenches = [
            *self._contact_wrenches(ground, coords),
            *self._torque_wrenches(coords, named_efforts),
            *self._weight_wrenches(coords),
        ]
        metadata: dict[str, Any] = {
            "source": "same_input_bundle",
            "spec_sha256": self._bundle.spec_sha256,
        }
        grip_wrenches, grip_meta = self._grip_wrenches(
            index, coords, rates, named_efforts
        )
        wrenches.extend(grip_wrenches)
        metadata.update(grip_meta)
        axial, axial_meta = self._axial_frame(
            index, coords, rates, ground, named_efforts
        )
        metadata.update(axial_meta)
        return ForceTorqueFrame(
            time_s=float(index * self._bundle.dt_s),
            engine=self._engine,
            wrenches=tuple(wrenches),
            axial_loads=axial,
            metadata=metadata,
        )

    def grip_analyses(self, indices: Sequence[int]) -> list[GripAnalysis]:
        """Grip analyses at ``indices`` (cached; needs a ``grip`` source)."""
        if self._grip is None:
            raise ValueError("this provider was built without a grip source")
        for k in indices:
            self.frame_at(k)
        return [self._grip_cache[k] for k in indices]

    def _grip_wrenches(
        self,
        index: int,
        coords: Mapping[str, float],
        rates: Mapping[str, float],
        efforts: Mapping[str, float],
    ) -> tuple[tuple[OverlayWrench, ...], dict[str, Any]]:
        """Grip wrenches and metadata; a failed solve is unavailable with a reason."""
        if self._grip is None:
            return (), {}
        analysis = self._grip_cache.get(index)
        if analysis is None:
            try:
                analysis = self._grip.grip_analysis(coords, rates, efforts)
            except _GRIP_FAILURES as exc:
                analysis = unavailable_analysis(f"grip solve failed: {exc}")
            self._grip_cache[index] = analysis
        wrenches, meta = grip_wrenches_and_metadata(analysis, source=self._grip_source)
        return wrenches, {**meta, "grip_source": self._grip_source}

    def series(self, stride: int = 1) -> ForceTorqueSeries:
        """Frames at every ``stride``-th state."""
        if not isinstance(stride, int) or stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride!r}")
        frames = tuple(self.frame_at(k) for k in range(0, len(self), stride))
        return ForceTorqueSeries(frames=frames, engine=self._engine)

    def _ground_wrenches(
        self, coords: Mapping[str, float], rates: Mapping[str, float]
    ) -> list[OverlayWrench]:
        """Loaded shared-law contact spheres as per-sphere ground wrenches.

        The one evaluation of the contact law per frame: it feeds both the
        ground-reaction glyphs and the axial-load inverse dynamics.
        """
        samples = self._contact.evaluate_contact_samples(coords, rates)
        moments = getattr(self._contact, "contact_free_moments", None)
        return contact_sample_wrenches(
            samples,
            self._sphere_body,
            source=f"{self._engine}:shared_contact_law",
            free_moments=moments(coords, rates) if moments else None,
        )

    def _contact_wrenches(
        self, ground: Sequence[OverlayWrench], coords: Mapping[str, float]
    ) -> list[OverlayWrench]:
        """Per-foot and net GRF breakdown of the shared contact law (GCV-2, GCV-6).

        Loaded contact spheres become ground-contact wrenches on their body;
        the shared ground-reaction core groups them by foot and reports the
        per-foot and net force, CoP, free moment and moment about the CoM.
        """
        return list(
            ground_reaction_overlay(
                list(ground),
                self._kin.center_of_mass_m(coords),
                source=f"{self._engine}:shared_contact_law",
            )
        )

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
