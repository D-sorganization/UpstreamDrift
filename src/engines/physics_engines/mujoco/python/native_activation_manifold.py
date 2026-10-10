"""Bounded native muscle action on a floating-root configuration manifold.

The optimizer state is physical q/v/activation. An independently replayed,
complete MuJoCo integration snapshot remains a separate execution artifact.
Only a self-contained, smooth built-in muscle model with disabled solver
warmstart is admitted; this is not a contact or golfer-model provider.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import crocoddyl
import mujoco as mj
import numpy as np
from defusedxml import ElementTree
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python import native_manifold_calculus
from src.engines.physics_engines.mujoco.python.native_manifold_calculus import (
    difference_jacobians,
    integration_jacobians,
    set_tracking_cost_derivatives,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import _CALLBACKS

Array = NDArray[np.float64]
_PROVIDER_VERSION = "3.8.0"
_DERIVATIVE_EPSILON = 1e-6


def _vector(value: Array, length: int, name: str) -> Array:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {length} finite values")
    return result


def _frozen(value: Array) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _callbacks_absent() -> None:
    if any(getattr(mj, "get_mjcb_" + name)() is not None for name in _CALLBACKS):
        raise ValueError("native callbacks are unsupported")


def _closed_source(path: Path) -> bytes:
    source = path.read_bytes()
    root = ElementTree.fromstring(source)
    for element in root.iter():
        if element.tag in {"include", "plugin"} or any(
            key.lower().endswith(("file", "dir")) for key in element.attrib
        ):
            raise ValueError("native source closure requires inline XML only")
    return source


def _admit_model(model: mj.MjModel) -> None:
    allowed_flags = int(
        mj.mjtDisableBit.mjDSBL_AUTORESET | mj.mjtDisableBit.mjDSBL_WARMSTART
    )
    if (
        mj.__version__ != _PROVIDER_VERSION
        or model.opt.integrator != mj.mjtIntegrator.mjINT_EULER
        or not np.isfinite(model.opt.timestep)
        or model.opt.timestep <= 0
        or model.opt.disableactuator
        or model.opt.disableflags & ~allowed_flags
        or model.nplugin
        or model.nmocap
        or model.nuserdata
        or model.neq
        or model.nflex
        or model.ntendon
        or model.nuser_body
        or model.nuser_jnt
        or model.nuser_geom
        or model.nuser_site
        or model.nuser_tendon
        or model.nuser_actuator
        or model.nq != model.nv + 1
        or model.njnt < 2
        or model.jnt_type[0] != mj.mjtJoint.mjJNT_FREE
        or np.any(
            ~np.isin(
                model.jnt_type[1:],
                (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE),
            )
        )
        or model.na < 1
        or model.na != model.nu
        or np.any(model.geom_contype)
        or np.any(model.geom_conaffinity)
        or np.any(model.jnt_actfrclimited)
    ):
        raise ValueError(
            "native activation action requires smooth contact-free floating muscle model"
        )
    if (
        np.any(model.actuator_dyntype != mj.mjtDyn.mjDYN_MUSCLE)
        or np.any(model.actuator_gaintype != mj.mjtGain.mjGAIN_MUSCLE)
        or np.any(model.actuator_biastype != mj.mjtBias.mjBIAS_MUSCLE)
        or np.any(model.actuator_trntype != mj.mjtTrn.mjTRN_JOINT)
        or np.any(model.actuator_actnum != 1)
        or not np.all(model.actuator_ctrllimited)
        or np.any(model.actuator_forcelimited)
        or not np.isfinite(model.actuator_ctrlrange).all()
    ):
        raise ValueError(
            "native activation action requires bounded built-in muscle laws"
        )


def _law_identity(model: mj.MjModel, names: tuple[str, ...]) -> str:
    fields = (
        "actuator_dyntype",
        "actuator_gaintype",
        "actuator_biastype",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_actadr",
        "actuator_actnum",
        "actuator_dynprm",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_gear",
        "actuator_ctrlrange",
        "actuator_lengthrange",
        "actuator_forcerange",
    )
    payload = {name: np.asarray(getattr(model, name)).tolist() for name in fields}
    payload["ordered_names"] = names
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _compiled_hash(model: mj.MjModel) -> str:
    binary = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=binary)
    return hashlib.sha256(binary.tobytes()).hexdigest()


@dataclass(frozen=True)
class NativeActivationIdentity:
    """Ordered command/law and exact compiled/runtime model identity."""

    source_model_sha256: str
    loaded_native_model_sha256: str
    compiled_law_sha256: str
    provider_sha256: str
    adapter_sha256: str
    provider_version: str
    ordered_input_channel_ids: tuple[str, ...]
    input_kind: str
    source_closure: str
    history_policy: str
    time_step_s: float

    def __post_init__(self) -> None:
        for name in (
            "source_model_sha256",
            "loaded_native_model_sha256",
            "compiled_law_sha256",
            "provider_sha256",
            "adapter_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be SHA-256")
        if (
            self.provider_version != _PROVIDER_VERSION
            or self.input_kind != "actuator_command"
            or self.source_closure != "self_contained_xml"
            or self.history_policy
            != "warmstart-disabled;autoreset-disabled;time-invariant;no-constraints"
            or not np.isfinite(self.time_step_s)
            or self.time_step_s <= 0
            or not self.ordered_input_channel_ids
            or any(not name for name in self.ordered_input_channel_ids)
            or len(set(self.ordered_input_channel_ids))
            != len(self.ordered_input_channel_ids)
        ):
            raise ValueError("native actuator identity or execution policy invalid")


class NativeActivationState(crocoddyl.StateAbstract):
    """Physical ``[qpos,qvel,activation]`` with a native q tangent."""

    def __init__(self, model: mj.MjModel) -> None:
        super().__init__(model.nq + model.nv + model.na, 2 * model.nv + model.na)
        self.model = model

    def checked(self, x: Array) -> Array:
        state = _vector(x, self.nx, "native physical state")
        normalized = state[: self.model.nq].copy()
        mj.mj_normalizeQuat(self.model, normalized)
        if not np.allclose(state[: self.model.nq], normalized, atol=1e-12, rtol=0):
            raise ValueError("native quaternion must be normalized")
        return state

    def diff(self, x0: Array, x1: Array) -> Array:
        before, after = self.checked(x0), self.checked(x1)
        nq, nv = self.model.nq, self.model.nv
        dq = np.empty(nv)
        mj.mj_differentiatePos(self.model, dq, 1.0, before[:nq], after[:nq])
        return np.r_[
            dq,
            after[nq : nq + nv] - before[nq : nq + nv],
            after[nq + nv :] - before[nq + nv :],
        ]

    def integrate(self, x: Array, dx: Array) -> Array:
        base = self.checked(x)
        tangent = _vector(dx, self.ndx, "native physical tangent")
        nq, nv = self.model.nq, self.model.nv
        qpos = base[:nq].copy()
        mj.mj_integratePos(self.model, qpos, tangent[:nv], 1.0)
        return np.r_[
            qpos,
            base[nq : nq + nv] + tangent[nv : 2 * nv],
            base[nq + nv :] + tangent[2 * nv :],
        ]

    def _jacobian(self, function: Any) -> Array:
        eye = np.eye(self.ndx) * _DERIVATIVE_EPSILON
        return np.column_stack(
            [(function(d) - function(-d)) / (2 * _DERIVATIVE_EPSILON) for d in eye]
        )

    def Jdiff(
        self, x0: Array, x1: Array, firstsecond: Any = crocoddyl.Jcomponent.both
    ) -> list[Array]:
        before, after = self.checked(x0), self.checked(x1)

        return difference_jacobians(self, before, after, firstsecond)

    def Jintegrate(
        self, x: Array, dx: Array, firstsecond: Any = crocoddyl.Jcomponent.both
    ) -> list[Array]:
        base, tangent = self.checked(x), _vector(dx, self.ndx, "native tangent")
        return integration_jacobians(self, base, tangent, firstsecond)

    def JintegrateTransport(
        self, x: Array, dx: Array, Jin: Array, firstsecond: Any
    ) -> Array:
        jacobian = self.Jintegrate(x, dx, firstsecond)[0]
        return np.asarray(np.linalg.solve(jacobian, np.asarray(Jin, dtype=np.float64)))

    def zero(self) -> Array:
        data = mj.MjData(self.model)
        data.act[:] = 0.5
        return np.r_[data.qpos, data.qvel, data.act]

    def rand(self) -> Array:
        tangent = (
            np.asarray(np.random.default_rng().normal(size=self.ndx), dtype=np.float64)
            * 0.05
        )
        return self.integrate(self.zero(), tangent)


@dataclass(frozen=True)
class NativeActivationStep:
    """Native Euler derivative in q/v/activation tangent coordinates."""

    A: Array
    B: Array
    next_physical_state: Array

    def __post_init__(self) -> None:
        matrix = np.asarray(self.A, dtype=np.float64)
        inputs = np.asarray(self.B, dtype=np.float64)
        next_state = np.asarray(self.next_physical_state, dtype=np.float64)
        if (
            matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or inputs.ndim != 2
            or inputs.shape[0] != matrix.shape[0]
            or next_state.shape != (matrix.shape[0] + 1,)
            or not np.isfinite(matrix).all()
            or not np.isfinite(inputs).all()
            or not np.isfinite(next_state).all()
        ):
            raise ValueError(
                "native activation derivative dimensions or values invalid"
            )
        object.__setattr__(self, "A", _frozen(matrix))
        object.__setattr__(self, "B", _frozen(inputs))
        object.__setattr__(self, "next_physical_state", _frozen(next_state))


class NativeActivationProvider:
    """Admitted smooth compiled model and exact physical/native step policy."""

    def __init__(
        self, path: Path, model: mj.MjModel, identity: NativeActivationIdentity
    ):
        self.path, self.model, self.identity = path, model, identity
        self.state = NativeActivationState(model)

    def project_physical(self, full_state: Array) -> Array:
        full = _vector(
            full_state,
            mj.mj_stateSize(self.model, mj.mjtState.mjSTATE_INTEGRATION),
            "complete native state",
        )
        data = mj.MjData(self.model)
        mj.mj_setState(self.model, data, full, mj.mjtState.mjSTATE_INTEGRATION)
        readback = np.empty_like(full)
        mj.mj_getState(self.model, data, readback, mj.mjtState.mjSTATE_INTEGRATION)
        if not np.array_equal(full, readback):
            raise ValueError("complete native state did not restore")
        return self.state.checked(np.r_[data.qpos, data.qvel, data.act])

    def _admit(self, data: mj.MjData, command: Array) -> Array:
        controls = _vector(command, self.model.nu, "native actuator command")
        low, high = (
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        if np.any(controls <= low + _DERIVATIVE_EPSILON) or np.any(
            controls >= high - _DERIVATIVE_EPSILON
        ):
            raise ValueError("native actuator command must remain interior")
        for joint in range(self.model.njnt):
            if self.model.jnt_limited[joint]:
                pos = data.qpos[self.model.jnt_qposadr[joint]]
                lower, upper = self.model.jnt_range[joint]
                if not lower + _DERIVATIVE_EPSILON < pos < upper - _DERIVATIVE_EPSILON:
                    raise ValueError("native joint limit blocks centered derivative")
        if np.any(data.act <= _DERIVATIVE_EPSILON) or np.any(
            data.act >= 1 - _DERIVATIVE_EPSILON
        ):
            raise ValueError("native activation must remain interior")
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError("external native loads are unsupported")
        _callbacks_absent()
        return controls

    def _physical_data(self, physical: Array) -> mj.MjData:
        state = self.state.checked(physical)
        data = mj.MjData(self.model)
        nq, nv = self.model.nq, self.model.nv
        data.qpos[:] = state[:nq]
        data.qvel[:] = state[nq : nq + nv]
        data.act[:] = state[nq + nv :]
        return data

    def _finish_step(self, data: mj.MjData, before_time: float) -> Array:
        _callbacks_absent()
        options = self.model.opt
        if (
            data.ncon
            or data.nefc
            or data.time != before_time + options.timestep
            or data.time <= before_time
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isfinite(data.act).all()
        ):
            raise ValueError("native step contacted, constrained, reset or diverged")
        self._admit(data, data.ctrl)
        if (
            hashlib.sha256(self.path.read_bytes()).hexdigest()
            != self.identity.source_model_sha256
        ):
            raise ValueError("native source changed after compilation")
        if _compiled_hash(self.model) != self.identity.loaded_native_model_sha256:
            raise ValueError("native compiled model identity changed")
        return self.state.checked(np.r_[data.qpos, data.qvel, data.act])

    def step(self, physical: Array, command: Array) -> Array:
        data = self._physical_data(physical)
        data.ctrl[:] = self._admit(data, command)
        before = float(data.time)
        mj.mj_step(self.model, data)
        return self._finish_step(data, before)

    def step_full(self, full_state: Array, command: Array) -> Array:
        self.project_physical(full_state)
        data = mj.MjData(self.model)
        mj.mj_setState(self.model, data, full_state, mj.mjtState.mjSTATE_INTEGRATION)
        data.ctrl[:] = self._admit(data, command)
        before = float(data.time)
        mj.mj_step(self.model, data)
        self._finish_step(data, before)
        result = np.empty_like(full_state)
        mj.mj_getState(self.model, data, result, mj.mjtState.mjSTATE_INTEGRATION)
        return result

    def linearize(self, physical: Array, command: Array) -> NativeActivationStep:
        data = self._physical_data(physical)
        data.ctrl[:] = self._admit(data, command)
        _callbacks_absent()
        mj.mj_forward(self.model, data)
        _callbacks_absent()
        if data.ncon or data.nefc or not np.isfinite(data.qacc).all():
            raise ValueError("active native constraint blocks derivative")
        A = np.empty((self.state.ndx, self.state.ndx))
        B = np.empty((self.state.ndx, self.model.nu))
        mj.mjd_transitionFD(self.model, data, _DERIVATIVE_EPSILON, 1, A, B, None, None)
        _callbacks_absent()
        next_state = self.step(physical, command)
        if not np.isfinite(A).all() or not np.isfinite(B).all():
            raise ValueError("native derivative is nonfinite")
        return NativeActivationStep(A, B, next_state)


def load_native_activation_provider(model_path: str | Path) -> NativeActivationProvider:
    """Load only a self-contained smooth MuJoCo 3.8 muscle model."""
    path = Path(model_path).resolve(strict=True)
    if mj.__version__ != _PROVIDER_VERSION:
        raise ValueError("native activation requires MuJoCo 3.8.0")
    _callbacks_absent()
    source = _closed_source(path)
    model = mj.MjModel.from_xml_path(str(path))
    if path.read_bytes() != source:
        raise ValueError("native source changed during loading")
    _admit_model(model)
    model.opt.disableflags |= int(
        mj.mjtDisableBit.mjDSBL_AUTORESET | mj.mjtDisableBit.mjDSBL_WARMSTART
    )
    names = tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    )
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("native actuator channels need unique names")
    runtime = importlib.import_module("mujoco._functions")
    if runtime.__file__ is None:
        raise ValueError("native provider binary is unavailable")
    identity = NativeActivationIdentity(
        hashlib.sha256(source).hexdigest(),
        _compiled_hash(model),
        _law_identity(model, names),
        hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),
        _adapter_source_hash(),
        mj.__version__,
        names,
        "actuator_command",
        "self_contained_xml",
        "warmstart-disabled;autoreset-disabled;time-invariant;no-constraints",
        float(model.opt.timestep),
    )
    return NativeActivationProvider(path, model, identity)


def _adapter_source_hash() -> str:
    """Bind the adapter and the shared native tangent/cost implementation."""
    sources = (Path(__file__), Path(native_manifold_calculus.__file__))
    closure = {
        source.name: hashlib.sha256(source.read_bytes()).hexdigest()
        for source in sources
    }
    return hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest()


class NativeActivationAction(crocoddyl.ActionModelAbstract):
    """One admitted native step and tangent tracking cost for BoxFDDP."""

    def __init__(self, provider: NativeActivationProvider, reference: Array):
        super().__init__(provider.state, provider.model.nu, provider.state.ndx)
        self.provider = provider
        self.command_count = provider.model.nu
        self.native_state = provider.state
        self.reference = provider.state.checked(reference).copy()
        self.state_weights = np.ones(provider.state.ndx)
        self.input_weights = np.full(provider.model.nu, 0.01)
        self.u_lb = provider.model.actuator_ctrlrange[:, 0] + _DERIVATIVE_EPSILON
        self.u_ub = provider.model.actuator_ctrlrange[:, 1] - _DERIVATIVE_EPSILON

    def calc(self, data: Any, x: Array, u: Array) -> None:
        command = _vector(u, self.command_count, "native command")
        data.xnext = self.provider.step(x, command)
        error = self.native_state.diff(self.reference, data.xnext)
        data.cost = float(
            np.dot(self.state_weights, error**2)
            + np.dot(self.input_weights, command**2)
        )

    def calcDiff(self, data: Any, x: Array, u: Array) -> None:
        command = _vector(u, self.command_count, "native command")
        derivative = self.provider.linearize(x, command)
        if (
            np.max(
                np.abs(
                    self.native_state.diff(derivative.next_physical_state, data.xnext)
                )
            )
            > 1e-10
        ):
            raise ValueError("native action and derivative disagree")
        data.Fx, data.Fu = derivative.A, derivative.B
        set_tracking_cost_derivatives(
            data,
            self.native_state,
            self.reference,
            self.state_weights,
            command,
            self.input_weights,
        )


def replay_native_activation_commands(
    model_path: str | Path,
    initial_integration_state: Array,
    commands: Array,
    expected_identity: NativeActivationIdentity,
) -> Array:
    """Reload the model and replay exact complete state with time-only commands."""
    provider = load_native_activation_provider(model_path)
    if provider.identity != expected_identity:
        raise ValueError("native actuator/model identity changed before replay")
    controls = np.asarray(commands, dtype=np.float64)
    if (
        controls.ndim != 2
        or controls.shape[1] != provider.model.nu
        or not np.isfinite(controls).all()
    ):
        raise ValueError("ordered native commands must be finite and two-dimensional")
    full = _vector(
        initial_integration_state,
        mj.mj_stateSize(provider.model, mj.mjtState.mjSTATE_INTEGRATION),
        "complete native initial state",
    )
    provider.project_physical(full)
    history = [full.copy()]
    for command in controls:
        history.append(provider.step_full(history[-1], command))
    return np.stack(history)
