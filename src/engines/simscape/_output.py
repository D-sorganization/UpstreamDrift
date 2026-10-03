"""Canonical numpy view of one Simscape simulation run.

This module defines :class:`SimscapeOutput`, the immutable result object
returned by :meth:`SimscapeAdapter.simulate_with_coefficients`. The
schema mirrors the headline fields produced by
``simulate_with_coefficients.m`` (issue #018) so the cost function used
by motion-matching can subtract a :class:`SimscapeOutput` from the
target club kinematics directly.

The arrays are deliberately flat ``np.ndarray`` views (no
``Simulink.SimulationData.Dataset`` cross-language marshalling): the
MATLAB-side helper packs ``logsout`` into a flat-double struct, and
:func:`src.engines.simscape._simscape_io.logsout_to_simscape_output`
unpacks it on the Python side.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Final

import numpy as np

from src.engines.simscape.force_channels import (
    DEFAULT_ROTATION_TOL,
    force_series_from_columns,
)
from src.shared.python.force_overlay import ForceTorqueSeries

__all__ = [
    "SimscapeOutput",
]


_TOL_QUAT_NORM: Final[float] = 1e-6
"""Tolerance for the unit-quaternion invariant on ``q_club``."""


@dataclass(frozen=True)
class SimscapeOutput:
    """Flat numpy view of one Simscape simulation run.

    Each row corresponds to one logged sample at the model's fixed
    sample rate (configured in MATLAB via ``set_param`` on the
    Simulink model). The arrays are read-only: the dataclass is frozen
    and consumers should treat the underlying buffers as immutable.

    Invariants (DbC, validated in :meth:`__post_init__`):
        - all arrays share the same ``N`` along axis 0
        - ``time`` is strictly increasing and ``time[0] == 0``
        - ``q``, ``qd``, ``qdd``, ``tau``, ``omega`` share the same
          ``n_joints`` along axis 1
        - ``r_butt``, ``r_clubhead``, ``v_clubhead`` have shape ``(N, 3)``
        - ``q_club`` has shape ``(N, 4)`` and each row is unit-norm to
          :data:`_TOL_QUAT_NORM`

    Args:
        time: Sample times in seconds, shape ``(N,)``.
        q: Joint angles, shape ``(N, n_joints)``.
        qd: Joint velocities, shape ``(N, n_joints)``.
        qdd: Joint accelerations, shape ``(N, n_joints)``.
        tau: Joint torques (N·m), shape ``(N, n_joints)``.
        omega: Joint angular velocities (rad/s), shape ``(N, n_joints)``.
        r_butt: Butt position in metres, shape ``(N, 3)``.
        r_clubhead: Club-head position in metres, shape ``(N, 3)``.
        q_club: Club orientation as unit quaternion ``[w, x, y, z]``,
            shape ``(N, 4)``.
        v_clubhead: Club-head linear velocity in m/s, shape ``(N, 3)``.
        force_columns: Optional logged force/torque channels (#11304),
            keyed by the dataset column names of
            ``SIMSCAPE_FORCE_CHANNELS`` (3x3 rotations flattened to
            ``_I11.._I33``), each a finite ``(N,)`` array. ``None`` (the
            default) means the run carried none; consumers then get an
            explicit error from :meth:`to_force_series`, never zeros.

    Raises:
        TypeError: If any field is not an ``np.ndarray``.
        ValueError: If any shape or invariant is violated.
    """

    time: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray
    omega: np.ndarray
    r_butt: np.ndarray
    r_clubhead: np.ndarray
    q_club: np.ndarray
    v_clubhead: np.ndarray
    force_columns: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        self._check_types()
        n = self._check_time()
        self._check_joint_arrays(n)
        self._check_three_vector_arrays(n)
        self._check_quaternion(n)
        self._check_force_columns(n)

    def _check_types(self) -> None:
        for fld in fields(self):
            if fld.name == "force_columns":
                continue
            value = getattr(self, fld.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"SimscapeOutput.{fld.name} must be np.ndarray, "
                    f"got {type(value).__name__}"
                )

    def _check_time(self) -> int:
        if self.time.ndim != 1:
            raise ValueError(
                f"SimscapeOutput.time must be 1-D, got ndim={self.time.ndim}"
            )
        n = self.time.shape[0]
        if n == 0:
            raise ValueError("SimscapeOutput.time must have at least one sample")
        if n > 1 and not bool(np.all(np.diff(self.time) > 0)):
            raise ValueError("SimscapeOutput.time must be strictly increasing")
        if float(self.time[0]) != 0.0:
            raise ValueError(
                f"SimscapeOutput.time[0] must be 0.0; got {float(self.time[0])}"
            )
        return n

    def _check_joint_arrays(self, n: int) -> None:
        joint_arrays = {
            "q": self.q,
            "qd": self.qd,
            "qdd": self.qdd,
            "tau": self.tau,
            "omega": self.omega,
        }
        n_joints = self.q.shape[1] if self.q.ndim == 2 else -1
        for name, arr in joint_arrays.items():
            if arr.ndim != 2 or arr.shape[0] != n or arr.shape[1] != n_joints:
                raise ValueError(
                    f"SimscapeOutput.{name} must have shape (N={n}, "
                    f"n_joints={n_joints}); got {arr.shape}"
                )

    def _check_three_vector_arrays(self, n: int) -> None:
        for name, arr in (
            ("r_butt", self.r_butt),
            ("r_clubhead", self.r_clubhead),
            ("v_clubhead", self.v_clubhead),
        ):
            if arr.ndim != 2 or arr.shape != (n, 3):
                raise ValueError(
                    f"SimscapeOutput.{name} must have shape (N={n}, 3); got {arr.shape}"
                )

    def _check_quaternion(self, n: int) -> None:
        if self.q_club.ndim != 2 or self.q_club.shape != (n, 4):
            raise ValueError(
                f"SimscapeOutput.q_club must have shape (N={n}, 4); "
                f"got {self.q_club.shape}"
            )
        norms = np.sqrt(np.einsum("ij,ij->i", self.q_club, self.q_club))
        if not np.all(np.abs(norms - 1.0) < _TOL_QUAT_NORM):
            max_dev = float(np.max(np.abs(norms - 1.0)))
            raise ValueError(
                f"SimscapeOutput.q_club rows must be unit-norm to "
                f"{_TOL_QUAT_NORM}; max deviation {max_dev:.3e}"
            )

    def _check_force_columns(self, n: int) -> None:
        if self.force_columns is None:
            return
        if not isinstance(self.force_columns, Mapping):
            raise TypeError("SimscapeOutput.force_columns must be a Mapping or None")
        for name, arr in self.force_columns.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"SimscapeOutput.force_columns[{name!r}] must be np.ndarray, "
                    f"got {type(arr).__name__}"
                )
            if arr.shape != (n,):
                raise ValueError(
                    f"SimscapeOutput.force_columns[{name!r}] must have shape "
                    f"(N={n},); got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"SimscapeOutput.force_columns[{name!r}] must be finite"
                )

    def to_force_series(
        self, *, rotation_tol: float = DEFAULT_ROTATION_TOL
    ) -> tuple[ForceTorqueSeries, tuple[str, ...]]:
        """World-frame force/torque series from the carried channels.

        Delegates to the FTO-18 loader core, so there is a single channel
        table and rotation convention. Returns ``(series, missing)`` exactly
        like ``load_simscape_force_series``.

        Raises:
            ValueError: the run carried no ``force_columns``, or a logged
                rotation is not orthonormal within ``rotation_tol``.
        """
        if self.force_columns is None:
            raise ValueError(
                "SimscapeOutput carries no force_columns; the exporter did "
                "not log force channels for this run"
            )
        columns = {**self.force_columns, "time": self.time}
        return force_series_from_columns(
            columns, rotation_tol=rotation_tol, source_name="SimscapeOutput"
        )

    @property
    def n_samples(self) -> int:
        """Number of time samples (``N``)."""
        return int(self.time.shape[0])

    @property
    def n_joints(self) -> int:
        """Number of joints inferred from ``q.shape[1]``."""
        return int(self.q.shape[1])
