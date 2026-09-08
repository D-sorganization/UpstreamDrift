"""``SwingBioModel``: bioptim's custom-model protocol over the swing (Phase 1.2).

Implements ``bioptim.StateDynamics`` (torque-driven, ``[q, qdot]`` states,
``tau`` controls) plus the ``BioModel`` surface bioptim's penalty library
calls on a model -- markers, marker velocities, centre of mass, mass,
gravity, ``tau_max`` -- all delegated to :class:`SymbolicSwingModel`.

bioptim is imported lazily through :func:`_compat.require_bioptim` so this
module imports without the optional extra; :func:`make_swing_bio_model` is
the constructor callers use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from src.shared.python.optimization._swing_kinematics import JOINTS
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.casadi_backend import require_casadi
from src.shared.python.optimization.ocp._compat import require_bioptim
from src.shared.python.optimization.ocp.symbolic_model import SymbolicSwingModel

__all__ = ["make_swing_bio_model"]  # SwingBioModel is provided lazily via __getattr__

_NO_CONTACTS = (
    "the seven-DOF swing rig is a fixed-base chain with no contact points; "
    "with_contact=True is not a valid request for this model"
)

_CLASS_CACHE: dict[str, type] = {}


class _SwingModelSurface:
    """The half of the adapter that does not need bioptim to be defined.

    Everything here either delegates to :class:`SymbolicSwingModel` or is a
    constant of the rig, so it can live at module level and be unit-tested
    without the optional extra. Only the pieces that genuinely close over
    bioptim symbols -- the configuration functions and ``dynamics`` -- stay
    inside :func:`_build_class`, which keeps that factory small.

    These attributes are set by the concrete subclass's ``__init__``;
    declaring them here is what lets mypy check the delegating methods.
    """

    symbolic: SymbolicSwingModel
    golfer: GolferModel
    club: ClubModel
    _parameter_names: tuple[str, ...]

    @property
    def name(self) -> str:
        return "UpstreamDriftSwing7DOF"

    @property
    def name_dofs(self) -> list[str]:
        return list(JOINTS)

    # -- sizes -------------------------------------------------------------

    @property
    def nb_q(self) -> int:
        return self.symbolic.n_q

    @property
    def nb_qdot(self) -> int:
        return self.symbolic.n_q

    @property
    def nb_qddot(self) -> int:
        return self.symbolic.n_q

    @property
    def nb_tau(self) -> int:
        return self.symbolic.n_q

    @property
    def nb_dof(self) -> int:
        return self.symbolic.n_q

    @property
    def nb_root(self) -> int:
        return 0

    @property
    def nb_quaternions(self) -> int:
        return 0

    @property
    def nb_parameters(self) -> int:
        return self.symbolic.n_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    # -- kinematics used by penalties -------------------------------------

    @property
    def marker_names(self) -> tuple[str, ...]:
        return self.symbolic.marker_names

    @property
    def nb_markers(self) -> int:
        return self.symbolic.n_markers

    def marker_index(self, name: str) -> int:
        return self.symbolic.marker_index(name)

    def markers(self) -> Any:
        return self.symbolic.markers

    def markers_velocities(self, reference_index: Any = None) -> Any:
        if reference_index is not None:
            # Genuinely unimplemented: the symbolic model expresses marker
            # velocities in the world frame only. Scope is set by the epic.
            raise NotImplementedError(  # tracked: #9762
                "marker velocities in a segment frame are not supported; "
                "omit reference_index to get them in the world frame"
            )
        return self.symbolic.markers_velocities

    def center_of_mass(self) -> Any:
        return self.symbolic.center_of_mass

    def center_of_mass_velocity(self) -> Any:
        return self.symbolic.center_of_mass_velocity

    def mass(self) -> Any:
        return self.symbolic.total_mass

    def gravity(self) -> Any:
        return self.symbolic.gravity

    # -- dynamics used by penalties ---------------------------------------

    # ``with_contact`` is part of bioptim's model protocol, but the swing
    # rig is a fixed-base chain with no contact points at all, so asking
    # for contact dynamics is an invalid argument rather than a feature
    # this adapter has yet to implement.
    def forward_dynamics(self, with_contact: bool = False) -> Any:
        if with_contact:
            raise ValueError(_NO_CONTACTS)
        return self.symbolic.forward_dynamics

    def inverse_dynamics(self, with_contact: bool = False) -> Any:
        if with_contact:
            raise ValueError(_NO_CONTACTS)
        return self.symbolic.rnea

    def tau_max(self) -> Any:
        """``(tau_max, tau_min)(q, qdot, parameters)`` from the golfer."""
        ca = require_casadi()
        limits = self.symbolic.torque_limits()
        p = self.symbolic.parameter_symbols()
        q = ca.SX.sym("q", self.nb_q)
        qdot = ca.SX.sym("qdot", self.nb_q)
        return ca.Function(
            "swing_tau_max",
            [q, qdot, p],
            [ca.SX(limits), ca.SX(-limits)],
            ["q", "qdot", "parameters"],
            ["tau_max", "tau_min"],
        )

    # -- bookkeeping bioptim expects ---------------------------------------

    def copy(self) -> Any:
        # The concrete subclass built by _build_class supplies the
        # ``(golfer, club, *, parameters)`` constructor; this mixin
        # deliberately has none, so the factory is untyped here.
        factory = cast(Any, type(self))
        return factory(self.golfer, self.club, parameters=self._parameter_names)

    def serialize(self) -> tuple[Any, dict[str, Any]]:
        return type(self), {
            "golfer": self.golfer,
            "club": self.club,
            "parameters": self._parameter_names,
        }

    def set_parameter(self, value: Any, **kwargs: Any) -> None:
        """``ParameterList.add(..., function=model.set_parameter)`` hook.

        bioptim calls this with the parameter's symbol at build time.
        Nothing to mutate: every function reads the OCP parameter vector
        directly, so the hook only records the symbol for inspection.
        """
        self.last_parameter_symbol = value

    def bounds_from_ranges(self, key: str, flexibility: float = 1.0) -> Any:
        """``Bounds`` for ``"q"`` (golfer ROMs) or ``"qdot"``."""
        from src.shared.python.optimization.model_provider import swing_joint_limits

        if key == "q":
            limits = swing_joint_limits(self.golfer)
            lower = np.array([limits[j][0] for j in JOINTS]) * flexibility
            upper = np.array([limits[j][1] for j in JOINTS]) * flexibility
        elif key == "qdot":
            lower = np.full(self.nb_q, -40.0)
            upper = np.full(self.nb_q, 40.0)
        else:
            raise KeyError(key)
        return require_bioptim().Bounds(key, min_bound=lower, max_bound=upper)


def _build_class() -> type:
    """Define the adapter class once bioptim is importable.

    The base class (``bioptim.StateDynamics``) only exists when bioptim is
    installed, so the class body lives in a function. Only the members that
    close over ``bioptim`` or ``casadi`` symbols are defined here; the rest
    of the protocol surface is inherited from :class:`_SwingModelSurface`.
    """
    bioptim = require_bioptim()
    ca = require_casadi()

    class SwingBioModel(_SwingModelSurface, bioptim.StateDynamics):  # type: ignore[misc,name-defined]
        """Torque-driven ``StateDynamics`` over :class:`SymbolicSwingModel`.

        Args:
            golfer, club: The numeric model.
            parameters: Names promoted to bioptim ``Parameter``s, in the order
                they will be added to the OCP's ``ParameterList`` (Phase 4).
                Every CasADi function then reads them from bioptim's
                ``parameters`` vector.
        """

        def __init__(
            self,
            golfer: GolferModel | None = None,
            club: ClubModel | None = None,
            *,
            parameters: Sequence[str] = (),
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.symbolic = SymbolicSwingModel(golfer, club, parameters=parameters)
            self.golfer = self.symbolic.golfer
            self.club = self.symbolic.club
            self._parameter_names = tuple(parameters)
            self._q = ca.MX.sym("q", self.nb_q)
            self._qdot = ca.MX.sym("qdot", self.nb_q)
            self._tau = ca.MX.sym("tau", self.nb_q)

        @property
        def state_configuration_functions(self) -> list[Any]:
            return [bioptim.States.Q, bioptim.States.QDOT]

        @property
        def control_configuration_functions(self) -> list[Any]:
            return [bioptim.Controls.TAU]

        @property
        def algebraic_configuration_functions(self) -> list[Any]:
            return []

        @property
        def extra_configuration_functions(self) -> list[Any]:
            return []

        def dynamics(
            self,
            time: Any,
            states: Any,
            controls: Any,
            parameters: Any,
            algebraic_states: Any,
            numerical_timeseries: Any,
            nlp: Any,
        ) -> Any:
            get = bioptim.DynamicsFunctions.get
            q = get(nlp.states["q"], states)
            qdot = get(nlp.states["qdot"], states)
            tau = get(nlp.controls["tau"], controls)
            qddot = self.symbolic.forward_dynamics(q, qdot, tau, parameters)
            defects = None
            ode_solver = nlp.dynamics_type.ode_solver
            if isinstance(ode_solver, bioptim.OdeSolver.COLLOCATION):
                # Direct collocation needs implicit defects: the polynomial
                # slopes must match the dynamics at every collocation point.
                slope_q = nlp.states_dot["q"].cx
                slope_qdot = nlp.states_dot["qdot"].cx
                if (
                    ode_solver.defects_type
                    == bioptim.DefectType.TAU_EQUALS_INVERSE_DYNAMICS
                ):
                    tau_id = self.symbolic.rnea(q, qdot, slope_qdot, parameters)
                    defects = ca.vertcat(slope_q - qdot, tau - tau_id)
                else:
                    defects = ca.vertcat(slope_q - qdot, slope_qdot - qddot)
            return bioptim.DynamicsEvaluation(
                dxdt=ca.vertcat(qdot, qddot), defects=defects
            )

    return SwingBioModel


def make_swing_bio_model(
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    parameters: Sequence[str] = (),
) -> Any:
    """Instantiate :class:`SwingBioModel` (defines the class on first use).

    Raises:
        BioptimNotAvailableError: When bioptim is not installed.
    """
    cls = _CLASS_CACHE.get("SwingBioModel")
    if cls is None:
        cls = _build_class()
        _CLASS_CACHE["SwingBioModel"] = cls
    return cls(golfer, club, parameters=parameters)


def __getattr__(name: str) -> Any:
    if name == "SwingBioModel":
        cls = _CLASS_CACHE.get("SwingBioModel")
        if cls is None:
            cls = _build_class()
            _CLASS_CACHE["SwingBioModel"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
