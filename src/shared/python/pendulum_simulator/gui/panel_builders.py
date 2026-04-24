# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Panel builder functions extracted from MainWindow.

Each builder creates and wires up a complete simulation panel for
a specific pendulum model (double, triple, golfer).

Design by Contract
------------------
- build_double_panel(main_window) returns a fully wired SimulationPanel.
- build_triple_panel(main_window) returns a fully wired SimulationPanel.
- build_golfer_panel(main_window) returns a fully wired SimulationPanel.
- wire_toolstrip(main_window) connects toolstrip signals.

DRY
---
Common panel setup logic is factored into _connect_common_signals().
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from ..constants import GRAVITY_MSS
from ..physics import (
    JointLimits,
    JointLimitsNDOF,
    PendulumParams,
    TorqueClamp,
)
from ..physics_golfer import GolferParams
from ..physics_triple import TriplePendulumParams
from ..simulation import make_polynomial_torque, run_simulation
from ..simulation_golfer import make_polynomial_torque as make_polynomial_torque_golfer
from ..simulation_golfer import run_simulation as run_simulation_golfer
from ..simulation_triple import make_polynomial_torque as make_polynomial_torque_triple
from ..simulation_triple import run_simulation as run_simulation_triple
from .controls_widget import ControlsWidget
from .controls_widget_golfer import ControlsWidgetGolfer
from .controls_widget_triple import ControlsWidgetTriple
from .golfer_pendulum_widget import GolferPendulumWidget
from .matrix_widget import MatrixWidget
from .matrix_widget_golfer import GolferMatrixWidget
from .matrix_widget_triple import TripleMatrixWidget
from .optimization_widget import OptimizationWidget
from .pendulum_widget import PendulumWidget
from .perturbation_panel import PerturbationPanel
from .simulation_panel import SimulationPanel
from .torque_history_widget import TorqueHistoryWidget

logger = logging.getLogger(__name__)


def _apply_pendulum_view(pendulum: Any, p: dict) -> float:
    """Apply swing-plane display angles and return the tilt in radians."""
    tilt_rad = np.radians(p.get("tilt_deg", 0.0))
    pendulum.set_tilt_angle(tilt_rad)
    pendulum.set_view_azimuth(np.radians(p.get("azimuth_deg", 0.0)))
    return tilt_rad


def _effective_gravity(p: dict) -> float:
    """Return gravity projected onto the swing plane."""
    g = GRAVITY_MSS if p.get("gravity_on", True) else 0.0
    return g * float(np.cos(np.radians(p.get("tilt_deg", 0.0))))


def _parse_coefficients(raw: object) -> list[float]:
    """Parse a comma-separated preset coefficient string."""
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()] or [0.0]


def _chunk_coefficients(coeffs: np.ndarray, parts: int) -> list[list[float]]:
    """Split a coefficient vector into `parts` ordered chunks."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    chunk = len(coeffs) // parts
    return [list(coeffs[i * chunk : (i + 1) * chunk]) for i in range(parts - 1)] + [
        list(coeffs[(parts - 1) * chunk :])
    ]


def _extract_tip_motion(
    result: object,
    position_key: str,
    velocity_key: str | None = None,
    fallback_position_key: str | None = None,
) -> tuple[float, np.ndarray]:
    """Return final tip speed and position from a simulation result."""
    pos = result.positions_at(result.n_steps - 1)  # type: ignore[attr-defined]
    tip_xy = pos.get(position_key, pos.get(fallback_position_key, (0.0, 0.0)))
    if velocity_key is not None:
        vels = result.joint_velocities_at(result.n_steps - 1)  # type: ignore[attr-defined]
        tip_v = vels.get(velocity_key, (0.0, 0.0))
        speed = float(np.hypot(tip_v[0], tip_v[1]))
    elif result.n_steps >= 2:  # type: ignore[attr-defined]
        dt = float(result.t[-1] - result.t[-2])  # type: ignore[attr-defined]
        pos_prev = result.positions_at(result.n_steps - 2)  # type: ignore[attr-defined]
        tip_prev = pos_prev.get(
            position_key,
            pos_prev.get(fallback_position_key, (0.0, 0.0)),
        )
        vx = (tip_xy[0] - tip_prev[0]) / max(dt, 1e-9)
        vy = (tip_xy[1] - tip_prev[1]) / max(dt, 1e-9)
        speed = float(np.hypot(vx, vy))
    else:
        speed = 0.0
    return speed, np.array([tip_xy[0], tip_xy[1]])


def _score_tip_speed(
    *,
    model_name: str,
    run_fn: Callable[..., object],
    run_kwargs: dict[str, object],
    position_key: str,
    velocity_key: str | None = None,
    fallback_position_key: str | None = None,
) -> float:
    """Run a simulation and score it by negative final tip speed."""
    try:
        result = run_fn(**run_kwargs)
        speed, _ = _extract_tip_motion(
            result,
            position_key=position_key,
            velocity_key=velocity_key,
            fallback_position_key=fallback_position_key,
        )
        return -speed
    except (
        RuntimeError,
        ValueError,
        ArithmeticError,
    ) as exc:  # noqa: BLE001
        logger.debug("%s objective simulation failed: %s", model_name, exc)
        return 0.0


def _wire_perturbation_panel(
    perturb: PerturbationPanel,
    coeffs_source: Callable[[], list[list[float]]],
    preset_names_source: Callable[[], list[str]],
    preset_coeffs: Callable[[str], list[list[float]]],
    simulate_fn: Callable[[list], object],
    extract_fn: Callable[[object], dict],
) -> None:
    """Connect perturbation-panel sources and callbacks."""
    perturb.set_coeffs_source(coeffs_source)
    perturb.set_preset_source(preset_names_source, preset_coeffs)
    perturb.set_simulation_callbacks(simulate_fn, extract_fn)


def _double_preset_coeffs(preset: object | None) -> list[list[float]]:
    if preset is None:
        return [[0.0], [0.0]]
    return [_parse_coefficients(preset[4]), _parse_coefficients(preset[5])]  # type: ignore[index]


def _triple_preset_coeffs(preset: object | None) -> list[list[float]]:
    if preset is None:
        return [[0.0], [0.0], [0.0]]
    return [
        _parse_coefficients(preset[6]),  # type: ignore[index]
        _parse_coefficients(preset[7]),  # type: ignore[index]
        _parse_coefficients(preset[8]),  # type: ignore[index]
    ]


def _golfer_preset_coeffs(preset: Mapping[str, object] | None) -> list[list[float]]:
    if preset is None:
        return [[0.0]] * 7
    keys = [
        "tau_hub",
        "tau_rs",
        "tau_re",
        "tau_rh",
        "tau_ls",
        "tau_le",
        "tau_lh",
    ]
    return [_parse_coefficients(preset.get(k, "0")) for k in keys]


def build_double_panel(main_window: Any) -> SimulationPanel:
    """Build and return the double pendulum simulation panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance (used to access state if needed).

    Returns
    -------
    SimulationPanel
        A fully wired simulation panel for the double pendulum model.
    """
    controls = ControlsWidget()
    pendulum = PendulumWidget()
    matrix = MatrixWidget()
    torque_history = TorqueHistoryWidget()

    def build_params(p: dict) -> PendulumParams:
        _apply_pendulum_view(pendulum, p)
        return PendulumParams(
            m1=p["m1"],
            m2=p["m2"],
            L1=p["L1"],
            L2=p["L2"],
            mClub=p.get("mClub", 0.0),
            g=_effective_gravity(p),
            b1=p.get("b1", 0.0),
            b2=p.get("b2", 0.0),
            mu1=p.get("mu1", 0.0),
            mu2=p.get("mu2", 0.0),
        )

    def build_state(p: dict) -> np.ndarray:
        return np.array([p["theta1_rad"], p["phi_rad"], p["dtheta1"], p["dphi"]])

    def build_torque(p: dict) -> object:
        return make_polynomial_torque(p["shoulder_coeffs"], p["wrist_coeffs"])

    def build_limits(p: dict) -> JointLimits | None:
        if not p.get("enable_limits", False):
            return None
        return JointLimits(
            phi_min=p.get("phi_min_rad", -np.pi / 2),
            phi_max=p.get("phi_max_rad", np.pi / 2),
            theta1_min=p.get("theta1_min_rad", -np.pi),
            theta1_max=p.get("theta1_max_rad", np.pi),
            stiffness=p.get("limit_stiffness", 500.0),
        )

    def build_clamp(p: dict) -> TorqueClamp | None:
        if not p.get("enable_clamp", False):
            return None
        return TorqueClamp(
            max_torque1=p.get("max_torque1", 50.0),
            max_torque2=p.get("max_torque2", 20.0),
        )

    # Optimizer (#1108)
    optimizer = OptimizationWidget(
        model_name="Double Pendulum",
        n_torque_params=2,
    )

    def _make_double_objective(p: dict) -> Callable:
        """Build a tip-speed objective from current controls."""
        params = build_params(p)
        initial_state = build_state(p)
        t_end = p["t_end"]
        limits = build_limits(p)
        clamp = build_clamp(p)

        def objective(coeffs: np.ndarray) -> float:
            s_coeffs, w_coeffs = _chunk_coefficients(coeffs, 2)
            torque_func = make_polynomial_torque(s_coeffs, w_coeffs)
            return _score_tip_speed(
                model_name="double",
                run_fn=run_simulation,
                run_kwargs={
                    "params": params,
                    "initial_state": initial_state,
                    "t_end": t_end,
                    "torque_func": torque_func,
                    "limits": limits,
                    "clamp": clamp,
                },
                position_key="tip",
                velocity_key="tip",
            )

        return objective

    panel = SimulationPanel(
        controls=controls,
        pendulum=pendulum,  # type: ignore[arg-type]
        matrix=matrix,  # type: ignore[arg-type]
        params_builder=build_params,
        torque_builder=build_torque,
        state_builder=build_state,
        run_simulation=run_simulation,
        torque_history=torque_history,
        limits_builder=build_limits,
        clamp_builder=build_clamp,
        optimizer=optimizer,
        objective_builder=_make_double_objective,
    )
    panel._settings_key = "splitter_double"

    # Wire perturbation panel (#1284)
    perturb = PerturbationPanel()

    def _double_simulate_fn(coeffs: list) -> object:
        p = controls.get_params()
        params = build_params(p)
        initial_state = build_state(p)
        limits = build_limits(p)
        clamp = build_clamp(p)
        torque_func = make_polynomial_torque(coeffs[0], coeffs[1])
        return run_simulation(
            params=params,
            initial_state=initial_state,
            t_end=p["t_end"],
            torque_func=torque_func,  # type: ignore[arg-type]
            limits=limits,
            clamp=clamp,
        )

    def _double_extract_fn(result: object) -> dict:
        speed, tip_xy = _extract_tip_motion(result, "tip", velocity_key="tip")
        return {
            "tip_speed_final": speed,
            "tip_position_final": np.array([tip_xy[0], tip_xy[1]]),
        }

    _wire_perturbation_panel(
        perturb,
        lambda: [
            controls.get_params().get("shoulder_coeffs", [0.0]),
            controls.get_params().get("wrist_coeffs", [0.0]),
        ],
        lambda: list(controls.PRESETS.keys()),
        lambda name: _double_preset_coeffs(controls.PRESETS.get(name)),
        _double_simulate_fn,
        _double_extract_fn,
    )
    panel.set_perturbation_panel(perturb)
    return panel


def build_triple_panel(main_window: Any) -> SimulationPanel:
    """Build and return the triple pendulum simulation panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance (used to access state if needed).

    Returns
    -------
    SimulationPanel
        A fully wired simulation panel for the triple pendulum model.
    """
    controls = ControlsWidgetTriple()
    pendulum = PendulumWidget()
    matrix = TripleMatrixWidget()
    torque_history = TorqueHistoryWidget()

    def build_params(p: dict) -> TriplePendulumParams:
        _apply_pendulum_view(pendulum, p)
        return TriplePendulumParams(
            m1=p["m1"],
            m2=p["m2"],
            m3=p["m3"],
            L1=p["L1"],
            L2=p["L2"],
            L3=p["L3"],
            g=_effective_gravity(p),
            b1=p.get("b1", 0.0),
            b2=p.get("b2", 0.0),
            b3=p.get("b3", 0.0),
            mu1=p.get("mu1", 0.0),
            mu2=p.get("mu2", 0.0),
            mu3=p.get("mu3", 0.0),
            scapula_offset_rad=np.radians(p.get("scapula_deg", 0.0)),
        )

    def build_state(p: dict) -> np.ndarray:
        return np.array(
            [
                p["theta1_rad"],
                p["phi1_rad"],
                p["phi2_rad"],
                p["dtheta1"],
                p["dphi1"],
                p["dphi2"],
            ],
        )

    def build_torque(p: dict) -> object:
        return make_polynomial_torque_triple(
            p["shoulder_coeffs"],
            p["elbow_coeffs"],
            p["wrist_coeffs"],
        )

    def build_limits(p: dict) -> JointLimitsNDOF | None:
        if not p.get("enable_limits", False):
            return None
        return JointLimitsNDOF(
            angle_min=np.array(p["limit_mins_rad"]),
            angle_max=np.array(p["limit_maxs_rad"]),
            stiffness=p.get("limit_stiffness", 500.0),
        )

    def build_clamp(p: dict) -> np.ndarray | None:
        if not p.get("enable_clamp", False):
            return None
        return np.array(p["torque_limits"])

    # Optimizer (#1109)
    optimizer = OptimizationWidget(
        model_name="Triple Pendulum",
        n_torque_params=3,
    )

    def _make_triple_objective(p: dict) -> Callable:
        """Build a tip-speed objective from current controls."""
        params = build_params(p)
        initial_state = build_state(p)
        t_end = p["t_end"]
        limits = build_limits(p)
        clamp = build_clamp(p)

        def objective(coeffs: np.ndarray) -> float:
            s_c, e_c, w_c = _chunk_coefficients(coeffs, 3)
            torque_func = make_polynomial_torque_triple(s_c, e_c, w_c)
            return _score_tip_speed(
                model_name="triple",
                run_fn=run_simulation_triple,
                run_kwargs={
                    "params": params,
                    "initial_state": initial_state,
                    "t_end": t_end,
                    "torque_func": torque_func,
                    "torque_limits": clamp,
                    "limits": limits,
                },
                position_key="tip",
            )

        return objective

    panel = SimulationPanel(
        controls=controls,
        pendulum=pendulum,  # type: ignore[arg-type]
        matrix=matrix,  # type: ignore[arg-type]
        params_builder=build_params,
        torque_builder=build_torque,
        state_builder=build_state,
        run_simulation=run_simulation_triple,
        torque_history=torque_history,
        limits_builder=build_limits,
        clamp_builder=build_clamp,
        optimizer=optimizer,
        objective_builder=_make_triple_objective,
    )
    panel._settings_key = "splitter_triple"

    # Wire perturbation panel (#1284)
    perturb = PerturbationPanel()

    def _triple_simulate_fn(coeffs: list) -> object:
        p = controls.get_params()
        params = build_params(p)
        initial_state = build_state(p)
        limits = build_limits(p)
        clamp = build_clamp(p)
        torque_func = make_polynomial_torque_triple(coeffs[0], coeffs[1], coeffs[2])
        return run_simulation_triple(
            params=params,
            initial_state=initial_state,
            t_end=p["t_end"],
            torque_func=torque_func,  # type: ignore[arg-type]
            limits=limits,
            clamp=clamp,
        )

    def _triple_extract_fn(result: object) -> dict:
        speed, tip_xy = _extract_tip_motion(result, "tip")
        return {
            "tip_speed_final": speed,
            "tip_position_final": np.array([tip_xy[0], tip_xy[1]]),
        }

    _wire_perturbation_panel(
        perturb,
        lambda: [
            controls.get_params().get("shoulder_coeffs", [0.0]),
            controls.get_params().get("elbow_coeffs", [0.0]),
            controls.get_params().get("wrist_coeffs", [0.0]),
        ],
        lambda: list(controls.PRESETS.keys()),
        lambda name: _triple_preset_coeffs(controls.PRESETS.get(name)),
        _triple_simulate_fn,
        _triple_extract_fn,
    )
    panel.set_perturbation_panel(perturb)
    return panel


def build_golfer_panel(main_window: Any) -> SimulationPanel:
    """Build and return the golfer upper body simulation panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance (used to access state if needed).

    Returns
    -------
    SimulationPanel
        A fully wired simulation panel for the golfer upper body model.
    """
    controls = ControlsWidgetGolfer()
    pendulum = GolferPendulumWidget()
    matrix = GolferMatrixWidget()
    torque_history = TorqueHistoryWidget()

    def build_params(p: dict) -> GolferParams:
        _apply_pendulum_view(pendulum, p)
        return GolferParams(
            m_hub=p["m_hub"],
            m_r_upper=p["m_r_upper"],
            m_r_fore=p["m_r_fore"],
            m_l_upper=p["m_l_upper"],
            m_l_fore=p["m_l_fore"],
            m_club=p["m_club"],
            L_hub=p["L_hub"],
            L_r_upper=p["L_r_upper"],
            L_r_fore=p["L_r_fore"],
            L_l_upper=p["L_l_upper"],
            L_l_fore=p["L_l_fore"],
            L_club=p["L_club"],
            d_rs=p["d_rs"],
            d_ls=p["d_ls"],
            grip_right=p["grip_right"],
            grip_left=p["grip_left"],
            m_clubhead=p.get("m_clubhead", 0.2),
            g=_effective_gravity(p),
            b_hub=p.get("b_hub", 0.0),
            b_rs=p.get("b_rs", 0.0),
            b_re=p.get("b_re", 0.0),
            b_rh=p.get("b_rh", 0.0),
            b_ls=p.get("b_ls", 0.0),
            b_le=p.get("b_le", 0.0),
            b_lh=p.get("b_lh", 0.0),
            L_rscap=p.get("L_rscap", 0.12),
            L_lscap=p.get("L_lscap", 0.12),
            m_rscap=p.get("m_rscap", 0.5),
            m_lscap=p.get("m_lscap", 0.5),
        )

    def build_state(p: dict) -> np.ndarray:
        return np.array(
            [
                p["theta_hub_rad"],
                p["alpha_rs_rad"],
                p["alpha_re_rad"],
                p["alpha_rh_rad"],
                p["alpha_ls_rad"],
                p["alpha_le_rad"],
                p["alpha_lh_rad"],
                0.0,  # theta_club (computed by projection)
                0.0,
                0.0,
                0.0,
                0.0,  # qdot (all zero)
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    def build_torque(p: dict) -> object:
        return make_polynomial_torque_golfer(
            p["hub_coeffs"],
            p["rs_coeffs"],
            p["re_coeffs"],
            p["rh_coeffs"],
            p["ls_coeffs"],
            p["le_coeffs"],
            p["lh_coeffs"],
        )

    def build_limits(p: dict) -> JointLimitsNDOF | None:
        if not p.get("enable_limits", False):
            return None
        return JointLimitsNDOF(
            angle_min=np.array(p["limit_mins_rad"]),
            angle_max=np.array(p["limit_maxs_rad"]),
            stiffness=p.get("limit_stiffness", 500.0),
        )

    def build_clamp(p: dict) -> np.ndarray | None:
        if not p.get("enable_clamp", False):
            return None
        return np.array(p["torque_limits"])

    # Optimizer (#1110)
    optimizer = OptimizationWidget(
        model_name="Golfer Upper Body",
        n_torque_params=7,
    )

    def _make_golfer_objective(p: dict) -> Callable:
        """Build a clubhead-speed objective from current controls."""
        params = build_params(p)
        initial_state = build_state(p)
        t_end = p["t_end"]
        limits = build_limits(p)
        clamp = build_clamp(p)

        def objective(coeffs: np.ndarray) -> float:
            slices = _chunk_coefficients(coeffs, 7)
            torque_func = make_polynomial_torque_golfer(*slices)
            return _score_tip_speed(
                model_name="golfer",
                run_fn=run_simulation_golfer,
                run_kwargs={
                    "params": params,
                    "initial_state": initial_state,
                    "t_end": t_end,
                    "torque_func": torque_func,
                    "torque_limits": clamp,
                    "limits": limits,
                },
                position_key="club_tip",
                velocity_key="club_tip",
                fallback_position_key="tip",
            )

        return objective

    panel = SimulationPanel(
        controls=controls,
        pendulum=pendulum,  # type: ignore[arg-type]
        matrix=matrix,  # type: ignore[arg-type]
        params_builder=build_params,
        torque_builder=build_torque,
        state_builder=build_state,
        run_simulation=run_simulation_golfer,
        torque_history=torque_history,
        limits_builder=build_limits,
        clamp_builder=build_clamp,
        optimizer=optimizer,
        objective_builder=_make_golfer_objective,
    )
    panel._settings_key = "splitter_golfer"

    # Wire perturbation panel (#1284)
    perturb = PerturbationPanel()

    def _golfer_simulate_fn(coeffs: list) -> object:
        p = controls.get_params()
        params = build_params(p)
        initial_state = build_state(p)
        limits = build_limits(p)
        clamp = build_clamp(p)
        torque_func = make_polynomial_torque_golfer(*coeffs)  # type: ignore[arg-type]
        return run_simulation_golfer(
            params=params,
            initial_state=initial_state,
            t_end=p["t_end"],
            torque_func=torque_func,  # type: ignore[arg-type]
            limits=limits,
            clamp=clamp,
        )

    def _golfer_extract_fn(result: object) -> dict:
        speed, tip_xy = _extract_tip_motion(
            result,
            "club_tip",
            velocity_key="club_tip",
            fallback_position_key="tip",
        )
        return {
            "tip_speed_final": speed,
            "tip_position_final": np.array([tip_xy[0], tip_xy[1]]),
        }

    def _golfer_coeffs_fn() -> list:
        p = controls.get_params()
        joint_keys = [
            "hip_coeffs",
            "spine_coeffs",
            "r_shoulder_coeffs",
            "r_elbow_coeffs",
            "l_shoulder_coeffs",
            "l_elbow_coeffs",
            "wrist_coeffs",
        ]
        return [p.get(k, [0.0]) for k in joint_keys]

    _wire_perturbation_panel(
        perturb,
        _golfer_coeffs_fn,
        lambda: list(controls.PRESETS.keys()),
        lambda name: _golfer_preset_coeffs(controls.PRESETS.get(name)),
        _golfer_simulate_fn,
        _golfer_extract_fn,
    )
    panel.set_perturbation_panel(perturb)
    return panel


def wire_toolstrip(main_window: Any) -> None:
    """Connect toolstrip signals — dispatched only to the active tab's panel.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance containing the toolstrip and panels.

    Design by Contract
    ------------------
    Pre: main_window._toolstrip, main_window._tabs, and all panels are initialized.
    Post: All toolstrip signals are wired to their respective handlers.
    """
    ts = main_window._toolstrip

    # Build the ordered panel list matching tab indices
    main_window._panels = (
        main_window._double_panel,
        main_window._triple_panel,
        main_window._golfer_panel,
    )

    # ── Simulation action signals → active panel only ──────────────
    ts.run_requested.connect(
        lambda: main_window._active_panel().controls.run_requested.emit()
    )
    ts.reset_requested.connect(
        lambda: main_window._active_panel().controls.reset_requested.emit()
    )
    ts.play_toggled.connect(
        lambda checked: main_window._active_panel().controls.play_toggled.emit(checked)
    )
    ts.speed_changed.connect(
        lambda val: main_window._active_panel().controls.speed_changed.emit(val)
    )
    ts.frame_scrubbed.connect(
        lambda idx: main_window._active_panel().scrub_to_frame(idx)
    )

    # ── Export actions (#1141) → active panel's controls ──────────
    ts.export_data_requested.connect(
        lambda: main_window._active_panel().controls.export_data_requested.emit()
    )
    ts.export_video_requested.connect(
        lambda: main_window._active_panel().controls.export_video_requested.emit()
    )

    # ── Pop-out chart (#1135) → active panel ─────────────────────
    ts.popout_chart_requested.connect(main_window._on_popout_chart)

    # ── Overlay toggles → active panel's pendulum widget ──────────
    _connect_common_signals(main_window)

    # ── Torque/MoF/Sum display toggles (#1208) → active panel's pendulum ──
    ts.torque_vectors_toggled.connect(
        lambda v: _fwd_overlay("set_show_torque_vectors", v)
    )
    ts.moment_of_force_toggled.connect(
        lambda v: _fwd_overlay("set_show_moment_of_force", v)
    )
    ts.sum_moments_toggled.connect(lambda v: _fwd_overlay("set_show_sum_moments", v))

    # ── Scale sliders → active panel's pendulum widget ────────────
    def _fwd_overlay(attr: str, value: object) -> None:
        pw = main_window._active_panel().pendulum
        if hasattr(pw, attr):
            getattr(pw, attr)(value)

    ts.force_scale_changed.connect(lambda v: _fwd_overlay("set_force_scale", v))
    ts.mob_scale_changed.connect(lambda v: _fwd_overlay("set_mob_ellipsoid_scale", v))
    ts.force_ell_scale_changed.connect(
        lambda v: _fwd_overlay("set_force_ellipsoid_scale", v)
    )

    # ── Rotation controls (#1146) → active panel's pendulum widget ──
    ts.azimuth_changed.connect(lambda v: _fwd_overlay("set_view_azimuth", v))
    ts.tilt_changed.connect(lambda v: _fwd_overlay("set_tilt_angle", v))

    # ── Reset view → active panel's pendulum widget ───────────────
    ts.reset_view_requested.connect(
        lambda: (
            main_window._active_panel().pendulum.reset_view()
            if hasattr(main_window._active_panel().pendulum, "reset_view")
            else None
        )
    )

    # ── Per-segment overlay visibility ────────────────────────────
    ts.segment_visibility_changed.connect(
        lambda vis: (
            main_window._active_panel().pendulum.set_visible_segments(vis)
            if hasattr(main_window._active_panel().pendulum, "set_visible_segments")
            else None
        )
    )

    # ── Model selection dropdown (#1149) ──────────────────────────
    def _on_model_dropdown_changed(idx: int) -> None:
        main_window._tabs.blockSignals(True)
        main_window._tabs.setCurrentIndex(idx)
        main_window._tabs.blockSignals(False)

    def _on_tab_changed(idx: int) -> None:
        ts.cmb_model.blockSignals(True)
        ts.cmb_model.setCurrentIndex(idx)
        ts.cmb_model.blockSignals(False)

    ts.model_changed.connect(_on_model_dropdown_changed)
    main_window._tabs.currentChanged.connect(_on_tab_changed)

    # ── Busy state and frame sync — only forward from the active panel ─
    # Guard each callback so non-active panels are silently ignored.
    for panel in main_window._panels:
        panel.sim_started.connect(
            lambda _p=panel: (
                ts.set_running(True) if _p is main_window._active_panel() else None
            )
        )
        panel.sim_finished.connect(
            lambda _p=panel: (
                [
                    ts.set_running(False),
                    ts.set_frame_range(_p.current_n_steps()),
                ]
                if _p is main_window._active_panel()
                else None
            )
        )
        panel.frame_changed.connect(
            lambda idx, _p=panel: (
                ts.set_frame(idx) if _p is main_window._active_panel() else None
            )
        )
        # Reset toolstrip play button when playback ends
        panel.playback_ended.connect(
            lambda _p=panel: (
                ts.btn_play.setChecked(False)
                if _p is main_window._active_panel()
                else None
            )
        )

    # Loop toggle — forward to all panels
    if hasattr(ts, "loop_toggled"):

        def _set_loop(v: bool) -> None:
            for p in main_window._panels:
                p._loop_playback = v

        ts.loop_toggled.connect(_set_loop)

    # Update segment checkboxes when tab changes
    main_window._tabs.currentChanged.connect(main_window._on_tab_changed)
    # Initialize with the default tab's segments
    main_window._on_tab_changed(main_window._tabs.currentIndex())


def _connect_common_signals(main_window: Any) -> None:
    """Connect common overlay toggle signals to the active panel's pendulum widget.

    This factored-out helper avoids duplication of the same signal wiring
    pattern across the three panel builders.

    Parameters
    ----------
    main_window : MainWindow
        The main window instance containing the toolstrip and panels.
    """
    ts = main_window._toolstrip

    def _fwd_overlay(attr: str, value: object) -> None:
        pw = main_window._active_panel().pendulum
        if hasattr(pw, attr):
            getattr(pw, attr)(value)

    ts.forces_toggled.connect(lambda v: _fwd_overlay("set_show_forces", v))
    ts.zero_torque_toggled.connect(
        lambda v: _fwd_overlay("set_show_zero_torque_forces", v)
    )
    ts.mob_ellipsoid_toggled.connect(
        lambda v: _fwd_overlay("set_show_mob_ellipsoids", v)
    )
    ts.force_ellipsoid_toggled.connect(
        lambda v: _fwd_overlay("set_show_force_ellipsoids", v)
    )
    ts.com_toggled.connect(lambda v: _fwd_overlay("set_show_com", v))

    # ── 3D segment rendering (#1155) ──────────────────────────────
    ts.mode_3d_toggled.connect(lambda v: _fwd_overlay("set_3d_mode", v))
