# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""
Docked Analysis tab for the Pendulum Simulator.

Provides an integrated plotting environment with:
- 2D line plots (any variable vs any variable)
- 3D surface plots (Z = f(X, Y) via parameter sweep)
- Series selection sidebar with model-aware variable lists
- Regression overlay support
- Dark-themed matplotlib figures

Design by Contract
------------------
- set_result() must be called before any 2D plotting.
- Surface plots require a sweep function and range specifications.
- All public methods validate inputs with assertions.

DRY
---
- Reuses data_extractor registry for series enumeration.
- Delegates regression to popout_chart.fit_regression().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _create_fallback_widget() -> Any:
    """Return a placeholder widget when matplotlib is unavailable."""
    from PyQt6.QtWidgets import QLabel

    lbl = QLabel("Install matplotlib for the Analysis tab:\n  pip install matplotlib")
    lbl.setStyleSheet("color: #c0c0d8; padding: 40px; font-size: 14px;")
    return lbl


class AnalysisTab:
    """Docked analysis panel with 2D and 3D plotting capabilities.

    Usage::

        tab = AnalysisTab(parent)
        main_layout.addWidget(tab.widget())
        tab.set_result(sim_result, model_type="double")
        tab.plot_2d("time", "tip_speed")
        tab.plot_surface(sweep_fn, x_range, y_range, "θ₁", "φ", "Tip Speed")
    """

    def __init__(self, parent: Any = None) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QComboBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSplitter,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )

        from .no_scroll_widgets import NoScrollSpinBox

        self._parent = parent
        self._result: Any = None
        self._model_type: str = "double"

        # --- Main widget ---
        self._widget = QWidget(parent)
        main_layout = QHBoxLayout(self._widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left sidebar: series selection ---
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(4, 4, 4, 4)

        # 2D controls
        group_2d = QGroupBox("2D Line Plot")
        form_2d = QVBoxLayout()

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._reg_spin = NoScrollSpinBox()
        self._reg_spin.setRange(0, 10)
        self._reg_spin.setValue(0)
        self._reg_spin.setToolTip("Polynomial regression degree (0 = none)")

        form_2d.addWidget(QLabel("X axis:"))
        form_2d.addWidget(self._x_combo)
        form_2d.addWidget(QLabel("Y axis:"))
        form_2d.addWidget(self._y_combo)
        form_2d.addWidget(QLabel("Regression degree:"))
        form_2d.addWidget(self._reg_spin)

        btn_plot_2d = QPushButton("Plot 2D")
        btn_plot_2d.clicked.connect(self._on_plot_2d)
        form_2d.addWidget(btn_plot_2d)
        group_2d.setLayout(form_2d)
        sidebar_layout.addWidget(group_2d)

        # 3D surface controls
        group_3d = QGroupBox("3D Surface Plot")
        form_3d = QVBoxLayout()

        self._x3_combo = QComboBox()
        self._y3_combo = QComboBox()
        self._z3_combo = QComboBox()
        self._sweep_points = NoScrollSpinBox()
        self._sweep_points.setRange(10, 100)
        self._sweep_points.setValue(30)
        self._sweep_points.setToolTip("Grid resolution per axis")

        form_3d.addWidget(QLabel("X axis (sweep):"))
        form_3d.addWidget(self._x3_combo)
        form_3d.addWidget(QLabel("Y axis (sweep):"))
        form_3d.addWidget(self._y3_combo)
        form_3d.addWidget(QLabel("Z axis (result):"))
        form_3d.addWidget(self._z3_combo)
        form_3d.addWidget(QLabel("Grid points:"))
        form_3d.addWidget(self._sweep_points)

        btn_plot_3d = QPushButton("Plot Surface")
        btn_plot_3d.clicked.connect(self._on_plot_surface)
        form_3d.addWidget(btn_plot_3d)
        group_3d.setLayout(form_3d)
        sidebar_layout.addWidget(group_3d)

        sidebar_layout.addStretch()
        splitter.addWidget(sidebar)

        # --- Right side: plot tabs ---
        self._plot_tabs = QTabWidget()

        if _HAS_MPL:
            # 2D canvas
            self._fig_2d = Figure(figsize=(7, 5), dpi=100)
            self._fig_2d.patch.set_facecolor("#1a1a28")
            self._ax_2d = self._fig_2d.add_subplot(111)
            self._canvas_2d = FigureCanvasQTAgg(self._fig_2d)
            self._plot_tabs.addTab(self._canvas_2d, "2D Plot")

            # 3D canvas
            self._fig_3d = Figure(figsize=(7, 5), dpi=100)
            self._fig_3d.patch.set_facecolor("#1a1a28")
            self._ax_3d = self._fig_3d.add_subplot(111, projection="3d")
            self._canvas_3d = FigureCanvasQTAgg(self._fig_3d)
            self._plot_tabs.addTab(self._canvas_3d, "3D Surface")
        else:
            self._plot_tabs.addTab(_create_fallback_widget(), "Plotting")

        splitter.addWidget(self._plot_tabs)
        splitter.setStretchFactor(0, 1)  # sidebar
        splitter.setStretchFactor(1, 3)  # plots

        main_layout.addWidget(splitter)

        # Surface outputs shared by all models
        self._surface_outputs = [
            ("mass_matrix_det", "det(M)"),
            ("mass_matrix_cond", "cond(M)"),
            ("potential_energy", "Potential energy"),
            ("manipulability", "Manipulability index"),
        ]
        self._populate_surface_combos()

    def widget(self) -> Any:
        """Return the top-level QWidget for embedding."""
        return self._widget

    def set_result(self, result: Any, model_type: str = "double") -> None:
        """Set the simulation result for 2D plotting.

        Pre: result has .t, .states, and data_extractor-compatible API.
        """
        if model_type is None:
            raise ValueError("model_type must be provided")
        self._result = result
        old_model = self._model_type
        self._model_type = model_type
        self._populate_series_combos()
        if model_type != old_model:
            self._populate_surface_combos()

    # Model-aware sweep variable definitions
    _SWEEP_VARS: dict[str, list[tuple[str, str, str, tuple[float, float]]]] = {
        "double": [
            ("theta1", "Shoulder angle θ₁", "rad", (-np.pi, np.pi)),
            ("phi", "Wrist angle φ", "rad", (-np.pi, np.pi)),
        ],
        "triple": [
            ("theta1", "Shoulder angle θ₁", "rad", (-np.pi, np.pi)),
            ("phi1", "Elbow angle φ₁", "rad", (-np.pi, np.pi)),
            ("phi2", "Wrist angle φ₂", "rad", (-np.pi, np.pi)),
        ],
        "golfer": [
            ("theta1", "Hub rotation θ_hub", "rad", (-np.pi, np.pi)),
            ("phi", "Right shoulder α_rs", "rad", (-np.pi, np.pi)),
            ("phi1", "Right elbow α_re", "rad", (-np.pi, np.pi)),
            ("phi2", "Right hand α_rh", "rad", (-np.pi, np.pi)),
        ],
    }

    def _populate_surface_combos(self) -> None:
        """Fill the surface sweep X/Y/Z dropdowns for the active model."""
        variables = self._SWEEP_VARS.get(self._model_type, self._SWEEP_VARS["double"])
        self._surface_variables = variables

        self._x3_combo.clear()
        self._y3_combo.clear()
        for key, desc, _, _ in variables:
            self._x3_combo.addItem(desc, key)
            self._y3_combo.addItem(desc, key)
        if len(variables) > 1:
            self._y3_combo.setCurrentIndex(1)

        self._z3_combo.clear()
        for key, desc in self._surface_outputs:
            self._z3_combo.addItem(desc, key)

    def _populate_series_combos(self) -> None:
        """Fill the X/Y dropdowns from the data_extractor registry."""
        from ..data_extractor import list_available_series

        series = list_available_series(self._model_type)
        self._x_combo.clear()
        self._y_combo.clear()
        for key, desc, unit in series:
            label = f"{desc} ({unit})"
            self._x_combo.addItem(label, key)
            self._y_combo.addItem(label, key)

        # Defaults
        x_idx = self._x_combo.findData("time")
        if x_idx >= 0:
            self._x_combo.setCurrentIndex(x_idx)
        y_idx = self._y_combo.findData("tip_speed")
        if y_idx >= 0:
            self._y_combo.setCurrentIndex(y_idx)

    def _on_plot_2d(self) -> None:
        """Handle Plot 2D button click."""
        if not _HAS_MPL or self._result is None:
            logger.warning("No result loaded or matplotlib unavailable")
            return

        x_key = self._x_combo.currentData()
        y_key = self._y_combo.currentData()
        if x_key is None or y_key is None:
            return

        from ..data_extractor import extract_series

        try:
            x_vals, x_desc, x_unit = extract_series(
                self._result, x_key, self._model_type
            )
            y_vals, y_desc, y_unit = extract_series(
                self._result, y_key, self._model_type
            )
        except (KeyError, AttributeError) as exc:
            logger.error("Failed to extract series: %s", exc)
            return

        self._plot_2d(
            x_vals,
            y_vals,
            f"{x_desc} ({x_unit})",
            f"{y_desc} ({y_unit})",
            f"{y_desc} vs {x_desc}",
        )

    def _plot_2d(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str,
        ylabel: str,
        title: str,
    ) -> None:
        """Render a 2D line plot on the embedded canvas."""
        if not (len(x) == len(y)):
            raise ValueError("x and y must have same length")

        ax = self._ax_2d
        ax.clear()
        _style_axes(ax)
        ax.plot(x, y, color="#6fa8dc", linewidth=1.5, label="Data")

        # Regression overlay
        deg = self._reg_spin.value()
        if deg > 0 and len(x) > deg:
            from .popout_chart import fit_regression

            x_fit, y_fit, _ = fit_regression(x, y, deg)
            ax.plot(
                x_fit,
                y_fit,
                color="#ff7043",
                linewidth=2,
                linestyle="--",
                label=f"Fit (deg {deg})",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(facecolor="#252540", edgecolor="#505070", labelcolor="#c0c0d8")
        ax.grid(True, color="#303050", alpha=0.5)
        self._fig_2d.tight_layout()
        self._canvas_2d.draw()
        self._plot_tabs.setCurrentIndex(0)
        logger.info("Plotted 2D: %s", title)

    def _on_plot_surface(self) -> None:
        """Handle Plot Surface button click."""
        if not _HAS_MPL:
            return

        x_key = self._x3_combo.currentData()
        y_key = self._y3_combo.currentData()
        z_key = self._z3_combo.currentData()
        n_pts = self._sweep_points.value()

        if x_key == y_key:
            logger.warning("X and Y axes must be different for surface plot")
            return

        # Find ranges
        x_range = (-np.pi, np.pi)
        y_range = (-np.pi, np.pi)
        for key, _, _, rng in self._surface_variables:
            if key == x_key:
                x_range = rng
            if key == y_key:
                y_range = rng

        x_desc = self._x3_combo.currentText()
        y_desc = self._y3_combo.currentText()
        z_desc = self._z3_combo.currentText()

        self._compute_and_plot_surface(
            x_key,
            y_key,
            z_key,
            x_range,
            y_range,
            n_pts,
            x_desc,
            y_desc,
            z_desc,
        )

    def _compute_and_plot_surface(
        self,
        x_key: str,
        y_key: str,
        z_key: str,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        n_pts: int,
        xlabel: str,
        ylabel: str,
        zlabel: str,
    ) -> None:
        """Compute a parameter sweep and render the 3D surface."""
        if x_key is None:
            raise ValueError("x_key must be provided")
        x_vals = np.linspace(x_range[0], x_range[1], n_pts)
        y_vals = np.linspace(y_range[0], y_range[1], n_pts)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)

        # Compute Z values via the appropriate physics function
        evaluator = self._get_surface_evaluator(z_key)
        if evaluator is None:
            logger.warning("No evaluator for z_key=%s", z_key)
            return

        for i in range(n_pts):
            for j in range(n_pts):
                q_angles = {x_key: X[i, j], y_key: Y[i, j]}
                try:
                    Z[i, j] = evaluator(q_angles)
                except (ValueError, np.linalg.LinAlgError):
                    Z[i, j] = np.nan

        self._render_surface(X, Y, Z, xlabel, ylabel, zlabel)

    def _get_surface_evaluator(self, z_key: str) -> Any:
        """Return a callable that evaluates z_key given angle dict.

        Supports double, triple, and golfer models. Each evaluator
        accepts a dict of angle names → values and returns a scalar.

        Design by Contract
        ------------------
        Pre:  z_key is a valid surface output key.
        Post: Returns a callable or None.
        """
        if z_key is None:
            raise ValueError("z_key must be provided")
        if self._model_type == "double":
            return self._evaluator_double(z_key)
        if self._model_type == "triple":
            return self._evaluator_triple(z_key)
        if self._model_type == "golfer":
            return self._evaluator_golfer(z_key)

        logger.warning(
            "Surface evaluator not available for model=%s z=%s", self._model_type, z_key
        )
        return None

    # ── Double pendulum evaluators ──────────────────────────────────

    def _evaluator_double(self, z_key: str) -> Any:
        """Return surface evaluator for the double pendulum model."""
        if z_key is None:
            raise ValueError("z_key must be provided")
        from ..physics import (
            PendulumParams,
            forward_kinematics,
            mass_matrix,
            potential_energy,
        )

        params = self._get_params_or_default(
            lambda: PendulumParams(
                m1=5.0,
                m2=1.0,
                L1=0.6,
                L2=1.1,
            )
        )

        if z_key == "mass_matrix_det":

            def _eval(angles: dict) -> float:
                M = mass_matrix(angles.get("phi", 0.0), params)
                return float(np.linalg.det(M))

            return _eval
        if z_key == "mass_matrix_cond":

            def _eval(angles: dict) -> float:
                M = mass_matrix(angles.get("phi", 0.0), params)
                return float(np.linalg.cond(M))

            return _eval
        if z_key == "potential_energy":

            def _eval(angles: dict) -> float:
                state = np.array(
                    [angles.get("theta1", 0.0), angles.get("phi", 0.0), 0.0, 0.0]
                )
                return potential_energy(state, params)

            return _eval
        if z_key == "manipulability":
            return self._numerical_manipulability(
                lambda a: forward_kinematics(a["theta1"], a["phi"], params),
                "tip",
                ["theta1", "phi"],
            )
        return None

    # ── Triple pendulum evaluators ──────────────────────────────────

    def _evaluator_triple(self, z_key: str) -> Any:
        """Return surface evaluator for the triple pendulum model."""
        if z_key is None:
            raise ValueError("z_key must be provided")
        from ..physics_triple import (
            TriplePendulumParams,
        )
        from ..physics_triple import forward_kinematics as triple_fk
        from ..physics_triple import mass_matrix as triple_mm
        from ..physics_triple import potential_energy as triple_pe

        params = self._get_params_or_default(
            lambda: TriplePendulumParams(
                m1=3.0,
                m2=2.0,
                m3=1.0,
                L1=0.4,
                L2=0.35,
                L3=0.3,
            )
        )

        if z_key == "mass_matrix_det":

            def _eval(angles: dict) -> float:
                M = triple_mm(angles.get("phi1", 0.0), angles.get("phi2", 0.0), params)
                return float(np.linalg.det(M))

            return _eval
        if z_key == "mass_matrix_cond":

            def _eval(angles: dict) -> float:
                M = triple_mm(angles.get("phi1", 0.0), angles.get("phi2", 0.0), params)
                return float(np.linalg.cond(M))

            return _eval
        if z_key == "potential_energy":

            def _eval(angles: dict) -> float:
                state = np.array(
                    [
                        angles.get("theta1", 0.0),
                        angles.get("phi1", 0.0),
                        angles.get("phi2", 0.0),
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
                return triple_pe(state, params)

            return _eval
        if z_key == "manipulability":
            return self._numerical_manipulability(
                lambda a: triple_fk(
                    a.get("theta1", 0.0), a.get("phi1", 0.0), a.get("phi2", 0.0), params
                ),
                "tip",
                ["theta1", "phi1", "phi2"],
            )
        return None

    # ── Golfer model evaluators ─────────────────────────────────────

    def _evaluator_golfer(self, z_key: str) -> Any:
        """Return surface evaluator for the golfer upper-body model."""
        if z_key is None:
            raise ValueError("z_key must be provided")
        from ..golfer_dynamics import potential_energy_from_q
        from ..physics_golfer import GolferParams
        from ..physics_golfer import mass_matrix as golfer_mm

        params = self._get_params_or_default(
            lambda: GolferParams(
                m_hub=40.0,
                m_r_upper=3.0,
                m_r_fore=1.5,
                m_l_upper=3.0,
                m_l_fore=1.5,
                m_club=0.5,
                L_hub=0.5,
                L_r_upper=0.3,
                L_r_fore=0.25,
                L_l_upper=0.3,
                L_l_fore=0.25,
                L_club=1.0,
                d_rs=0.2,
                d_ls=0.2,
                grip_right=0.3,
                grip_left=0.3,
            )
        )

        if z_key == "mass_matrix_det":

            def _eval(angles: dict) -> float:
                q = self._golfer_q_from_angles(angles)
                M = golfer_mm(q, params)
                return float(np.linalg.det(M))

            return _eval
        if z_key == "mass_matrix_cond":

            def _eval(angles: dict) -> float:
                q = self._golfer_q_from_angles(angles)
                M = golfer_mm(q, params)
                return float(np.linalg.cond(M))

            return _eval
        if z_key == "potential_energy":

            def _eval(angles: dict) -> float:
                q = self._golfer_q_from_angles(angles)
                return potential_energy_from_q(q, params)

            return _eval
        if z_key == "manipulability":
            try:
                from ..golfer_dynamics import analytical_fk_jacobians

                def _eval(angles: dict) -> float:
                    q = self._golfer_q_from_angles(angles)
                    jacs = analytical_fk_jacobians(q, params)
                    J = jacs.get("club_tip", np.zeros((2, 8)))
                    return float(np.sqrt(max(0, np.linalg.det(J @ J.T))))

                return _eval
            except ImportError:
                logger.warning("analytical_fk_jacobians unavailable")
                return None
        return None

    # ── Shared helpers ──────────────────────────────────────────────

    def _get_params_or_default(self, default_factory: Any) -> Any:
        """Extract params from the loaded result, or build defaults."""
        try:
            if self._result is not None and hasattr(self._result, "params"):
                return self._result.params
        except AttributeError:
            pass
        return default_factory()

    @staticmethod
    def _golfer_q_from_angles(angles: dict) -> np.ndarray:
        """Build an 8-DOF generalized coordinate vector from sweep angles.

        Unmapped DOFs default to 0.0.
        """
        # Map sweep variable names to q indices
        _MAP = {
            "theta1": 0,  # hub rotation
            "phi": 1,  # right shoulder
            "phi1": 2,  # right elbow
            "phi2": 3,  # right hand / club coupling
        }
        q = np.zeros(8)
        for name, val in angles.items():
            idx = _MAP.get(name)
            if idx is not None:
                q[idx] = val
        return q

    @staticmethod  # noqa: B027
    def _numerical_manipulability(
        fk_fn: Any,
        tip_key: str,
        angle_keys: list[str],
    ) -> Any:
        """Build a numerical manipulability evaluator via finite differences.

        Returns w = sqrt(det(J J^T)) where J is the 2 × n_dof Jacobian
        approximated by central differences.
        """
        if tip_key is None:
            raise ValueError("tip_key must be provided")
        eps = 1e-7
        n_dof = len(angle_keys)

        def _eval(angles: dict) -> float:
            fk0 = fk_fn(angles)
            tip0 = np.asarray(fk0[tip_key], dtype=float)
            J = np.zeros((2, n_dof))
            for k, key in enumerate(angle_keys):
                a_plus = dict(angles)
                a_plus[key] = angles.get(key, 0.0) + eps
                fkp = fk_fn(a_plus)
                tip_p = np.asarray(fkp[tip_key], dtype=float)
                J[:, k] = (tip_p[:2] - tip0[:2]) / eps
            return float(np.sqrt(max(0, np.linalg.det(J @ J.T))))

        return _eval

    def _render_surface(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        xlabel: str,
        ylabel: str,
        zlabel: str,
    ) -> None:
        """Render a 3D surface on the embedded canvas."""
        if X is None:
            raise ValueError("X must be provided")
        ax = self._ax_3d
        ax.clear()

        # Style 3D axes
        ax.set_facecolor("#1a1a28")
        ax.tick_params(colors="#c0c0d8")
        ax.xaxis.label.set_color("#c0c0d8")
        ax.yaxis.label.set_color("#c0c0d8")
        ax.zaxis.label.set_color("#c0c0d8")  # type: ignore[attr-defined]
        ax.xaxis.pane.fill = False  # type: ignore[attr-defined]
        ax.yaxis.pane.fill = False  # type: ignore[attr-defined]
        ax.zaxis.pane.fill = False  # type: ignore[attr-defined]

        # Mask NaN for cleaner rendering
        Z_masked = np.ma.array(Z, mask=~np.isfinite(Z))

        ax.plot_surface(  # type: ignore[attr-defined]
            X,
            Y,
            Z_masked,
            cmap="viridis",
            alpha=0.85,
            edgecolor="none",
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_zlabel(zlabel, fontsize=10)  # type: ignore[attr-defined]
        ax.set_title(f"{zlabel} surface", color="#c0c0d8", fontsize=12)

        self._fig_3d.tight_layout()
        self._canvas_3d.draw()
        self._plot_tabs.setCurrentIndex(1)
        logger.info("Rendered surface: %s", zlabel)

    def plot_2d(self, x_key: str, y_key: str) -> None:
        """Programmatic 2D plot (for external callers)."""
        if x_key is None:
            raise ValueError("x_key must be provided")
        if self._result is None:
            logger.warning("No result loaded")
            return
        from ..data_extractor import extract_series

        x_vals, x_desc, x_unit = extract_series(self._result, x_key, self._model_type)
        y_vals, y_desc, y_unit = extract_series(self._result, y_key, self._model_type)
        self._plot_2d(
            x_vals,
            y_vals,
            f"{x_desc} ({x_unit})",
            f"{y_desc} ({y_unit})",
            f"{y_desc} vs {x_desc}",
        )


def _style_axes(ax: Any) -> None:
    """Apply dark theme to a matplotlib axes."""
    ax.set_facecolor("#1a1a28")
    ax.tick_params(colors="#c0c0d8")
    ax.xaxis.label.set_color("#c0c0d8")
    ax.yaxis.label.set_color("#c0c0d8")
    ax.title.set_color("#c0c0d8")
    for spine in ax.spines.values():
        spine.set_color("#505070")
