"""Regression tests for issue #15012 — WBC contact torques and hierarchy."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.robotics.control.whole_body.qp_solver import (
    NullspaceQPSolver,
    create_default_solver,
)
from src.robotics.control.whole_body.task import Task, TaskType
from src.robotics.control.whole_body.wbc_controller import (
    WBCConfig,
    WholeBodyController,
)
from tests.robotics.wave6_robotics.test_wbc_controller import FakeEngine, N_V


class BiasedFakeEngine(FakeEngine):
    """FakeEngine with configurable bias forces."""

    def __init__(self, n_v: int = N_V, nle: np.ndarray | None = None) -> None:
        super().__init__(n_v=n_v)
        self._nle = np.zeros(n_v) if nle is None else np.asarray(nle, dtype=float)

    def compute_bias_forces(self):
        return self._nle.copy()


def _one_dof_task(name: str, target_qdd: float, priority: int) -> Task:
    J = np.zeros((1, N_V))
    J[0, 0] = 1.0
    return Task(
        name=name,
        task_type=TaskType.SOFT,
        priority=priority,
        jacobian=J,
        target=np.array([target_qdd]),
        weight=np.array([1.0]),
    )


@pytest.mark.unit
def test_fixed_base_default_config_nonzero_contact_torques() -> None:
    """Default WBCConfig must not zero torques on fixed-base models with contact."""
    nle = np.array([0.0, 0.0, 2.0, 0.0])
    cfg = WBCConfig(use_hierarchical=False, regularization=1e-4)
    wbc = WholeBodyController(
        BiasedFakeEngine(nle=nle),
        config=cfg,
        solver=create_default_solver(),
    )
    wbc.add_task(
        Task(
            name="posture",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, N_V),
            target=np.zeros(3),
            weight=np.ones(3),
        )
    )
    J_c = np.zeros((3, N_V))
    J_c[2, 2] = 1.0
    wbc.set_contact_jacobians([J_c])

    sol = wbc.solve()
    assert sol.success
    assert sol.joint_torques is not None
    assert np.linalg.norm(sol.joint_torques) > 1e-6


@pytest.mark.unit
def test_contact_support_yields_nonzero_actuated_torques() -> None:
    """Actuated DOFs must not receive zero torque when contacts are active."""
    nle = np.array([0.0, 0.0, 2.0, 0.0])
    cfg = WBCConfig(
        use_hierarchical=False,
        floating_base_dofs=0,
        regularization=1e-4,
    )
    wbc = WholeBodyController(
        BiasedFakeEngine(nle=nle),
        config=cfg,
        solver=create_default_solver(),
    )
    wbc.add_task(
        Task(
            name="posture",
            task_type=TaskType.SOFT,
            priority=2,
            jacobian=np.eye(3, N_V),
            target=np.zeros(3),
            weight=np.ones(3),
        )
    )
    J_c = np.zeros((3, N_V))
    J_c[2, 2] = 1.0
    wbc.set_contact_jacobians([J_c])

    sol = wbc.solve()
    assert sol.success
    assert sol.joint_torques is not None
    assert np.linalg.norm(sol.joint_torques) > 1e-6, (
        f"Expected non-zero torques with contact; got {sol.joint_torques}"
    )


@pytest.mark.unit
def test_floating_base_rows_only_constrained_with_contact() -> None:
    """Unactuated rows satisfy dynamics; actuated torques stay non-zero."""
    n_v = 10
    n_fb = 6
    nle = np.linspace(0.5, 5.0, n_v)

    class Engine10(BiasedFakeEngine):
        def __init__(self) -> None:
            super().__init__(n_v=n_v, nle=nle)

    cfg = WBCConfig(use_hierarchical=False, floating_base_dofs=n_fb)
    wbc = WholeBodyController(Engine10(), config=cfg, solver=create_default_solver())
    wbc.add_task(
        Task(
            name="soft",
            task_type=TaskType.SOFT,
            priority=1,
            jacobian=np.eye(3, n_v),
            target=np.zeros(3),
            weight=np.ones(3),
        )
    )
    J_c = np.zeros((3, n_v))
    J_c[:, 6] = 1.0
    wbc.set_contact_jacobians([J_c])

    sol = wbc.solve()
    assert sol.success
    assert sol.joint_torques is not None
    actuated = sol.joint_torques[n_fb:]
    assert np.max(np.abs(actuated)) > 1e-6


@pytest.mark.unit
def test_hierarchy_respects_priority_zero_highest() -> None:
    """Higher-priority (numerically lower) task wins when targets conflict."""
    cfg = WBCConfig(use_hierarchical=True, regularization=1e-4, floating_base_dofs=0)
    wbc = WholeBodyController(FakeEngine(), config=cfg, solver=NullspaceQPSolver())
    wbc.add_task(_one_dof_task("high", target_qdd=1.0, priority=0))
    wbc.add_task(_one_dof_task("low", target_qdd=-1.0, priority=5))
    sol = wbc.solve()

    assert sol.success
    assert sol.joint_accelerations is not None
    assert sol.joint_accelerations[0] > 0.5

    wbc.clear_tasks()
    wbc.add_task(_one_dof_task("low", target_qdd=-1.0, priority=5))
    wbc.add_task(_one_dof_task("high", target_qdd=1.0, priority=0))
    sol2 = wbc.solve()
    assert sol2.success
    assert sol2.joint_accelerations is not None
    assert sol2.joint_accelerations[0] > 0.5


@pytest.mark.unit
def test_tasks_sorted_lowest_priority_number_first() -> None:
    wbc = WholeBodyController(FakeEngine())
    wbc.add_task(_one_dof_task("p4", 0.0, priority=4))
    wbc.add_task(_one_dof_task("p0", 0.0, priority=0))
    wbc.add_task(_one_dof_task("p2", 0.0, priority=2))
    assert [t.name for t in wbc.tasks] == ["p0", "p2", "p4"]


@pytest.mark.unit
def test_linearize_friction_cone_is_inscribed_inner_approximation() -> None:
    """C4: linearized pyramid must sit strictly inside ||f_t|| <= mu * f_n."""
    from src.robotics.contact.friction_cone import linearize_friction_cone

    mu = 0.5
    A, b = linearize_friction_cone(mu=mu, normal=np.array([0.0, 0.0, 1.0]), num_faces=8)
    # Vertex angle pi/8 bisects two face normals; norm > mu must violate A @ f <= b.
    angle = np.pi / 8.0
    outside_cone = np.array([0.52 * np.cos(angle), 0.52 * np.sin(angle), 1.0])
    assert np.any(A @ outside_cone > b)
    # Inscribed vertex at mu exactly satisfies A @ f <= b.
    vertex_on_cone = np.array([mu * np.cos(angle), mu * np.sin(angle), 1.0])
    assert np.all(A @ vertex_on_cone <= b + 1e-12)


@pytest.mark.unit
def test_project_to_friction_cone_returns_true_euclidean_projection() -> None:
    """C5: Euclidean projection onto the second-order Lorentz cone."""
    from src.robotics.contact.friction_cone import (
        FrictionCone,
        project_to_friction_cone,
    )

    cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]))
    f = np.array([3.0, 0.0, 1.0])
    projected = project_to_friction_cone(f, cone)
    np.testing.assert_allclose(projected, np.array([1.0, 0.0, 2.0]), atol=1e-12)
    assert cone.contains(projected)

    polar_f = np.array([1.0, 0.0, -0.6])
    np.testing.assert_allclose(
        project_to_friction_cone(polar_f, cone), np.zeros(3), atol=1e-12
    )

    surface_f = np.array([3.0, 0.0, -1.0])
    np.testing.assert_allclose(
        project_to_friction_cone(surface_f, cone), np.array([0.2, 0.0, 0.4]), atol=1e-12
    )


@pytest.mark.unit
def test_rl_floating_base_obs_and_action_dimensions_and_contact_forces() -> None:
    """C7: free-flyer uses n_q for pos, n_v for vel, actuated DOFs, plus contacts."""
    pytest.importorskip("gymnasium")
    from src.learning.rl.configs import ObservationConfig
    from src.learning.rl.humanoid_envs import HumanoidWalkEnv

    engine = SimpleNamespace(
        n_q=19,
        n_v=18,
        reset=lambda: None,
        step=lambda: None,
        set_joint_torques=lambda _a: None,
        get_joint_positions=lambda: np.zeros(19),
        get_joint_velocities=lambda: np.zeros(18),
        get_joint_torques=lambda: np.zeros(12),
        get_contact_forces=lambda: np.zeros(6),
        get_base_velocity=lambda: np.array([1.0, 0.0, 0.0]),
        get_base_position=lambda: np.array([0.0, 0.0, 0.9]),
        get_base_orientation=lambda: np.array([1.0, 0.0, 0.0, 0.0]),
    )
    env = HumanoidWalkEnv(
        engine=engine, obs_config=ObservationConfig(include_contact_forces=True)
    )
    assert env.action_space.shape == (12,)
    assert env.observation_space.shape == (19 + 18 + 6,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


@pytest.mark.unit
def test_domain_randomization_gravity_opt_in_and_inertia_coupling() -> None:
    """C8: gravity randomization defaults to False and inertia scales with mass."""
    from src.learning.sim2real.domain_randomization import (
        DomainRandomizationConfig,
        DomainRandomizer,
    )

    state = {
        "m": np.array([2.0, 4.0]),
        "I": np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]),
    }
    engine = SimpleNamespace(
        get_link_masses=lambda: state["m"].copy(),
        set_link_masses=lambda v: state.__setitem__("m", np.asarray(v, float).copy()),
        get_link_inertias=lambda: state["I"].copy(),
        set_link_inertias=lambda v: state.__setitem__("I", np.asarray(v, float).copy()),
    )
    cfg = DomainRandomizationConfig(
        randomize_friction=False, randomize_damping=False, randomize_motor=False
    )
    assert cfg.randomize_gravity is False
    nominal_inertia = engine.get_link_inertias()
    rand = DomainRandomizer(engine, config=cfg, seed=7)
    applied = rand.randomize()
    assert "gravity" not in applied
    np.testing.assert_allclose(
        engine.get_link_inertias(), nominal_inertia * applied["mass_scale"]
    )
    rand.reset_to_nominal()
    np.testing.assert_allclose(engine.get_link_inertias(), nominal_inertia)


@pytest.mark.unit
def test_offline_adversarial_imitation_honesty_and_rollout_hook() -> None:
    """C9: expose OfflineAdversarialImitation and support optional rollout_fn."""
    from src.learning.imitation import GAIL, Demonstration, DemonstrationDataset
    from src.learning.imitation import OfflineAdversarialImitation, TrainingConfig

    assert (
        OfflineAdversarialImitation is GAIL and GAIL.USES_ENVIRONMENT_ROLLOUTS is False
    )
    demo = Demonstration(
        timestamps=np.arange(8, dtype=np.float32) * 0.01,
        joint_positions=np.ones((8, 2), dtype=np.float32),
        joint_velocities=np.ones((8, 1), dtype=np.float32),
        actions=np.zeros((8, 2), dtype=np.float32),
    )
    ds = DemonstrationDataset([demo])
    learner = GAIL(
        3, 2, TrainingConfig(epochs=2, batch_size=4, hidden_sizes=[8], seed=1)
    )
    called_counts: list[int] = []

    def fake_rollout(_policy_fn, n_samples: int) -> np.ndarray:
        called_counts.append(n_samples)
        return np.full((n_samples, 3), 2.0, dtype=np.float32)

    history = learner.train(ds, validation_split=0.0, rollout_fn=fake_rollout)
    assert called_counts and len(history["policy_loss"]) == 2


@pytest.mark.unit
def test_wbc_config_exposes_documented_friction_and_force_bounds() -> None:
    """C11: WBCConfig exposes configurable friction_coeff and max_contact_force."""
    cfg = WBCConfig(friction_coeff=0.6, max_contact_force=5000.0)
    assert cfg.friction_coeff == pytest.approx(0.6)
    assert cfg.max_contact_force == pytest.approx(5000.0)
    wbc = WholeBodyController(FakeEngine(), config=cfg)
    wbc.set_contact_jacobians([np.eye(3, N_V)])
    x_lb, x_ub = wbc._build_variable_bounds(N_V, 3, np.zeros(N_V))
    assert np.allclose(x_lb[N_V:], -5000.0)
    assert np.allclose(x_ub[N_V:], 5000.0)
