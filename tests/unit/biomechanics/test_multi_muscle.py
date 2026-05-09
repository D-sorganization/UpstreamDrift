"""Tests for src.shared.python.biomechanics.multi_muscle (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.multi_muscle import (
    AntagonistPair,
    MuscleAttachment,
    MuscleGroup,
    create_elbow_muscle_system,
)
from src.shared.python.core.contracts import PreconditionError


def _make_muscle(f_max: float = 500.0) -> HillMuscleModel:
    return HillMuscleModel(MuscleParameters(F_max=f_max, l_opt=0.10, l_slack=0.20))


def _make_group(name: str = "Flexors") -> MuscleGroup:
    group = MuscleGroup(name)
    group.add_muscle("m1", _make_muscle(), moment_arm=0.05)
    return group


class TestMuscleAttachment:
    def test_multi_muscle_construction(self) -> None:
        att = MuscleAttachment("biceps", 0.05)
        assert att.muscle_name == "biceps"
        assert att.moment_arm == pytest.approx(0.05)


class TestMuscleGroup:
    def test_multi_muscle_construction(self) -> None:
        group = MuscleGroup("Flexors")
        assert group.name == "Flexors"

    def test_add_muscle(self) -> None:
        group = MuscleGroup("Flexors")
        muscle = _make_muscle()
        group.add_muscle("biceps", muscle, moment_arm=0.05)
        assert "biceps" in group.muscles
        assert "biceps" in group.attachments

    def test_add_zero_moment_arm_raises(self) -> None:
        group = MuscleGroup("Flexors")
        muscle = _make_muscle()
        with pytest.raises(PreconditionError):
            group.add_muscle("biceps", muscle, moment_arm=0.0)

    def test_add_empty_name_raises(self) -> None:
        group = MuscleGroup("Flexors")
        muscle = _make_muscle()
        with pytest.raises(PreconditionError):
            group.add_muscle("", muscle, moment_arm=0.05)

    def test_compute_net_torque_zero_activation(self) -> None:
        group = _make_group()
        torque = group.compute_net_torque(
            activations={"m1": 0.0},
            muscle_states={"m1": (0.10, 0.0)},
        )
        assert isinstance(torque, float)

    def test_compute_net_torque_finite(self) -> None:
        group = _make_group()
        torque = group.compute_net_torque(
            activations={"m1": 0.5},
            muscle_states={"m1": (0.10, 0.0)},
        )
        import numpy as np

        assert np.isfinite(torque)

    def test_activation_out_of_range_raises(self) -> None:
        group = _make_group()
        with pytest.raises(PreconditionError):
            group.compute_net_torque(
                activations={"m1": 1.5},  # > 1.0
                muscle_states={"m1": (0.10, 0.0)},
            )

    def test_positive_torque_for_positive_activation(self) -> None:
        group = _make_group()
        torque = group.compute_net_torque(
            activations={"m1": 0.8},
            muscle_states={"m1": (0.10, 0.0)},
        )
        # Positive moment arm × positive force → positive torque
        assert torque > 0.0


class TestAntagonistPair:
    def test_multi_muscle_construction(self) -> None:
        agonist = _make_group("Flexors")
        antagonist = MuscleGroup("Extensors")
        antagonist.add_muscle("triceps", _make_muscle(), moment_arm=-0.04)
        pair = AntagonistPair(agonist, antagonist)
        assert pair.agonist is agonist
        assert pair.antagonist is antagonist

    def test_compute_net_torque_finite(self) -> None:
        agonist = _make_group("Flexors")
        antagonist = MuscleGroup("Extensors")
        antagonist.add_muscle("triceps", _make_muscle(300.0), moment_arm=-0.04)
        pair = AntagonistPair(agonist, antagonist)

        torque = pair.compute_net_torque(
            agonist_activations={"m1": 0.8},
            antagonist_activations={"triceps": 0.2},
            muscle_states={"m1": (0.10, 0.0), "triceps": (0.10, 0.0)},
        )
        import numpy as np

        assert np.isfinite(torque)


class TestCreateElbowMuscleSystem:
    def test_returns_antagonist_pair(self) -> None:
        system = create_elbow_muscle_system()
        assert isinstance(system, AntagonistPair)

    def test_has_agonist_and_antagonist(self) -> None:
        system = create_elbow_muscle_system()
        assert isinstance(system.agonist, MuscleGroup)
        assert isinstance(system.antagonist, MuscleGroup)

    def test_system_computes_torque(self) -> None:
        system = create_elbow_muscle_system()
        # Use default muscle names from factory
        agonist_names = list(system.agonist.muscles.keys())
        antagonist_names = list(system.antagonist.muscles.keys())
        agonist_activations = dict.fromkeys(agonist_names, 0.5)
        antagonist_activations = dict.fromkeys(antagonist_names, 0.1)
        muscle_states = {
            n: (m.params.l_opt, 0.0) for n, m in system.agonist.muscles.items()
        }
        muscle_states.update(
            {n: (m.params.l_opt, 0.0) for n, m in system.antagonist.muscles.items()}
        )
        torque = system.compute_net_torque(
            agonist_activations, antagonist_activations, muscle_states
        )
        import numpy as np

        assert np.isfinite(torque)
