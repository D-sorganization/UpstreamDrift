from __future__ import annotations

from dataclasses import dataclass
import sys
import types

import numpy as np
import pytest

import src.learning.rl.base_env as base_env
from src.learning.rl.manipulation_envs import ManipulationPickPlaceEnv

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _FakeBox:
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: object


class _FakeSpaces:
    Box = _FakeBox


class _FakeEngine:
    n_q = 4
    n_v = 4

    def reset(self) -> None:
        pass

    def step(self) -> None:
        pass

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(self.n_q, dtype=np.float64)

    def get_joint_velocities(self) -> np.ndarray:
        return np.zeros(self.n_q, dtype=np.float64)


@pytest.fixture(autouse=True)
def _gymnasium_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_gymnasium = types.ModuleType("gymnasium")
    fake_gymnasium.spaces = _FakeSpaces

    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    monkeypatch.setattr(base_env, "GYMNASIUM_AVAILABLE", True)
    monkeypatch.setattr(base_env, "spaces", _FakeSpaces)


def test_pick_place_accepts_explicit_ndarray_positions() -> None:
    object_initial_pos = np.array([0.25, -0.15, 0.2], dtype=np.float64)
    target_pos = np.array([0.75, 0.2, 0.35], dtype=np.float64)

    env = ManipulationPickPlaceEnv(
        engine=_FakeEngine(),
        object_initial_pos=object_initial_pos,
        target_pos=target_pos,
    )

    np.testing.assert_allclose(env._object_pos, object_initial_pos)
    np.testing.assert_allclose(env._target_pos, target_pos)
    np.testing.assert_allclose(env.task_config.target_position, target_pos)
