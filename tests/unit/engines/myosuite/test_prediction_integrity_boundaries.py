"""Native prediction bounds verification work while rejecting preparation mutation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.myosuite_project_forecast import ProjectTaskForecaster

from .test_project_task_forecast import _observation, _task
from .test_project_task_native_search import _module, _search
from .test_project_task_producer import _native_runtime

pytestmark = pytest.mark.unit


def test_repeated_predictions_verify_provider_at_three_complete_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound expensive byte reads per prediction without sharing across calls."""
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    calls = []
    original = module.direct.direct_provider_sha256

    def verified(*args: object) -> str:
        result = original(*args)
        calls.append(result)
        return result

    try:
        observation = _observation(task)
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                monkeypatch.setattr(module.direct, "direct_provider_sha256", verified)
                commands = np.full((2, 2), 0.2)
                first = search.predict(observation, commands)
                assert len(calls) == 3
                second = search.predict(observation, commands)
                assert len(calls) == 6
                assert set(calls) == {search._seed.registration.provider_sha256}
                np.testing.assert_array_equal(
                    first.integration_states, second.integration_states
                )
    finally:
        task.close()


@pytest.mark.parametrize("mutation", ["model", "source", "handles", "callback", "live"])
def test_preparation_mutation_is_rejected_before_any_native_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    mj = _native_runtime()
    module = _module()
    task = _task(tmp_path, mj)
    steps = []
    original_step = mj.mj_step

    def step(*args: object, **kwargs: object) -> None:
        steps.append(True)
        original_step(*args, **kwargs)

    try:
        observation = _observation(task)
        with ProjectTaskForecaster(task) as forecast:
            with _search(module, forecast, observation) as search:
                original_bundle = search._state_bundle
                original_data = search._environment.data
                original_mass = search._model.body_mass.copy()
                source = search._seed.registration.model_path
                original_bytes = source.read_bytes()

                def prepare(*args: object) -> object:
                    bundle = original_bundle(*args)
                    if mutation == "model":
                        search._model.body_mass[1] *= 2
                    elif mutation == "source":
                        source.write_bytes(original_bytes + b"\n<!-- changed -->")
                    elif mutation == "handles":
                        search._environment.data = mj.MjData(search._model)
                    elif mutation == "callback":
                        mj.set_mjcb_control(lambda model, data: None)
                    else:
                        task.data.act[:] += 0.1
                    return bundle

                monkeypatch.setattr(search, "_state_bundle", prepare)
                monkeypatch.setattr(mj, "mj_step", step)
                try:
                    with pytest.raises(ValueError):
                        search.predict(observation, np.full((2, 2), 0.2))
                    assert not steps, "changed preparation reached native execution"
                finally:
                    mj.set_mjcb_control(None)
                    search._environment.data = original_data
                    search._model.body_mass[:] = original_mass
                    source.write_bytes(original_bytes)
    finally:
        task.close()
