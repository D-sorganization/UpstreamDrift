"""CMB-12 (#11663): the tutorial's MuJoCo simulate step matches a real run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.humanoid_character_builder.spec_export import (
    compile_character,
    export_character,
)

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit

TUTORIAL = (
    Path(__file__).resolve().parents[3]
    / "docs/user_guide/character_builder_tutorial.md"
)

STEPS = 100


@pytest.fixture(scope="module")
def tutorial_text() -> str:
    return TUTORIAL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def stepped(tmp_path_factory):
    compiled = compile_character("tour_average_male", {"mass_kg": 90.0})
    text, _, filename = export_character(compiled, "mjcf")
    path = tmp_path_factory.mktemp("mjcf") / filename
    path.write_text(text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
    return filename, model, data


def test_simulate_step_matches_tutorial(stepped, tutorial_text: str) -> None:
    filename, model, data = stepped
    assert filename == "tour_average_male_2bba2beb.xml"
    assert (model.nq, model.nu, model.opt.timestep) == (44, 0, 0.002)
    assert np.isfinite(data.qpos).all()
    assert data.time == pytest.approx(0.2)
    assert data.ncon == 0
    for documented in ("44 coordinates", "no actuators", "0.2 s", "-0.198"):
        assert documented in tutorial_text
    assert "was not run" not in tutorial_text


def test_unsupported_body_falls_freely(stepped) -> None:
    """With no floor and no actuators, the root drops as in free fall."""
    _, model, data = stepped
    drop = -data.qpos[2]
    free_fall = 0.5 * -model.opt.gravity[2] * data.time**2
    assert f"{-drop:.3f}" == "-0.198"
    assert drop == pytest.approx(free_fall, rel=0.02)
