"""The generated MuJoCo golfer touches only the floor, never itself (#11811).

Without collision filtering, non-adjacent segment geoms overlapped at the
initial pose (70 contacts), launching the model off the floor. Checked on both
a fresh export and the committed ``models/generated/golfer.xml``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

mujoco = pytest.importorskip("mujoco")

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.model_converter.build_models import (  # noqa: E402
    DEFAULT_CANONICAL_YAML,
    DEFAULT_MJCF_OUT,
)
from tools.model_converter.mjcf_exporter import export_mjcf  # noqa: E402
from tools.model_converter.schema_validator import (  # noqa: E402
    validate_canonical_model,
)

#: Pelvis lift that clears every foot geom off the floor.
LIFT_M = 1.0


def _exported() -> str:
    return export_mjcf(validate_canonical_model(DEFAULT_CANONICAL_YAML))


def _committed() -> str:
    return DEFAULT_MJCF_OUT.read_text(encoding="utf-8")


@pytest.fixture(params=[_exported, _committed], ids=["exported", "committed"])
def golfer(request):
    model = mujoco.MjModel.from_xml_string(request.param())
    return model, mujoco.MjData(model)


def test_lifted_golfer_has_no_contacts(golfer) -> None:
    model, data = golfer
    data.qpos[2] += LIFT_M
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_every_contact_at_the_initial_pose_involves_the_floor(golfer) -> None:
    model, data = golfer
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    data.qpos[2] -= 0.02  # press the feet into the floor
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    for i in range(data.ncon):
        contact = data.contact[i]
        assert floor in (contact.geom1, contact.geom2)
