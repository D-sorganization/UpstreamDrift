"""MyoSuite runtime-layout contract (#12635).

MyoSuite 3.0.0 removed ``myosuite.envs.env_base``.  These tests import the
runtime the code actually uses from the *installed* MyoSuite.  They skip only
when MyoSuite is absent; when it is present but broken (an unsupported
layout) they FAIL, so a future breaking release cannot hide behind a skip.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from src.engines.physics_engines.myosuite.python import (
    myosuite_runtime_compat as compat,
)

pytestmark = pytest.mark.unit

MINIMAL_MJCF = (
    '<mujoco><worldbody><body name="b"><joint name="j" type="hinge"/>'
    '<geom size="0.1" mass="1"/></body></worldbody></mujoco>'
)

myosuite_present = find_spec("myosuite") is not None
requires_myosuite = pytest.mark.skipif(
    not myosuite_present, reason="myosuite is not installed"
)


@requires_myosuite
def test_installed_myosuite_has_a_supported_runtime_layout() -> None:
    """Present+broken must fail loudly, never skip."""
    layout = compat.detect_runtime_layout()
    assert layout in compat.SUPPORTED_LAYOUTS


@requires_myosuite
def test_load_runtime_holds_spec_model_and_data() -> None:
    runtime = compat.load_runtime_from_xml(MINIMAL_MJCF)
    assert runtime.mj_model.nq == 1
    assert runtime.mj_data.qpos.shape == (1,)
    assert runtime.mj_spec is not None


@requires_myosuite
def test_runtime_matches_the_physics_adapter_call_site() -> None:
    from src.engines.physics_engines.myosuite.python.full_body_parity import (
        load_myosuite_runtime,
    )

    env = load_myosuite_runtime(MINIMAL_MJCF)
    assert env.mj_model is not None and env.mj_data is not None
    assert env.mj_spec is not None


def test_unsupported_layout_fails_with_actionable_message(monkeypatch) -> None:
    def _missing(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(compat, "import_module", _missing)
    with pytest.raises(compat.MyoSuiteLayoutError, match=r"myosuite.*#12635"):
        compat.detect_runtime_layout()


def test_empty_xml_violates_precondition() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compat.load_runtime_from_xml("  ")
