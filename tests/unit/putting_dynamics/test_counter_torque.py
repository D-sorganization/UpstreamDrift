"""Counter-torque stabilization model in the shared core (#14469, epic #14452).

Golden parity: the core reproduces the studio's closed-form couple model
(``js/state.js`` ``evalPhysicsAtTime``, run under node) sample for sample,
so the studio is drift-gated against the core. Closure: with no noise the
two hands' couples carry exactly the moment the stroke requires, the
putting form of the two-hand couple check in ``grip_contact/couple_check.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    PUTTER_FACE_COR,
)
from src.shared.python.putting_dynamics import counter_torque as ct

pytestmark = pytest.mark.unit

_JS_DIR = Path(__file__).resolve().parents[3] / "src" / "apps" / "putting_studio" / "js"
_NODE = shutil.which("node")
_requires_node = pytest.mark.skipif(_NODE is None, reason="node is not on PATH")

#: core field -> studio field of ``evalPhysicsAtTime``'s result.
_JS_FIELDS = {
    "theta_deg": "thDeg",
    "phi_preload_deg": "phiPreDeg",
    "phi_slack_deg": "phiSlkDeg",
    "tau_net_nm": "tauNet",
    "tau_off_required_nm": "tauOffReq",
    "tau_left_in_nm": "tauL_in",
    "tau_right_in_nm": "tauR_in",
    "tau_left_off_nm": "tauL_off",
    "tau_right_off_nm": "tauR_off",
    "force_left_in_n": "FL_in",
    "force_right_in_n": "FR_in",
    "force_left_off_n": "FL_off",
    "force_right_off_n": "FR_off",
    "face_angle_impact_deg": "faceAngleImpDeg",
    "club_path_impact_deg": "clubPathImpDeg",
}

CASES = [
    ct.CounterTorqueInputs(),
    ct.CounterTorqueInputs(0.2, 0.9, 0.8, 0.6, 74.0, "cb"),
    ct.CounterTorqueInputs(noise_scale=0.0, gust_impulse_nm=0.0, putter="lab"),
]
TIMES = [0.0, 0.2, 0.6, 0.7, 0.72, 0.8168, 0.9, 1.05, 1.6]


def _studio(inputs: ct.CounterTorqueInputs) -> list[dict[str, float]]:
    """``evalPhysicsAtTime`` at :data:`TIMES` with the studio's state set to ``inputs``."""
    source = (_JS_DIR / "state.js").read_text(encoding="utf-8")
    script = (
        f"const PuttingPhysics = require({json.dumps(str(_JS_DIR / 'putting_physics.js'))});\n"
        f"{source}\n"
        f"state.noiseScale = {inputs.noise_scale};\n"
        f"state.tauOffPreload = {inputs.off_preload_nm};\n"
        f"state.pertImpulseNm = {inputs.gust_impulse_nm};\n"
        f"state.putterType = {json.dumps(inputs.putter)};\n"
        f"const out = {json.dumps(TIMES)}.map(t => evalPhysicsAtTime("
        f"t, {inputs.preload_nm}, {inputs.lie_deg}));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    assert _NODE is not None
    result = subprocess.run(
        [_NODE, "-"],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(result.stdout)


@_requires_node
@pytest.mark.parametrize("inputs", CASES, ids=["default", "loose-cb", "quiet-lab"])
def test_core_matches_the_studio_model(inputs: ct.CounterTorqueInputs) -> None:
    for t, studio in zip(TIMES, _studio(inputs), strict=True):
        sample = asdict(ct.evaluate(t, inputs))
        for core_key, js_key in _JS_FIELDS.items():
            assert sample[core_key] == pytest.approx(studio[js_key], abs=1e-12), (
                t,
                core_key,
            )


def test_hand_couples_carry_the_required_moment_without_noise() -> None:
    quiet = ct.CounterTorqueInputs(noise_scale=0.0, gust_impulse_nm=0.0)
    for sample in ct.stroke_series(quiet):
        in_plane = sample.tau_left_in_nm + sample.tau_right_in_nm
        off_plane = sample.tau_left_off_nm + sample.tau_right_off_nm
        assert in_plane == pytest.approx(sample.tau_net_nm, abs=1e-12)
        assert off_plane == pytest.approx(sample.tau_off_required_nm, abs=1e-12)
        lever = ct.HAND_LEVER_M
        assert sample.force_right_in_n * lever == pytest.approx(sample.tau_right_in_nm)
        assert -sample.force_left_in_n * lever == pytest.approx(sample.tau_left_in_nm)


def test_post_impact_head_speed_ratio_is_the_restitution_law() -> None:
    m, e = float(GOLF_BALL_MASS_KG), float(PUTTER_FACE_COR)
    head = ct.STUDIO_HEAD_MASS_KG
    assert ct.post_impact_speed_ratio() == pytest.approx((head - e * m) / (head + m))


def test_preload_damps_the_gust_response() -> None:
    def peak(preload: float) -> float:
        inputs = ct.CounterTorqueInputs(preload_nm=preload, noise_scale=0.0)
        return max(abs(s.phi_preload_deg) for s in ct.stroke_series(inputs))

    slack = max(
        abs(s.phi_slack_deg)
        for s in ct.stroke_series(ct.CounterTorqueInputs(noise_scale=0.0))
    )
    assert peak(1.2) < peak(0.3) < slack
    assert peak(0.0) == pytest.approx(slack)


def test_series_spans_the_stroke_on_the_studio_grid() -> None:
    series = ct.stroke_series(ct.CounterTorqueInputs(), n_points=210)
    times = np.array([s.t_s for s in series])
    assert len(series) == 211
    np.testing.assert_allclose(times, np.linspace(0.0, ct.T_STROKE_S, 211))


@pytest.mark.parametrize(
    "overrides",
    [
        {"preload_nm": -0.1},
        {"off_preload_nm": 3.0},
        {"noise_scale": -1.0},
        {"gust_impulse_nm": 5.0},
        {"lie_deg": 40.0},
        {"putter": "anser"},
    ],
)
def test_bad_inputs_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ct.CounterTorqueInputs(**overrides)  # type: ignore[arg-type]
