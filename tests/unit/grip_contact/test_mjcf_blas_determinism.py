"""The generated grip MJCF must not depend on the BLAS kernel (#13720).

``test_old_pad_path_mjcf_is_byte_identical`` passed under the default OpenBLAS
kernel but failed under ``OPENBLAS_CORETYPE=Prescott``: the modal damping design
(LAPACK ``eigh`` + ``nnls``) changed the last ulp of ``solref`` damping.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.optimize import nnls

from src.shared.python.grip_contact.deterministic_linalg import (
    det_eigh_generalized,
    det_matmul,
    det_nnls,
    det_quad_form,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]

_CHILD = """
import json, sys
from pathlib import Path
from unittest.mock import MagicMock
sys.modules.setdefault("mujoco", MagicMock())
from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.pad_contact import build_pad_model
from src.engines.physics_engines.mujoco.python import grip_contact_sim as m
m.ClubInHands._rebuild = lambda self: None
m.WeldClubKinematics = lambda *a: None
raw = Path("docs/development/full_body_models/full_body_spec_anthro_driver.json").read_bytes()
spec = json.loads(raw)
iface = GripInterface.from_spec(spec)
pads = build_pad_model(iface, 1100.0)
sim = m.ClubInHands(raw, spec["coordinate_order"], iface, pads)
text = m.contact_mjcf(sim.club, sim.gravity, sim.timestep_s, sim.interface,
                      sim.pads, sim.solref, sim.friction_time_s, sim.pad_sides)
sys.stdout.buffer.write(text.encode("utf-8"))
"""


def _generate(coretype: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_CORETYPE"}
    if coretype:
        env["OPENBLAS_CORETYPE"] = coretype
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    done = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=ROOT,
        env=env,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]
    return done.stdout.decode("utf-8")


def _first_difference(a: str, b: str) -> str:
    # Tokenise on tag boundaries so the report names the differing element.
    left, right = a.replace("<", "\n<").split("\n"), b.replace("<", "\n<").split("\n")
    for x, y in zip(left, right, strict=False):
        if x != y:
            return f"first differing line:\n- {x}\n+ {y}"
    diff = "\n".join(difflib.unified_diff(left, right, lineterm="", n=0))
    return f"length differs:\n{diff[:500]}"


def test_mjcf_is_identical_under_prescott_openblas_kernel() -> None:
    default = _generate(None)
    prescott = _generate("Prescott")
    assert default == prescott, _first_difference(default, prescott)


def _spd(seed: int, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_det_eigh_generalized_matches_scipy(seed: int) -> None:
    k, m = _spd(seed), _spd(seed + 10)
    w2, vec = det_eigh_generalized(k, m)
    ref_w2, _ = eigh(k, m)
    np.testing.assert_allclose(w2, ref_w2, rtol=1e-10)
    np.testing.assert_allclose(vec.T @ m @ vec, np.eye(6), atol=1e-10)
    np.testing.assert_allclose(k @ vec, m @ vec * w2, rtol=1e-8, atol=1e-8)


def test_det_eigh_generalized_rejects_indefinite_mass() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        det_eigh_generalized(np.eye(2), -np.eye(2))


def test_det_nnls_matches_scipy_with_active_constraint() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6))
    b = rng.normal(size=6)
    x, res = det_nnls(a, b)
    ref_x, ref_res = nnls(a, b)
    assert (x >= 0.0).all()
    np.testing.assert_allclose(x, ref_x, atol=1e-9)
    assert res == pytest.approx(ref_res, abs=1e-9)


def test_det_matmul_and_quad_form_match_numpy() -> None:
    a, b = _spd(4, 3), _spd(5, 3)
    np.testing.assert_allclose(det_matmul(a, b), a @ b, rtol=1e-12)
    v = np.arange(1.0, 7.0)
    m = _spd(6)
    assert det_quad_form(v, m) == pytest.approx(float(v @ m @ v), rel=1e-12)
