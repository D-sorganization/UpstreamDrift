"""Hand visual backends must not move MuJoCo physics (#15446).

The capsule hands stay the physics baseline of the flagship export. A spec may
switch ``subject.hand_visual_backend`` (``capsule`` | ``skinned_human``,
:mod:`~src.shared.python.model_appearance.hand_spec`), which only swaps the
meshes wrapped on the club grip. Injection keeps every visual geom
non-colliding and massless through one ``default class="visual"``
(``visual_layer.py``), on top of ``<flag contact="disable"/>`` and
``inertiafromgeom="false"`` (``full_body_mjcf.py``).

This module proves the switch is observable in rendering only: identical
compiled mass and inertia, matched-state qacc within the documented tolerance,
and zero contact wrenches either way. The collision flags alone are not
trusted — the negative-control test breaks one geom's flags and requires the
guard to catch it, and the whole-model scans read the compiled flags, not the
XML.

Documented tolerances (pinned here, referenced by the issue):

* ``QACC_ATOL = 1e-9`` — matched-state qacc, the same bound the visual-layer
  invariance test already uses for the plain-vs-visual switch.
* ``WRENCH_ATOL = 0.0`` — bit-identical internal wrench streams: the flagship
  disables the contact solver, so both backends must report no contacts at all.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"

QACC_ATOL = 1e-9
WRENCH_ATOL = 0.0

#: Groups reserved for rendering: 1 = visual layer, 3 = marker spheres
#: (``visual_layer._MARKER_SPHERE_GROUP``). A group-0 geom is physics.
VISUAL_GROUPS = (1, 3)

PHYSICS_ARRAY_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_pos",
    "body_quat",
    "dof_armature",
    "dof_damping",
    "jnt_axis",
    "jnt_pos",
)


def _spec_with_backend(backend: str | None) -> dict:
    spec = json.loads(FULL_BODY.read_text(encoding="utf-8"))
    if backend is not None:
        spec["subject"]["hand_visual_backend"] = backend
    return spec


def _export(mujoco, spec: dict, **kwargs):
    xml, _meta = exporter.export_full_body_mjcf(json.dumps(spec).encode(), **kwargs)
    return mujoco.MjModel.from_xml_string(xml), xml


def _rollout(mujoco, model, steps: int = 12):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    data.qpos[:] = rng.normal(scale=0.02, size=model.nq)
    data.qvel[:] = rng.normal(scale=0.02, size=model.nv)
    trace = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        trace.append(
            (
                data.qpos.copy(),
                data.qvel.copy(),
                data.qacc.copy(),
                data.cfrc_int.copy(),
            )
        )
    assert all(np.isfinite(a).all() for step in trace for a in step)
    return trace


def _geom_name(model, index: int) -> str:
    import mujoco

    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""


def _geom_names(model) -> set[str]:
    return {name for name in (_geom_name(model, i) for i in range(model.ngeom)) if name}


def colliding_visual_geoms(model, baseline_names: set[str]) -> list[str]:
    """Names of geoms absent from the physics-only model that can collide.

    Scans *every* group, not just ``geom_group == 1``: the marker spheres land
    in group 3 and a group-1-only sweep misses them.
    """
    offenders: list[str] = []
    for i in range(model.ngeom):
        name = _geom_name(model, i)
        if not name or name in baseline_names:
            continue  # pre-existing collision geometry, unchanged
        if model.geom_group[i] in VISUAL_GROUPS:
            if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0:
                offenders.append(name)
        else:
            offenders.append(f"{name} (group {model.geom_group[i]})")
    return offenders


def _visual_hand_indices(model) -> list[int]:
    return [i for i in range(model.ngeom) if "hand" in _geom_name(model, i)]


# --- the injection policy is one default class, pin all four attributes ------


def test_visual_default_class_declares_the_invariance_policy() -> None:
    xml, _ = exporter.export_full_body_mjcf(
        json.dumps(_spec_with_backend(None)).encode(), visual=True
    )
    visual_default = [
        d
        for d in ET.fromstring(xml).iter("default")
        if d.get("class") == "visual" and d.find("geom") is not None
    ]
    assert visual_default, "the export declares no <default class='visual'> block"
    geom = visual_default[0].find("geom")
    assert geom is not None
    assert geom.get("contype") == "0"
    assert geom.get("conaffinity") == "0"
    assert geom.get("group") == "1"
    assert geom.get("mass") == "0", (
        "mass='0' dropped from the visual class — geoms inherit body mass "
        "wherever inertiafromgeom is true"
    )


# --- compiled-flag scans: the XML policy must survive compilation ------------


def test_injected_visual_geoms_never_collide() -> None:
    mujoco = pytest.importorskip("mujoco")
    spec = _spec_with_backend(None)
    plain, _ = _export(mujoco, spec)
    visual, _ = _export(mujoco, spec, visual=True)
    assert visual.ngeom > plain.ngeom
    assert colliding_visual_geoms(visual, _geom_names(plain)) == []


def test_hand_mesh_geoms_are_present_and_noncolliding() -> None:
    mujoco = pytest.importorskip("mujoco")
    visual, _ = _export(mujoco, _spec_with_backend(None), visual=True)
    hands = _visual_hand_indices(visual)
    assert hands, "the visual export injects no hand geoms at all"
    for i in hands:
        assert visual.geom_contype[i] == 0 and visual.geom_conaffinity[i] == 0
        assert visual.geom_group[i] in VISUAL_GROUPS, _geom_name(visual, i)


# --- the contact-solver caveat the invariance depends on ----------------------


def test_flagship_export_keeps_contacts_disabled() -> None:
    mujoco = pytest.importorskip("mujoco")
    model, xml = _export(mujoco, _spec_with_backend(None), visual=True)
    body = xml if isinstance(xml, bytes) else xml.encode()
    assert b'contact="disable"' in body, (
        "<flag contact='disable'/> vanished — the wrench equality below would "
        "silently become luck; re-run it against a contact-enabled builder"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


# --- backend switch, compiled: the actual #15446 guarantee -------------------


def test_explicit_capsule_backend_compiles_to_the_default_model() -> None:
    mujoco = pytest.importorskip("mujoco")
    default, _ = _export(mujoco, _spec_with_backend(None), visual=True)
    explicit, _ = _export(mujoco, _spec_with_backend("capsule"), visual=True)
    for attr in PHYSICS_ARRAY_FIELDS:
        np.testing.assert_array_equal(
            getattr(explicit, attr), getattr(default, attr), err_msg=attr
        )
    for a, b in zip(_rollout(mujoco, default), _rollout(mujoco, explicit), strict=True):
        for left, right in zip(a, b, strict=True):
            np.testing.assert_allclose(left, right, atol=WRENCH_ATOL)


@pytest.fixture(scope="module")
def backend_pair():
    """capsule + skinned_human exports, compiled once (the solve costs ~1 min)."""
    mujoco = pytest.importorskip("mujoco")
    capsule, _ = _export(mujoco, _spec_with_backend("capsule"), visual=True)
    skinned, _ = _export(mujoco, _spec_with_backend("skinned_human"), visual=True)
    return mujoco, capsule, skinned


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.xdist_group("skinned_hand_backends")
def test_backend_switch_preserves_compiled_physics(backend_pair) -> None:
    mujoco, capsule, skinned = backend_pair
    assert skinned.nbody == capsule.nbody
    assert skinned.nq == capsule.nq and skinned.nv == capsule.nv
    for attr in PHYSICS_ARRAY_FIELDS:
        np.testing.assert_array_equal(
            getattr(skinned, attr), getattr(capsule, attr), err_msg=attr
        )
    assert colliding_visual_geoms(skinned, _geom_names(capsule)) == []
    for a, b in zip(_rollout(mujoco, capsule), _rollout(mujoco, skinned), strict=True):
        for left, right in zip(a[:3], b[:3], strict=True):
            np.testing.assert_allclose(left, right, atol=QACC_ATOL)
        # cfrc_int: no contacts exist, so the internal wrench stream must match
        # bit for bit — any drift means a visual geom is transmitting force.
        np.testing.assert_array_equal(a[3], b[3])


# --- the flags alone are not trusted: prove the guard can fail ---------------


def test_guard_catches_a_hand_geom_that_keeps_colliding() -> None:
    """Negative control: break the flags, require the guard to name it.

    With ``<flag contact="disable"/>`` a straying contype cannot move physics,
    so a flag-blind invariance test would stay green forever. This pins that
    :func:`colliding_visual_geoms` is the half of #15446 doing the policing.
    """
    mujoco = pytest.importorskip("mujoco")
    spec = copy.deepcopy(_spec_with_backend(None))
    plain, _ = _export(mujoco, spec)
    _, xml = _export(mujoco, spec, visual=True)
    root = ET.fromstring(xml)
    tampered = 0
    for geom in root.iter("geom"):
        if "hand" in (geom.get("name") or ""):
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            tampered += 1
    assert tampered, "no hand geom in the MJCF to tamper with"
    model = mujoco.MjModel.from_xml_string(ET.tostring(root))
    offenders = colliding_visual_geoms(model, _geom_names(plain))
    assert offenders, "the guard stayed silent while hand geoms were made collidable"
    assert any("hand" in o for o in offenders), offenders
