"""Player presets must use the current golfer and preserve skin-only physics."""

import json

import numpy as np
import pytest

from src.shared.python.model_appearance.player_models import (
    build_player_spec,
    player_appearance,
    player_profile,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "player,height,mass",
    [
        ("tiger_woods", 1.8542, 83.91458845),
        ("ben_hogan", 1.7272, 68.0388555),
    ],
)
def test_player_dimensions_and_current_model(player, height, mass):
    """Check selected priors, mass, segment distances and current topology."""
    from src.shared.python.model_appearance.canonical_golfer_gate import (
        require_canonical_golfer,
    )

    profile = player_profile(player)
    assert profile["stature_m"] == height
    assert profile["mass_kg"] == mass
    assert profile["sources"] and profile["limitations"]
    spec = build_player_spec(player)
    require_canonical_golfer(spec)
    human_mass = (
        sum(s["mass_kg"] for b in spec["bodies"] for s in b["solids"])
        - spec["club"]["total_mass_kg"]
    )
    assert human_mass == pytest.approx(mass, abs=1e-9)
    assert spec["subject"]["stature_m"] == height
    assert spec["subject"]["neutral_visual_calibration"][
        "neutral_visual_height_m"
    ] == pytest.approx(height, abs=0.0001)
    from src.shared.python.motion_matching.anthropometric_candidate import (
        zero_pose_joint_positions,
    )
    from src.shared.python.motion_matching.anthropometry import segment_parameters
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    assert spec["coordinate_order"] == reference_spec("driver")["coordinate_order"]
    positions = zero_pose_joint_positions(spec)
    for side in ("l", "r"):
        for segment, proximal, distal in (
            ("thigh", "hip", "knee"),
            ("shank", "knee", "ankle"),
        ):
            actual = np.linalg.norm(
                positions[f"{distal}_{side}"] - positions[f"{proximal}_{side}"]
            )
            assert actual == pytest.approx(
                segment_parameters(height, mass, segment).length_m, abs=1e-10
            )


def test_unknown_player_is_rejected():
    """Reject unrecognized profiles without guessing."""
    with pytest.raises(ValueError, match="Unknown player"):
        player_profile("../tiger_woods")


def test_distinct_faces_and_profile_reads_are_isolated():
    """Keep player faces distinct and catalog reads independent."""
    tiger = player_appearance("tiger_woods")
    hogan = player_appearance("ben_hogan")
    assert tiger.head.morph != hogan.head.morph
    changed = player_profile("tiger_woods")
    changed["mass_kg"] = 1
    assert player_profile("tiger_woods")["mass_kg"] == 83.91458845


def test_skin_export_retains_supplied_spec(tmp_path, monkeypatch):
    """Keep every supplied physics field during a skin-only export."""
    from src.shared.python.model_appearance import player_models
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    spec = reference_spec("driver")
    source = tmp_path / "base.json"
    source.write_text(json.dumps(spec))

    def reject_rebuild(*args, **kwargs):
        """Fail if appearance-only export attempts anatomy construction."""
        raise AssertionError("Skin-only export must never rebuild anatomy")

    monkeypatch.setattr(player_models, "build_player_spec", reject_rebuild)
    result = player_models.export_player(
        "tiger_woods", tmp_path / "skin", base_spec=source
    )
    assert result.is_file()
    assert json.loads((result.parent / "model.json").read_text()) == spec
    receipt = json.loads((result.parent / "receipt.json").read_text())
    assert receipt["mode"] == "appearance_only"
    assert "appearance_override" not in receipt
    visual = json.loads((result.parent / "model.visual.json").read_text())
    assert visual.pop("appearance")["name"] == "tiger_woods_stylized"
    assert visual == spec


def test_cap_style_override_preserves_physics_and_records_override(
    tmp_path, monkeypatch
):
    """Apply cap styling without rewriting source profile or base physics."""
    from src.shared.python.model_appearance import player_models
    from src.shared.python.model_appearance.schema import physics_spec_sha256
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )
    from src.engines.physics_engines.mujoco.python import full_body_mjcf

    spec = reference_spec("driver")
    source = tmp_path / "base.json"
    source.write_text(json.dumps(spec))
    monkeypatch.setattr(
        full_body_mjcf,
        "export_full_body_mjcf",
        lambda raw, appearance=None: ("<mujoco/>", {"mocked": True}),
    )

    xml = player_models.export_player(
        "sam_snead", tmp_path / "flat-cap", base_spec=source, cap_style="flat_cap"
    )

    assert xml.read_text() == "<mujoco/>"
    assert json.loads((xml.parent / "model.json").read_text()) == spec
    appearance = json.loads((xml.parent / "model.appearance.json").read_text())
    assert appearance["head"]["cap_style"] == "flat_cap"
    assert appearance["head"]["cap_material"] == "cap_white"
    assert appearance["head"]["cap_brim"] == 0.12
    receipt = json.loads((xml.parent / "receipt.json").read_text())
    assert receipt["physics_spec_sha256"] == physics_spec_sha256(spec)
    assert receipt["profile"]["appearance"]["head"]["cap_style"] == "straw_hat"
    assert receipt["appearance_override"] == {
        "head.cap_style": "flat_cap",
        "head.cap_material": "cap_white",
        "head.cap_brim": 0.12,
    }


def test_unknown_cap_style_fails_before_output_creation(tmp_path):
    """Reject unknown cap styles before creating export destinations."""
    from src.shared.python.model_appearance.player_models import export_player

    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        export_player("tiger_woods", output, cap_style="invented")
    assert not output.exists()


def test_cli_forwards_cap_style_to_exporter(tmp_path, monkeypatch):
    """Forward a selected cap override to each fresh appearance export."""
    from scripts import export_player_models

    received = []

    def fake_export(player, folder, *, base_spec=None, cap_style=None):
        """Record the override and create the minimum exporter outputs."""
        received.append(cap_style)
        folder.mkdir(parents=True)
        (folder / "model.json").write_text(json.dumps({"coordinate_order": []}))
        xml = folder / "model.xml"
        xml.write_text("<mujoco/>")
        return xml

    monkeypatch.setattr(export_player_models, "_common_camera_distance", lambda _: 4.0)
    monkeypatch.setattr(
        export_player_models,
        "_render",
        lambda _xml, out, *_args: (out.mkdir(parents=True), {"previews": {}})[1],
    )
    from src.shared.python.model_appearance import player_models

    monkeypatch.setattr(player_models, "export_player", fake_export)
    export_player_models.main(
        [
            "--player",
            "tiger_woods",
            "--output",
            str(tmp_path),
            "--cap-style",
            "flat_cap",
        ]
    )
    assert received == ["flat_cap"]


def test_cli_rejects_cap_override_for_render_existing_before_writes(tmp_path):
    """Reject applying a new cap to XML that is only being rendered."""
    from scripts.export_player_models import main

    output = tmp_path / "render-existing"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--player",
                "tiger_woods",
                "--output",
                str(output),
                "--render-existing",
                "--cap-style",
                "flat_cap",
            ]
        )
    assert error.value.code == 2
    assert not output.exists()


def test_skin_preserves_native_physics_and_joint_order():
    """Check compiled dynamics and forward kinematics with both player skins."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=doc)[0])
        for doc in (
            None,
            player_appearance("tiger_woods"),
            player_appearance("ben_hogan"),
        )
    ]
    base = models[0]
    for other in models[1:]:
        for attr in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "jnt_type",
            "jnt_qposadr",
            "jnt_dofadr",
            "jnt_range",
            "qpos0",
        ):
            np.testing.assert_array_equal(getattr(base, attr), getattr(other, attr))
        assert [base.joint(i).name for i in range(base.njnt)] == [
            other.joint(i).name for i in range(other.njnt)
        ]
        for phase in (0.0, 0.03, -0.04):
            states = []
            for model in (base, other):
                data = mujoco.MjData(model)
                data.qpos[:] += phase
                data.qvel[:] = phase
                mujoco.mj_forward(model, data)
                states.append(data)
            for attr in ("qM", "qfrc_bias", "qacc", "xpos", "xmat"):
                np.testing.assert_array_equal(
                    getattr(states[0], attr), getattr(states[1], attr)
                )
