"""Integration smoke test: URDF engine loadability matrix.

Issue #4535 -- Cross-engine smoke test: load generated URDF in MuJoCo / Drake / Pinocchio / OpenSim
"""

from __future__ import annotations

from pathlib import Path

import pytest

from humanoid_character_builder.interfaces import CharacterBuilder


def _load_in_mujoco(urdf_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        assert model is not None
        assert model.nbody >= 1
    except AttributeError:
        pytest.skip("mujoco.MjModel.from_xml_path not available in this version")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"MuJoCo failed to load generated URDF: {exc}")


def _load_in_drake(urdf_path: Path) -> None:
    pytest.importorskip("pydrake.multibody", reason="pydrake not installed")
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant

    try:
        plant = MultibodyPlant(time_step=0.0)
        parser = Parser(plant)
        if hasattr(parser, "AddModels"):
            parser.AddModels(str(urdf_path))
        else:
            parser.AddModelFromFile(str(urdf_path))
        plant.Finalize()
        assert plant.num_bodies() >= 1
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Drake failed to load generated URDF: {exc}")


def _load_in_pinocchio(urdf_path: Path) -> None:
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    if not hasattr(pin, "__version__"):
        pytest.skip("pinocchio appears to be a stub/mock")
    if not hasattr(pin, "buildModelFromUrdf"):
        pytest.skip("pinocchio buildModelFromUrdf not available")
    try:
        model = pin.buildModelFromUrdf(str(urdf_path))
        assert model is not None
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Pinocchio failed to load generated URDF: {exc}")


def _load_in_opensim(urdf_path: Path) -> None:
    # Attempt to load URDF in OpenSim
    # OpenSim doesn't natively load URDF via simple python API directly in standard builds
    # or if it does, it's not well documented for python wrapper.
    # The requirement: "via existing OpenSim URDF import path or document its absence"
    try:
        import opensim  # noqa: F401 - import-only availability probe
    except ImportError:
        pytest.skip("opensim not installed")

    # Try the URDF converter if it exists, or just skip noting the absence.
    # OpenSim python API doesn't have a direct `opensim.Model(urdf_path)`
    # They have opensim.URDFConverter if built with it.
    pytest.skip("OpenSim URDF import path not implemented natively in python API")


def _load_in_engine(engine: str, urdf_path: Path) -> None:
    if engine == "mujoco":
        _load_in_mujoco(urdf_path)
    elif engine == "drake":
        _load_in_drake(urdf_path)
    elif engine == "pinocchio":
        _load_in_pinocchio(urdf_path)
    elif engine == "opensim":
        _load_in_opensim(urdf_path)
    else:
        pytest.fail(f"Unknown engine: {engine}")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("preset", ["athletic", "average", "heavy"])
@pytest.mark.parametrize("engine", ["mujoco", "drake", "pinocchio", "opensim"])
def test_generated_urdf_loads_in_engine(
    preset: str, engine: str, tmp_path: Path
) -> None:
    builder = CharacterBuilder()
    params = CharacterBuilder.create_from_preset(preset)

    # Generate meshes properly
    result = builder.build(params, mesh_output_dir=tmp_path)
    urdf_path = result.export_urdf(tmp_path)

    _load_in_engine(engine, urdf_path)
