"""Native partial abdominal reduction and adversarial XML admission."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from io import StringIO
from typing import Any

import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.opensim.python.native_abdominal_reduction import (
    AbdominalReductionRequest,
    derive_abdominal_zero_translations,
    _derive_tree,
)

pytestmark = pytest.mark.integration


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree() -> Any:
    tree = ET.parse(
        StringIO("""<Model><CustomJoint name="Abdjnt"><coordinates>
    <Coordinate name="Abs_r3"/><Coordinate name="Abs_t1"/><Coordinate name="Abs_t2"/>
    </coordinates><SpatialTransform>
    <TransformAxis name="rotation1"><coordinates>Abs_r3</coordinates><axis>0 0 1</axis>
    <LinearFunction name="function"><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
    <TransformAxis name="translation1"><coordinates>Abs_t1</coordinates><axis>1 0 0</axis>
    <LinearFunction name="function"><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
    <TransformAxis name="translation2"><coordinates>Abs_t2</coordinates><axis>0 1 0</axis>
    <LinearFunction name="function"><coefficients>1 0</coefficients></LinearFunction></TransformAxis>
    </SpatialTransform></CustomJoint><CoordinateCouplerConstraint name="Abs_r3_con">
    <dependent_coordinate_name>Abs_r3</dependent_coordinate_name>
    <independent_coordinate_names>flex_extension</independent_coordinate_names>
    <coupled_coordinates_function><PiecewiseLinearFunction/></coupled_coordinates_function>
    </CoordinateCouplerConstraint></Model>""")
    )
    return tree


def test_only_zero_translations_are_removed() -> None:
    tree = _tree()
    rotation = ET.tostring(tree.getroot().find('.//TransformAxis[@name="rotation1"]'))
    _derive_tree(tree)
    assert [
        node.get("name") for node in tree.getroot().find(".//CustomJoint/coordinates")
    ] == ["Abs_r3"]
    assert (
        ET.tostring(tree.getroot().find('.//TransformAxis[@name="rotation1"]'))
        == rotation
    )
    assert len(tree.getroot().findall(".//Constant/value")) == 2


@pytest.mark.parametrize(
    "mutation",
    ["target", "axis", "unknown_text", "unknown_attribute", "unknown_tail", "coupler"],
)
def test_unadmitted_declarations_reject(mutation: str) -> None:
    tree = _tree()
    root = tree.getroot()
    if mutation == "target":
        root.find(
            './/TransformAxis[@name="translation1"]/LinearFunction/coefficients'
        ).text = "1 .001"
    elif mutation == "axis":
        root.find('.//TransformAxis[@name="translation2"]/axis').text = "0 0 1"
    elif mutation == "coupler":
        root.find(".//CoordinateCouplerConstraint").set("name", "unknown")
    else:
        node = ET.fromstring("<unknown/>")
        root.append(node)
        if mutation == "unknown_text":
            node.text = "/jointset/Abdjnt/Abs_t1/value"
        elif mutation == "unknown_tail":
            node.tail = "Abs_t2"
        else:
            node.set("consumer", "Abs_t2")
    with pytest.raises(ValueError):
        _derive_tree(tree)


def _observations(osim: Any, source: Path, prepared: dict[str, float]) -> tuple:
    model = osim.Model(str(source))
    observations = []
    for value, speed in ((0.0, 0.0), (-0.03, 0.02), (0.03, -0.02)):
        state = model.initSystem()
        for name, initial in prepared.items():
            model.setStateVariableValue(state, name, initial)
        coordinates = model.getCoordinateSet()
        coordinates.get("flex_extension").setValue(state, value, False)
        coordinates.get("flex_extension").setSpeedValue(state, speed)
        for _ in range(model.getConstraintSet().getSize() + 1):
            for raw in model.getConstraintSet():
                coupler = osim.CoordinateCouplerConstraint.safeDownCast(raw)
                independent = coupler.getIndependentCoordinateNames()
                arguments = osim.Vector(independent.getSize(), 0.0)
                for index in range(independent.getSize()):
                    arguments.set(
                        index, coordinates.get(independent.get(index)).getValue(state)
                    )
                function = coupler.getFunction()
                dependent = coordinates.get(coupler.getDependentCoordinateName())
                dependent.setValue(state, function.calcValue(arguments), False)
                velocity = 0.0
                for index in range(independent.getSize()):
                    component = osim.StdVectorInt()
                    component.append(index)
                    velocity += function.calcDerivative(
                        component, arguments
                    ) * coordinates.get(independent.get(index)).getSpeedValue(state)
                dependent.setSpeedValue(state, velocity)
        observations.append(
            tuple((name, model.getStateVariableValue(state, name)) for name in prepared)
        )
    return tuple(observations)


@pytest.mark.parametrize("variant", ["source473", "assembled557"])
def test_native_buet_fresh_reload_preserves_complete_prepared_state(
    tmp_path: Path,
    variant: str,
) -> None:
    osim = pytest.importorskip("opensim")
    prefix = "BUET_REDUCTION" if variant == "source473" else "BUET_ASSEMBLED_REDUCTION"
    source_text = os.environ.get(prefix + "_SOURCE")
    prepared_text = os.environ.get(prefix + "_PREPARED")
    if not source_text or not prepared_text:
        pytest.skip("owned BUET source and prepared receipt opt-in absent")
    import json

    source = Path(source_text)
    payload = json.loads(Path(prepared_text).read_text(encoding="utf-8"))
    prepared = (
        payload["prepared"]["named_state"]
        if variant == "source473"
        else payload["named_state"]
    )
    states = _observations(osim, source, prepared)
    request = AbdominalReductionRequest(
        source, _sha(source), tmp_path / "derived.osim", states
    )
    receipt = derive_abdominal_zero_translations(request)
    assert receipt.removed_state_count == 4
    assert receipt.mobility_lift_rank == (52 if variant == "source473" else 60)
    assert receipt.observed_pose_count == 3
    assert max(receipt.max_errors) < 1e-9
    assert receipt.derived_sha256 == _sha(request.derived_model_path)
    assert receipt.source_sha256 == _sha(source)
    derived = osim.Model(str(request.derived_model_path))
    state = derived.initSystem()
    assert state.getNQ() == state.getNU() == (52 if variant == "source473" else 60)
    assert derived.getMuscles().getSize() == (473 if variant == "source473" else 557)
    assert derived.getConstraintSet().getSize() == 17
    study = osim.MocoStudy()
    study.updProblem().setModelAsCopy(derived)
    if variant == "source473":
        study.initCasADiSolver()
    else:
        assert {
            coordinate.getName()
            for coordinate in derived.getCoordinateSet()
            if coordinate.getLocked(state)
        } == {"subtalar_angle_r", "subtalar_angle_l", "mtp_angle_r", "mtp_angle_l"}
        with pytest.raises(RuntimeError, match="locked"):
            study.initCasADiSolver()
    with pytest.raises(FileExistsError):
        derive_abdominal_zero_translations(request)


@pytest.mark.parametrize(
    "mutation",
    [
        "lock_target",
        "speed",
        "unlock",
        "coupler_function",
        "frame",
        "observation_target",
        "state_coverage",
        "source_hash",
    ],
)
def test_native_source_and_state_mutations_reject(
    tmp_path: Path, mutation: str
) -> None:
    pytest.importorskip("opensim")
    source_text, prepared_text = (
        os.environ.get("BUET_REDUCTION_SOURCE"),
        os.environ.get("BUET_REDUCTION_PREPARED"),
    )
    if not source_text or not prepared_text:
        pytest.skip("owned BUET source and prepared receipt opt-in absent")
    import json

    original = Path(source_text)
    values = json.loads(Path(prepared_text).read_text(encoding="utf-8"))["prepared"][
        "named_state"
    ]
    tree = ET.parse(original)
    root = tree.getroot()
    if mutation in ("lock_target", "speed", "unlock"):
        coordinate = root.find(
            './/CustomJoint[@name="Abdjnt"]/coordinates/Coordinate[@name="Abs_t1"]'
        )
        field, value = {
            "lock_target": ("default_value", ".001"),
            "speed": ("default_speed_value", ".001"),
            "unlock": ("locked", "false"),
        }[mutation]
        coordinate.find(field).text = value
    elif mutation == "coupler_function":
        root.find(
            './/CoordinateCouplerConstraint[@name="Abs_r3_con"]/coupled_coordinates_function/PiecewiseLinearFunction/y'
        ).text = "1 2 3 4"
    elif mutation == "frame":
        root.find(
            './/CustomJoint[@name="Abdjnt"]/frames/PhysicalOffsetFrame/translation'
        ).text = "0 0 0"
    elif mutation == "observation_target":
        values["/jointset/Abdjnt/Abs_t1/value"] = 0.001
    elif mutation == "state_coverage":
        values.pop(next(iter(values)))
    source = tmp_path / "mutated.osim"
    tree.write(source, encoding="utf-8", xml_declaration=True)
    digest = "0" * 64 if mutation == "source_hash" else _sha(source)
    request = AbdominalReductionRequest(
        source, digest, tmp_path / "derived.osim", (tuple(values.items()),)
    )
    with pytest.raises(ValueError):
        derive_abdominal_zero_translations(request)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_native_scalar_observation_rejects(value: float) -> None:
    from src.engines.physics_engines.opensim.python.native_abdominal_reduction import (
        _scalar_error,
    )

    with pytest.raises(ValueError):
        _scalar_error(0.0, value)
    with pytest.raises(ValueError):
        _scalar_error(value, 0.0)
