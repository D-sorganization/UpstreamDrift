"""Guarded XML-derived removal of BUET's two zero abdominal translations.

No source parameter, muscle state, retained rotation or constraint is repaired.
Caller-supplied complete observation states are measured without equilibrium,
assembly or projection. Sampled mechanics do not qualify capture matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from defusedxml import ElementTree

from src.engines.physics_engines.opensim.python.native_mtp_reduction import _pose_error

_REMOVED = ("Abs_t1", "Abs_t2")
_REFERENCE = re.compile(r"(?<![A-Za-z0-9_])(?:Abs_t1|Abs_t2)(?![A-Za-z0-9_])")
_RECIPE = "buet-abdjnt-zero-translations/1"
# Exact reviewed native property declarations; unrelated body/force properties
# are retained and compared independently, not inferred from these identities.
_REVIEWED_JOINT = "1760cd3b6a7aa868e3e727526667d784f335e5f5a582bfa1e89d59ee0ee7251c"
_REVIEWED_COUPLER = "f01454d23ef710cf5c898d8f709b0128359e52091df4e4724f4aa08f4f95ec24"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AbdominalReductionRequest:
    """Exact source, fresh derived artifact and complete immutable observations."""

    source_model_path: Path
    source_sha256: str
    derived_model_path: Path
    comparison_states: tuple[tuple[tuple[str, float], ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_model_path, Path) or not isinstance(
            self.derived_model_path, Path
        ):
            raise TypeError("source and output must be Path objects")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source SHA-256 must be lowercase hex")
        if self.source_model_path.resolve() == self.derived_model_path.resolve():
            raise ValueError("source and derived paths must differ")
        if not isinstance(self.comparison_states, tuple) or not self.comparison_states:
            raise ValueError("explicit complete observation states required")
        for state in self.comparison_states:
            if not isinstance(state, tuple) or not state:
                raise ValueError("immutable nonempty named state required")
            names = []
            for row in state:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise ValueError("immutable named state pairs required")
                name, value = row
                if (
                    not isinstance(name, str)
                    or not name
                    or isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("finite named state required")
                names.append(name)
            if len(set(names)) != len(names):
                raise ValueError("duplicate named observation state")


@dataclass(frozen=True)
class AbdominalReductionReceipt:
    """Bound source/derived/recipe/runtime and sampled mechanical observations."""

    recipe: str
    source_sha256: str
    derived_sha256: str
    reducer_sha256: str
    comparison_helper_sha256: str
    observation_sha256: str
    native_version: str
    native_binary_sha256: tuple[tuple[str, str], ...]
    removed_state_count: int
    mobility_lift_rank: int
    mobility_lift_condition: float
    observed_pose_count: int
    max_errors: tuple[float, ...]


def _scalar_error(left: float, right: float) -> float:
    difference = abs(left - right)
    if not all(math.isfinite(value) for value in (left, right, difference)):
        raise ValueError("nonfinite native scalar comparison")
    return difference


def _required(parent: Any, expression: str) -> Any:
    node = parent.find(expression)
    if node is None:
        raise ValueError(f"required abdominal declaration absent: {expression}")
    return node


def _admit_axis(joint: Any, index: int, name: str) -> Any:
    axis = _required(
        joint, f"SpatialTransform/TransformAxis[@name='translation{index}']"
    )
    if (_required(axis, "coordinates").text or "").split() != [name]:
        raise ValueError("translation coordinate declaration changed")
    expected = (1.0, 0.0, 0.0) if index == 1 else (0.0, 1.0, 0.0)
    if (
        tuple(float(value) for value in (_required(axis, "axis").text or "").split())
        != expected
    ):
        raise ValueError("translation axis changed")
    function = _required(axis, "LinearFunction")
    if tuple(
        float(value)
        for value in (_required(function, "coefficients").text or "").split()
    ) != (1.0, 0.0):
        raise ValueError("translation function must be exact identity")
    if [node.tag for node in axis] != ["coordinates", "axis", "LinearFunction"]:
        raise ValueError("unknown translation-axis declaration")
    return axis


def _derive_tree(tree: Any) -> None:
    root = tree.getroot()
    matches = root.findall(".//CustomJoint[@name='Abdjnt']")
    if len(matches) != 1:
        raise ValueError("exactly one abdominal CustomJoint required")
    joint = matches[0]
    coordinates = _required(joint, "coordinates")
    if [node.get("name") for node in coordinates] != ["Abs_r3", *_REMOVED]:
        raise ValueError("abdominal coordinate ownership changed")
    coupler = _required(root, ".//CoordinateCouplerConstraint[@name='Abs_r3_con']")
    if (
        _required(coupler, "dependent_coordinate_name").text or ""
    ).strip() != "Abs_r3" or (
        _required(coupler, "independent_coordinate_names").text or ""
    ).split() != ["flex_extension"]:
        raise ValueError("surviving abdominal coupling changed")
    _required(coupler, "coupled_coordinates_function/PiecewiseLinearFunction")
    admitted_nodes = {node for node in coordinates if node.get("name") in _REMOVED}
    axes = tuple(
        _admit_axis(joint, index, name) for index, name in enumerate(_REMOVED, 1)
    )
    admitted_nodes.update(_required(axis, "coordinates") for axis in axes)
    # Scan every unadmitted field, including attributes and tails, for unknown consumers.
    for node in root.iter():
        if node not in admitted_nodes and any(
            _REFERENCE.search(value)
            for value in (*node.attrib.values(), node.text or "")
        ):
            raise ValueError("unknown removed-coordinate consumer")
        if _REFERENCE.search(node.tail or ""):
            raise ValueError("unknown removed-coordinate tail consumer")
    for coordinate in list(coordinates)[1:]:
        coordinates.remove(coordinate)
    for axis in axes:
        _required(axis, "coordinates").text = ""
        axis.remove(_required(axis, "LinearFunction"))
        axis.append(
            ElementTree.fromstring(
                '<Constant name="function"><value>0</value></Constant>'
            )
        )


def _native_admission(model: Any, state: Any) -> None:
    for component, expected in (
        (model.getJointSet().get("Abdjnt"), _REVIEWED_JOINT),
        (model.getConstraintSet().get("Abs_r3_con"), _REVIEWED_COUPLER),
    ):
        if hashlib.sha256(component.dump().encode()).hexdigest() != expected:
            raise ValueError("reviewed abdominal joint or coupling properties changed")
    coordinates = model.getCoordinateSet()
    for name in _REMOVED:
        coordinate = coordinates.get(name)
        if not coordinate.getDefaultLocked() or not coordinate.getLocked(state):
            raise ValueError("source abdominal translation must remain locked")
        if coordinate.getDefaultIsPrescribed() or coordinate.isPrescribed(state):
            raise ValueError("prescribed abdominal translation is unsupported")
        values = (
            coordinate.getDefaultValue(),
            coordinate.getValue(state),
            coordinate.getDefaultSpeedValue(),
            coordinate.getSpeedValue(state),
        )
        if any(value != 0.0 for value in values):
            raise ValueError(
                "abdominal source/default/runtime target and speed must be exactly zero"
            )


def _names(model: Any) -> tuple[str, ...]:
    values = model.getStateVariableNames()
    return tuple(values.get(index) for index in range(values.getSize()))


def _verify_inventory(source: Any, derived: Any) -> tuple[str, ...]:
    for method in ("getBodySet", "getForceSet", "getConstraintSet", "getMarkerSet"):
        before = tuple(value.dump() for value in getattr(source, method)())
        after = tuple(value.dump() for value in getattr(derived, method)())
        if before != after:
            raise ValueError("retained native component properties changed")
    for joint in source.getJointSet():
        matched = derived.getJointSet().get(joint.getName())
        if joint.getName() != "Abdjnt" and joint.dump() != matched.dump():
            raise ValueError("unrelated joint changed")
    for coordinate in derived.getCoordinateSet():
        if (
            coordinate.dump()
            != source.getCoordinateSet().get(coordinate.getName()).dump()
        ):
            raise ValueError("retained coordinate properties changed")
    removed = {
        f"/jointset/Abdjnt/{name}/{kind}"
        for name in _REMOVED
        for kind in ("value", "speed")
    }
    before_names, after_names = set(_names(source)), set(_names(derived))
    if before_names - after_names != removed or after_names - before_names:
        raise ValueError("unexpected continuous-state ownership change")
    return tuple(coordinate.getName() for coordinate in derived.getCoordinateSet())


def _measure_states(
    source: Any, derived: Any, request: AbdominalReductionRequest
) -> tuple[tuple[float, ...], int, float]:
    names = _verify_inventory(source, derived)
    source_names, derived_names = _names(source), _names(derived)
    maxima = [0.0] * 10
    rank = 0
    condition = 1.0
    for observation in request.comparison_states:
        values = dict(observation)
        if set(values) != set(source_names):
            raise ValueError("complete exact source continuous-state coverage required")
        source_state, derived_state = source.initSystem(), derived.initSystem()
        for name in _REMOVED:
            if any(
                values[f"/jointset/Abdjnt/{name}/{kind}"] != 0.0
                for kind in ("value", "speed")
            ):
                raise ValueError(
                    "declared removed translation state must be exactly zero"
                )
        for name in source_names:
            source.setStateVariableValue(source_state, name, values[name])
        if any(
            source.getStateVariableValue(source_state, name) != values[name]
            for name in source_names
        ):
            raise ValueError("source native state restoration differs from declaration")
        _native_admission(source, source_state)
        for name in derived_names:
            derived.setStateVariableValue(derived_state, name, values[name])
            if derived.getStateVariableValue(
                derived_state, name
            ) != source.getStateVariableValue(source_state, name):
                raise ValueError("common continuous state restoration changed")
        errors = _pose_error(source, derived, names, source_state, derived_state)
        if not all(math.isfinite(value) and value >= 0.0 for value in errors):
            raise ValueError("nonfinite or negative native mechanical observation")
        for index, value in enumerate(errors[:8]):
            maxima[index] = max(maxima[index], value)
        rank, condition = errors[8], max(condition, errors[9])
        for muscle in source.getMuscles():
            matched = derived.getMuscles().get(muscle.getName())
            maxima[8] = max(
                maxima[8],
                _scalar_error(
                    muscle.getLengtheningSpeed(source_state),
                    matched.getLengtheningSpeed(derived_state),
                ),
            )
            maxima[9] = max(
                maxima[9],
                _scalar_error(
                    muscle.getActuation(source_state),
                    matched.getActuation(derived_state),
                ),
            )
        if rank != derived_state.getNU() or not all(
            math.isfinite(value) and value <= 1e-9 for value in maxima
        ):
            raise ValueError(
                "fresh derived sampled mechanics differ or lift is rank deficient"
            )
    return tuple(maxima), rank, condition


def derive_abdominal_zero_translations(
    request: AbdominalReductionRequest,
) -> AbdominalReductionReceipt:
    """Create a fresh reduced artifact and verify caller-declared native states.

    Failed derived artifacts are retained for diagnosis; existing outputs are
    never overwritten. Receipt describes sampled mechanical equivalence only.
    """
    import json
    import opensim as osim
    from src.engines.physics_engines.opensim.python import native_mtp_reduction

    source, output = request.source_model_path, request.derived_model_path
    if not source.is_file() or _sha(source) != request.source_sha256:
        raise ValueError("source digest mismatch or source missing")
    if output.exists():
        raise FileExistsError(output)
    reducer_sha = _sha(Path(__file__))
    helper_sha = _sha(Path(native_mtp_reduction.__file__))
    tree = ElementTree.parse(source)
    _derive_tree(tree)
    original = osim.Model(str(source))
    _native_admission(original, original.initSystem())
    with output.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    derived_sha = _sha(output)
    derived = osim.Model(str(output))
    derived.initSystem()
    errors, rank, condition = _measure_states(original, derived, request)
    if (
        _sha(source) != request.source_sha256
        or _sha(output) != derived_sha
        or _sha(Path(__file__)) != reducer_sha
        or _sha(Path(native_mtp_reduction.__file__)) != helper_sha
    ):
        raise ValueError(
            "source or reduction implementation changed during verification"
        )
    observation_sha = hashlib.sha256(
        json.dumps(
            request.comparison_states, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    binaries = tuple(
        (name, _sha(Path(getattr(osim, name).__file__)))
        for name in ("_simulation", "_simbody", "_actuators")
    )
    return AbdominalReductionReceipt(
        _RECIPE,
        request.source_sha256,
        derived_sha,
        reducer_sha,
        helper_sha,
        observation_sha,
        osim.GetVersionAndDate(),
        binaries,
        4,
        rank,
        condition,
        len(request.comparison_states),
        errors,
    )
