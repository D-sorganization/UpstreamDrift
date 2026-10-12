"""Native MPP optimizer intervals exclude conditional-point activity ranges."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
    NativePathWorkIdentity,
)

pytestmark = pytest.mark.unit


def _identity(
    chart: tuple[float, float],
    active: tuple[float, float],
    function: tuple[float, float],
) -> NativePathWorkIdentity:
    path = "/forceset/muscle/geometrypath"
    point = path + "/pathpointset/guide"
    coordinate = "/jointset/knee/knee_angle"
    return NativePathWorkIdentity(
        source_sha256="a" * 64,
        path_component=path,
        coordinate_path=coordinate,
        point_paths=(point,),
        source_component_paths=(path, point, coordinate),
        chart_bounds=chart,
        active_bounds=active,
        function_bounds=function,
    )


def test_optimizer_bounds_intersect_chart_and_spline_not_cpp_activity() -> None:
    identity = _identity((-0.8, 0.8), (-0.1, 0.1), (-0.4, 0.3))

    assert identity.optimizer_bounds == (-0.4, 0.3)


def test_empty_optimizer_domain_is_rejected_even_when_cpp_range_overlaps() -> None:
    identity = _identity((-0.8, 0.1), (-0.1, 0.1), (0.2, 0.3))

    with pytest.raises(ValueError, match="optimizer.*domain"):
        _ = identity.optimizer_bounds


class _Coordinate:
    def __init__(self, path="/jointset/knee/knee_angle", bounds=(-1.0, 1.0)):
        self.path = path
        self.bounds = bounds

    def getAbsolutePathString(self):
        return self.path

    def getRangeMin(self):
        return self.bounds[0]

    def getRangeMax(self):
        return self.bounds[1]


class _Spline:
    def __init__(self, knots=(-0.4, 0.3)):
        self.knots = knots

    def getNumberOfPoints(self):
        return len(self.knots)

    def getX(self, index):
        return self.knots[index]


class _UnknownFunction:
    pass


class _MovingPoint:
    def __init__(
        self,
        function,
        *,
        path="/forceset/muscle/geometrypath/pathpointset/guide",
        coordinates=None,
        functions=None,
    ):
        self.path = path
        self.coordinates = coordinates or (_Coordinate(),) * 3
        self.functions = functions or (function,) * 3

    def getAbsolutePathString(self):
        return self.path

    def getXCoordinate(self):
        return self.coordinates[0]

    def getYCoordinate(self):
        return self.coordinates[1]

    def getZCoordinate(self):
        return self.coordinates[2]

    def get_x_location(self):
        return self.functions[0]

    def get_y_location(self):
        return self.functions[1]

    def get_z_location(self):
        return self.functions[2]

    def getConcreteClassName(self):
        return "MovingPathPoint"


class _ConditionalPoint:
    def getCoordinate(self):
        return _Coordinate()

    def get_range(self, index):
        return (-0.1, 0.1)[index]

    def getConcreteClassName(self):
        return "ConditionalPathPoint"


class _PointSet:
    def __init__(self, points):
        self.points = points

    def getSize(self):
        return len(self.points)

    def get(self, index):
        return self.points[index]


class _Geometry:
    def __init__(self, point_set):
        self.point_set = point_set

    def getAbsolutePathString(self):
        return "/forceset/muscle/geometrypath"

    def getPathPointSet(self):
        return self.point_set

    def getConcreteClassName(self):
        return "GeometryPath"


class _Downcast:
    accepted = ()

    @classmethod
    def safeDownCast(cls, value):
        return value if isinstance(value, cls.accepted) else None


class _OpenSim:
    class GeometryPath(_Downcast):
        accepted = (_Geometry,)

    class MovingPathPoint(_Downcast):
        accepted = (_MovingPoint,)

    class ConditionalPathPoint(_Downcast):
        accepted = (_ConditionalPoint,)

    class SimmSpline(_Downcast):
        accepted = (_Spline,)

    class LinearFunction(_Downcast):
        accepted = ()

    class Constant(_Downcast):
        accepted = ()


class _Model:
    def __init__(self, components):
        self.components = components
        self.finalize_calls = 0

    def finalizeConnections(self):
        self.finalize_calls += 1

    def getComponentsList(self):
        return self.components


def _source_model(
    tmp_path, function=None, *, declaration_present=True, coordinate=None
):
    from hashlib import sha256

    from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
        native_path_optimizer_domains,
    )

    model_path = tmp_path / "source.osim"
    payload = b"source-bound synthetic model"
    model_path.write_bytes(payload)
    coordinate = coordinate or _Coordinate()
    point_set = _PointSet(
        [
            _ConditionalPoint(),
            _MovingPoint(
                _Spline() if function is None else function,
                coordinates=(coordinate, coordinate, coordinate),
            ),
        ]
    )
    model = _Model([_Geometry(point_set), *point_set.points])
    declaration = SimpleNamespace(
        model_path=model_path,
        source_sha256=sha256(payload).hexdigest(),
        chart_bounds={"/jointset/knee/knee_angle": (-0.8, 0.8)},
    )
    return native_path_optimizer_domains(
        model, declaration if declaration_present else None, _OpenSim()
    )


def test_source_inventory_intersects_every_axis_and_excludes_cpp_activity(tmp_path):
    assert _source_model(tmp_path) == {"/jointset/knee/knee_angle": (-0.4, 0.3)}


def test_source_inventory_fails_closed_for_unreviewed_mpp_function(tmp_path):
    with pytest.raises(ValueError, match="no reviewed coordinate domain"):
        _source_model(tmp_path, _UnknownFunction())


def test_source_inventory_rejects_missing_declared_coordinate_chart(tmp_path):
    from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
        native_path_optimizer_domains,
    )

    model_path = tmp_path / "source.osim"
    model_path.write_bytes(b"source")
    points = _PointSet([_ConditionalPoint(), _MovingPoint(_Spline())])
    model = _Model([_Geometry(points), *points.points])
    declaration = SimpleNamespace(
        model_path=model_path,
        source_sha256=__import__("hashlib").sha256(b"source").hexdigest(),
        chart_bounds={},
    )
    with pytest.raises(ValueError, match="absent from the declared chart"):
        native_path_optimizer_domains(model, declaration, _OpenSim())


def test_preparation_blocks_probe_when_optimizer_exceeds_source_domain(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext
    from hashlib import sha256
    import sys

    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_preparation as preparation,
    )

    payload = b"source-bound synthetic model"
    model_path = tmp_path / "source.osim"
    model_path.write_bytes(payload)
    point_set = _PointSet([_ConditionalPoint(), _MovingPoint(_Spline())])
    model = _Model([_Geometry(point_set), *point_set.points])
    declaration = SimpleNamespace(
        model_path=model_path,
        source_sha256=sha256(payload).hexdigest(),
        chart_bounds={"/jointset/knee/knee_angle": (-0.8, 0.8)},
    )
    request = SimpleNamespace(
        model_path=model_path,
        model_sha256=sha256(payload).hexdigest(),
        constrained_cold_start=declaration,
        bindings=SimpleNamespace(
            state_bounds={"/jointset/knee/knee_angle/value": (-0.5, 0.4)}
        ),
        passive_policy=None,
        mixed_actuation=None,
    )
    monkeypatch.setitem(
        sys.modules, "opensim", SimpleNamespace(Model=lambda _path: model)
    )
    domain_results = []

    def record_domains(*_args):
        result = {"/jointset/knee/knee_angle": (-0.4, 0.3)}
        domain_results.append(result)
        return result

    monkeypatch.setattr(preparation, "native_path_optimizer_domains", record_domains)
    monkeypatch.setattr(preparation, "_declared_chart_blockers", lambda _request: [])
    monkeypatch.setattr(preparation, "_native_runtime_identity", lambda _osim: "b" * 64)
    monkeypatch.setattr(
        preparation, "_audit_loaded_model_components", lambda *_args: []
    )
    monkeypatch.setattr(
        preparation,
        "reconstruct_declared_cold_start",
        lambda _declared: nullcontext(SimpleNamespace(model=model, state=object())),
    )
    monkeypatch.setattr(
        preparation, "_audit_native_state_bindings", lambda *_args: "c" * 64
    )
    monkeypatch.setattr(preparation, "_audit_native_passive", lambda *_args: (None, []))
    monkeypatch.setattr(
        preparation, "_audit_reference_convention", lambda *_args: ("d" * 64, [])
    )
    probe_calls = []

    def probe_gate(
        _model, _state, _request, _runtime_sha, _blockers, run_probe, _callable
    ):
        probe_calls.append(run_probe)
        return None, False, None, None, None

    monkeypatch.setattr(preparation, "_probe_native_state_if_eligible", probe_gate)
    result = preparation._native_preparation(request, run_probe=True)

    assert "native-moco-path-optimizer-domain-invalid" in result[3]
    assert model.finalize_calls == 1
    assert domain_results == [{"/jointset/knee/knee_angle": (-0.4, 0.3)}]
    assert probe_calls == [False]
    assert result[6] is False


def test_optimizer_domain_diagnostic_is_a_hard_execution_block():
    from src.engines.physics_engines.opensim.python.tour_matching.native_diagnostic_contract import (
        DiagnosticCode,
        ExecutionEffect,
        QualificationEffect,
        diagnostic_effects,
    )

    assert diagnostic_effects(DiagnosticCode.NATIVE_PATH_OPTIMIZER_DOMAIN_INVALID) == (
        ExecutionEffect.BLOCK,
        QualificationEffect.NONE,
    )


@pytest.mark.parametrize(
    "bounds",
    [(float("nan"), 1.0), (-1.0, float("inf")), (0.5, -0.5)],
)
def test_native_coordinate_range_must_be_finite_and_ordered(tmp_path, bounds):
    with pytest.raises(ValueError, match="native coordinate range"):
        _source_model(
            tmp_path,
            declaration_present=False,
            coordinate=_Coordinate(bounds=bounds),
        )


def test_native_domain_is_available_without_optional_chart_declaration(tmp_path):
    assert _source_model(tmp_path, declaration_present=False) == {
        "/jointset/knee/knee_angle": (-0.4, 0.3)
    }


def test_all_mpp_functions_intersect_for_shared_coordinate(tmp_path):
    from hashlib import sha256
    from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
        native_path_optimizer_domains,
    )

    model_path = tmp_path / "shared.osim"
    payload = b"shared source"
    model_path.write_bytes(payload)
    coordinate = _Coordinate()
    first = _MovingPoint(
        _Spline((-0.8, 0.4)),
        path="/forceset/first/geometrypath/pathpointset/guide",
        coordinates=(coordinate,) * 3,
    )
    second = _MovingPoint(
        _Spline((-0.2, 0.1)),
        path="/forceset/second/geometrypath/pathpointset/guide",
        coordinates=(coordinate,) * 3,
    )
    first_set = _PointSet([first])
    second_set = _PointSet([second])
    model = _Model(
        [
            _Geometry(first_set),
            _Geometry(second_set),
            first,
            second,
        ]
    )
    domains = native_path_optimizer_domains(model, None, _OpenSim())

    assert domains == {"/jointset/knee/knee_angle": (-0.2, 0.1)}


def test_each_mpp_axis_uses_its_own_coordinate_function_domain(tmp_path):
    from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
        native_path_optimizer_domains,
    )

    qx = _Coordinate("/jointset/hip/rotation_x", (-1.0, 1.0))
    qy = _Coordinate("/jointset/hip/rotation_y", (-2.0, 2.0))
    qz = _Coordinate("/jointset/hip/translation_z", (-3.0, 3.0))
    point = _MovingPoint(
        _Spline(),
        coordinates=(qx, qy, qz),
        functions=(_Spline((-0.5, 0.5)), _Spline((-1.5, 1.5)), _Spline((-2.5, 2.5))),
    )
    point_set = _PointSet([point])
    model = _Model([_Geometry(point_set), point])

    domains = native_path_optimizer_domains(model, None, _OpenSim())

    assert domains == {
        "/jointset/hip/rotation_x": (-0.5, 0.5),
        "/jointset/hip/rotation_y": (-1.5, 1.5),
        "/jointset/hip/translation_z": (-2.5, 2.5),
    }


@pytest.mark.parametrize(
    "bounds",
    [None, (float("nan"), 0.2), (-0.2, float("inf")), (0.2, -0.2), (0.1,)],
)
def test_preparation_domain_guard_blocks_invalid_optimizer_boxes(monkeypatch, bounds):
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_preparation as preparation,
    )

    coordinate_path = "/jointset/knee/knee_angle"
    request = SimpleNamespace(
        constrained_cold_start=object(),
        bindings=SimpleNamespace(
            state_bounds={} if bounds is None else {coordinate_path + "/value": bounds}
        ),
    )
    monkeypatch.setattr(
        preparation,
        "native_path_optimizer_domains",
        lambda *_args: {coordinate_path: (-0.4, 0.3)},
    )

    assert preparation._native_path_optimizer_blockers(
        model=object(), request=request, osim=object()
    ) == ["native-moco-path-optimizer-domain-invalid"]


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((0.0, 0.0), []),
        ((0.4, 0.4), ["native-moco-path-optimizer-domain-invalid"]),
    ],
)
def test_fixed_optimizer_box_is_allowed_only_inside_source_domain(
    monkeypatch, bounds, expected
):
    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_preparation as preparation,
    )

    coordinate_path = "/jointset/knee/knee_angle"
    request = SimpleNamespace(
        constrained_cold_start=object(),
        bindings=SimpleNamespace(state_bounds={coordinate_path + "/value": bounds}),
    )
    monkeypatch.setattr(
        preparation,
        "native_path_optimizer_domains",
        lambda *_args: {coordinate_path: (-0.4, 0.3)},
    )

    assert (
        preparation._native_path_optimizer_blockers(
            model=object(), request=request, osim=object()
        )
        == expected
    )


def _run_preparation_with_lifecycle_spies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_finalize: bool,
    use_declared_cold_start: bool = True,
) -> tuple[tuple[object, ...], list[str], object]:
    from contextlib import nullcontext
    from hashlib import sha256
    import sys

    from src.engines.physics_engines.opensim.python.tour_matching import (
        native_moco_preparation as preparation,
    )

    events = []
    model_path = tmp_path / "lifecycle.osim"
    payload = b"lifecycle model"
    model_path.write_bytes(payload)

    class Model:
        def finalizeConnections(self):
            events.append("finalize")
            if fail_finalize:
                raise RuntimeError("connection finalization failed")

        def initSystem(self):
            events.append("initSystem")
            raise AssertionError("finalization must not initialize the model")

    model = Model()
    monkeypatch.setitem(
        sys.modules, "opensim", SimpleNamespace(Model=lambda _path: model)
    )
    monkeypatch.setattr(preparation, "_declared_chart_blockers", lambda _request: [])
    monkeypatch.setattr(
        preparation,
        "_native_runtime_identity",
        lambda _osim: events.append("runtime") or "b" * 64,
    )
    monkeypatch.setattr(
        preparation,
        "_native_path_optimizer_blockers",
        lambda *_args: events.append("domain") or [],
    )
    monkeypatch.setattr(
        preparation,
        "_audit_loaded_model_components",
        lambda *_args: events.append("components") or [],
    )
    monkeypatch.setattr(
        preparation,
        "reconstruct_declared_cold_start",
        lambda _declared: (
            events.append("reconstruct")
            or nullcontext(SimpleNamespace(model=model, state=object()))
        ),
    )
    monkeypatch.setattr(
        preparation,
        "_audit_native_state_bindings",
        lambda *_args: events.append("state") or "c" * 64,
    )
    monkeypatch.setattr(
        preparation,
        "_audit_native_passive",
        lambda *_args: events.append("passive") or (None, []),
    )
    monkeypatch.setattr(
        preparation,
        "_audit_reference_convention",
        lambda *_args: events.append("reference") or ("d" * 64, []),
    )

    def probe(*_args):
        events.append("probe")
        return "e" * 64, True, 0.0, 0.1, None

    monkeypatch.setattr(preparation, "_probe_native_state_if_eligible", probe)
    from src.engines.physics_engines.opensim.python.tour_matching import (
        muscle_replay,
    )

    monkeypatch.setattr(
        muscle_replay,
        "_restore_continuous_state",
        lambda *_args: events.append("restore") or (object(), None, None),
    )
    request = SimpleNamespace(
        model_path=model_path,
        model_sha256=sha256(payload).hexdigest(),
        constrained_cold_start=object() if use_declared_cold_start else None,
        bindings=SimpleNamespace(state_bounds={}, initial_state={}),
        passive_policy=None,
        mixed_actuation=None,
        config=SimpleNamespace(t_start_s=0.0),
    )
    return preparation._native_preparation(request, run_probe=True), events, model


def test_native_preparation_finalizes_before_all_native_audits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, events, model = _run_preparation_with_lifecycle_spies(
        monkeypatch, tmp_path, fail_finalize=False
    )

    assert result[3] == []
    assert events == [
        "finalize",
        "domain",
        "runtime",
        "components",
        "reconstruct",
        "state",
        "passive",
        "reference",
        "probe",
    ]


@pytest.mark.parametrize("use_declared_cold_start", [True, False])
def test_native_preparation_finalization_failure_stops_all_later_native_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_declared_cold_start: bool,
) -> None:
    result, events, _model = _run_preparation_with_lifecycle_spies(
        monkeypatch,
        tmp_path,
        fail_finalize=True,
        use_declared_cold_start=use_declared_cold_start,
    )

    assert result == (
        None,
        None,
        None,
        ["native-moco-path-optimizer-domain-invalid"],
        None,
        None,
        False,
        None,
        None,
        None,
    )
    assert events == ["finalize"]
