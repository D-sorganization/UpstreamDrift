"""Issue #4819 — every engine adapter round-trips losslessly.

For each adapter registered in :data:`ADAPTER_REGISTRY` (drake,
pinocchio, myosuite, opensim, simscape) we write a representative
:class:`SubjectAnthropometrics` to disk and read it back. Every
inertial scalar / vector / tensor recovers to ``rtol=1e-9, atol=1e-12``
of the original.

Engine-wheel-gated paths use :func:`pytest.importorskip` so the suite
remains green on systems without the optional native physics wheels.
The five adapters in this repo do not require any third-party engine
wheel — Drake/Pinocchio/MyoSuite write their own URDF/MJCF, OpenSim
is XML, Simscape is ``scipy.io.savemat`` — but the importorskip is
plumbed in for forward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anthropometrics import (
    ADAPTER_REGISTRY,
    SegmentProperties,
    SubjectAnthropometrics,
)
from anthropometrics.estimators import DeLevaEstimator

_RTOL: float = 1.0e-9
_ATOL: float = 1.0e-12

# scipy.io.savemat is the only optional dependency on the round-trip
# path. simscape's adapter imports it eagerly, so importing the
# adapter module itself already pulls scipy.io. The probe stays here
# as a defensive guard for environments where scipy might be missing.
_OPTIONAL_BACKENDS: dict[str, str] = {
    "simscape": "scipy.io",
}


# Default file extension per adapter. Lines up with
# ``pipeline._ENGINE_EXTENSIONS`` but is duplicated here so this
# integration test is independent of the pipeline module.
_DEFAULT_EXT: dict[str, str] = {
    "drake": "urdf",
    "pinocchio": "urdf",
    "myosuite": "xml",
    "opensim": "osim",
    "simscape": "mat",
}


@pytest.fixture(scope="module")
def reference_subject() -> SubjectAnthropometrics:
    """Realistic 16+ segment subject used by every adapter in this suite.

    Built from the de Leva estimator at the canonical (1.78 m, 75 kg)
    male reference. The resulting record carries 21 named segments —
    well above the 16 required by the issue acceptance criterion —
    covering head, neck, thorax, lumbar, pelvis, plus left/right
    shoulder, upper arm, forearm, hand, hip, thigh, shin, foot.
    """
    return DeLevaEstimator().estimate(
        subject_id="round_trip_subject",
        height_m=1.78,
        mass_kg=75.0,
        sex="M",
        age_years=32.0,
    )


def _assert_segment_close(
    expected: SegmentProperties, actual: SegmentProperties
) -> None:
    """Assert two :class:`SegmentProperties` agree on every numeric field."""
    assert actual.name == expected.name
    assert actual.body_part_id == expected.body_part_id
    assert actual.proximal_marker == expected.proximal_marker
    assert actual.distal_marker == expected.distal_marker
    assert actual.source_method == expected.source_method
    assert actual.length_m == pytest.approx(expected.length_m, rel=_RTOL, abs=_ATOL)
    assert actual.mass_kg == pytest.approx(expected.mass_kg, rel=_RTOL, abs=_ATOL)
    assert actual.source_subject_height_m == pytest.approx(
        expected.source_subject_height_m, rel=_RTOL, abs=_ATOL
    )
    assert actual.source_subject_mass_kg == pytest.approx(
        expected.source_subject_mass_kg, rel=_RTOL, abs=_ATOL
    )
    np.testing.assert_allclose(
        actual.com_xyz_m, expected.com_xyz_m, rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        actual.inertia_tensor, expected.inertia_tensor, rtol=_RTOL, atol=_ATOL
    )


def _assert_record_close(
    expected: SubjectAnthropometrics, actual: SubjectAnthropometrics
) -> None:
    """Assert two :class:`SubjectAnthropometrics` records agree fully."""
    assert actual.subject_id == expected.subject_id
    assert actual.source_method == expected.source_method
    assert actual.sex == expected.sex
    if expected.age_years is None:
        assert actual.age_years is None
    else:
        assert actual.age_years == pytest.approx(
            expected.age_years, rel=_RTOL, abs=_ATOL
        )
    assert actual.height_m == pytest.approx(expected.height_m, rel=_RTOL, abs=_ATOL)
    assert actual.mass_kg == pytest.approx(expected.mass_kg, rel=_RTOL, abs=_ATOL)
    assert len(actual.segments) == len(expected.segments)
    for (an_name, an_props), (ex_name, ex_props) in zip(
        actual.segments, expected.segments, strict=True
    ):
        assert an_name == ex_name
        _assert_segment_close(ex_props, an_props)


@pytest.mark.parametrize("engine_name", sorted(ADAPTER_REGISTRY.keys()))
def test_adapter_round_trip(
    engine_name: str,
    reference_subject: SubjectAnthropometrics,
    tmp_path: Path,
) -> None:
    """``adapter.export → adapter.import_back`` recovers the canonical record."""
    optional = _OPTIONAL_BACKENDS.get(engine_name)
    if optional is not None:
        pytest.importorskip(optional, reason=f"{engine_name} requires {optional}")

    adapter = ADAPTER_REGISTRY[engine_name]
    ext = _DEFAULT_EXT[engine_name]
    target = tmp_path / f"{engine_name}.{ext}"

    adapter.export(reference_subject, target)
    # MyoSuite writes both .urdf and .xml siblings off the supplied
    # stem; every other adapter writes the file at the supplied path.
    if engine_name == "myosuite":
        assert target.with_suffix(".urdf").exists()
        assert target.with_suffix(".xml").exists()
    else:
        assert target.exists()

    recovered = adapter.import_back(target)
    _assert_record_close(reference_subject, recovered)


def test_myosuite_imports_back_from_urdf_sibling(
    reference_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """The MyoSuite adapter accepts either the ``.urdf`` or ``.xml`` sibling."""
    adapter = ADAPTER_REGISTRY["myosuite"]
    target = tmp_path / "myosuite.xml"
    adapter.export(reference_subject, target)
    recovered_from_urdf = adapter.import_back(target.with_suffix(".urdf"))
    _assert_record_close(reference_subject, recovered_from_urdf)


def test_unknown_extension_raises_for_myosuite(
    reference_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """MyoSuite ``import_back`` rejects unknown extensions with ``ValueError``."""
    adapter = ADAPTER_REGISTRY["myosuite"]
    bad = tmp_path / "myosuite.bogus"
    bad.write_text("not a real file", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.urdf|\.xml|\.mjcf"):
        adapter.import_back(bad)


def test_every_registry_entry_satisfies_engine_adapter_protocol() -> None:
    """Every adapter in the registry conforms to :class:`EngineAdapter`."""
    from anthropometrics import EngineAdapter

    for name, adapter in ADAPTER_REGISTRY.items():
        assert isinstance(adapter, EngineAdapter), f"{name} is not an EngineAdapter"
        assert getattr(adapter, "engine_name", "") == name
