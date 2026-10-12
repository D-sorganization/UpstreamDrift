"""Real native camera/geometry admission and separately fresh shaft diagnostics."""

from copy import deepcopy
import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _geometric_request(case, tmp_path):
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.workspace import NativeHypothesisRequest

    library, request, parent = case
    record = request.to_record()
    definition = deepcopy(parent["provenance"]["native_definition"])
    definition["joints"][1]["parent_to_base"][2][3] += 0.025
    xml, _ = export_full_body_mjcf(json.dumps(definition).encode())
    path = tmp_path / "geometric.xml"
    path.write_text(xml, encoding="utf-8")
    asset = library.add_model(
        "geometric",
        "practice",
        path,
        engine="mujoco",
        dofs=tuple(parent["coordinate_order"]),
    )
    record["model"].update(
        model_id=asset.dataset_id,
        model_hash=asset.metadata["hash"],
        definition=definition,
    )
    record["camera"] = parent["evidence"]["original_fit"]["camera"]
    return NativeHypothesisRequest.from_record(record)


def test_camera_only_admission_preserves_model_identity(hypothesis_case):
    from src.shared.python.workspace import (
        NativeHypothesisRequest,
        author_native_hypothesis,
    )

    library, request, parent = hypothesis_case
    record = request.to_record()
    record["model"].update(
        model_id=parent["model_id"],
        model_hash=parent["model_hash"],
        definition=parent["provenance"]["native_definition"],
    )
    saved = author_native_hypothesis(
        library, "parent", "camera-only", NativeHypothesisRequest.from_record(record)
    )
    fit = library.load_fit(saved.dataset_id)
    assert fit["model_hash"] == parent["model_hash"]
    assert fit["q"] == parent["q"]
    assert (
        fit["evidence"]["original_fit"]["camera"]
        != parent["evidence"]["original_fit"]["camera"]
    )


def test_geometric_dimension_changes_real_fk_not_coordinate_identity(
    hypothesis_case, tmp_path
):
    from src.shared.python.workspace import author_native_hypothesis
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    library, _, parent = hypothesis_case
    request = _geometric_request(hypothesis_case, tmp_path)
    saved = author_native_hypothesis(library, "parent", "geometry-seed", request)
    before = load_native_fit_binding(library, "parent")
    after = load_native_fit_binding(library, saved.dataset_id)
    body = parent["provenance"]["native_definition"]["joints"][1]["child"]
    pose = np.asarray(parent["q"][0])
    old = before.plant.marker_positions(pose, {"torso": (body, (0, 0, 0))})
    new = after.plant.marker_positions(pose, {"torso": (body, (0, 0, 0))})
    assert np.linalg.norm(new - old) == pytest.approx(0.025, abs=1e-12)
    assert before.model_hash != after.model_hash
    assert before.coordinate_units == after.coordinate_units
    assert tuple(before.plant.coordinate_order) == tuple(after.plant.coordinate_order)


def test_new_geometry_seed_shaft_is_fresh_bound_and_projected(
    hypothesis_case, tmp_path
):
    from src.shared.python.workspace import author_native_hypothesis
    from src.shared.python.workspace.necromatcher_capture_identity import (
        capture_identity,
    )
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        load_shaft_image_residuals,
    )
    from src.shared.python.motion_matching.historical_fit.shaft_observations import (
        ShaftAxisEvidence,
        ShaftAxisSegment,
        SourceBoundShaftFrame,
    )

    library, _, _ = hypothesis_case
    request = _geometric_request(hypothesis_case, tmp_path)
    saved = author_native_hypothesis(library, "parent", "shaft-seed", request)
    identity = capture_identity(library, request.parents.capture_id)
    # Synthetic fixture annotation, never a historical-player observation.
    segment = ShaftAxisSegment(
        "observed", ((5, 5), (15, 15)), "Fixture", "Synthetic", 0.5, None, 2
    )
    evidence = ShaftAxisEvidence(
        identity.capture_id,
        identity.capture_hash,
        identity.source_sha256,
        (32, 32),
        (
            SourceBoundShaftFrame(
                1, identity.frames[1], identity.png_sha256[1], segment
            ),
        ),
    )
    binding, bundle = load_shaft_image_residuals(
        library, saved.dataset_id, evidence, 0.5
    )
    camera, _ = binding.review_inputs()
    term = bundle.terms[0]
    assessment = term.assess(binding.plant, camera, np.asarray(binding.fit["q"])[[1]])
    assert assessment.evidence_sha256 == evidence.sha256
    assert assessment.raw_rms_pixels is not None
    assert np.isfinite(assessment.raw_rms_pixels)
    assert term.axis.native_model_sha == binding.plant.plant_sha
    assert binding.model_hash == request.model.model_hash
    assert "shaft_axis" not in binding.fit["evidence"]
