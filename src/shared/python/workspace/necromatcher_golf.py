"""Authenticated saved replay impacts for explicit unverified local research shots.

No impact, flight or historical clock calibration is performed by this bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast
from zipfile import ZipFile

from src.shared.python.golf_simulator.contracts import (
    AimContext,
    ContactStatus,
    NumericalStatus,
    ScientificStatus,
    ShotEnvelope,
    ShotMetadata,
    ShotQualification,
    SourceKind,
)
from .artifact_handoff import compute_file_sha256

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary
    from src.shared.python.physics.impact_model.types import PostImpactState
    from src.shared.python.physics.swing_ball_flight_pipeline import SwingState


@dataclass(frozen=True)
class ResearchImpactShot:
    """Detached shot and immutable JSON context, without transferable qualification."""

    shot: ShotEnvelope
    replay_id: str
    run_id: str
    result_sha256: str
    receipt_sha256: str
    trajectory_sha256: str
    context_json: str

    def __post_init__(self) -> None:
        from src.shared.python.shadow_tracker import check_sha256

        if (
            not isinstance(self.shot, ShotEnvelope)
            or self.shot.source_kind is not SourceKind.MODEL_CONTACT
        ):
            raise ValueError(
                "Research shot requires the canonical model-contact envelope"
            )
        if not isinstance(self.shot.qualification, ShotQualification) or any(
            getattr(self.shot.qualification, name)
            is not getattr(_qualification(), name)
            for name in ("contact", "numerical", "scientific")
        ):
            raise ValueError("Research qualification must remain entirely unverified")
        if (
            not isinstance(self.replay_id, str)
            or not self.replay_id
            or not isinstance(self.run_id, str)
            or not self.run_id
        ):
            raise ValueError("Research identities require strings")
        for digest in (self.result_sha256, self.receipt_sha256, self.trajectory_sha256):
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ValueError("Research artifact hashes require canonical prefix")
            check_sha256(digest[7:], "Research artifact hash")
        if not isinstance(self.context_json, str):
            raise ValueError("Research context requires immutable JSON text")
        context = json.loads(self.context_json)
        _context(context, self.replay_id)
        if (
            not isinstance(context, dict)
            or _canonical(context) != self.context_json
            or context.get("scientific_qualified") is not False
            or context.get("physical_source_time_qualified") is not False
            or context.get("qualification")
            != {
                "contact": "unverified",
                "numerical": "unverified",
                "scientific": "unverified",
            }
            or context.get("recorded_time_s") != self.shot.impact_time_s
        ):
            raise ValueError("Research context requires canonical finite JSON")

    def to_record(self) -> dict[str, Any]:
        """Return a detached API-safe whitelist; never expose local artifact paths."""
        return {
            "replay_id": self.replay_id,
            "run_id": self.run_id,
            "result_sha256": self.result_sha256,
            "receipt_sha256": self.receipt_sha256,
            "trajectory_sha256": self.trajectory_sha256,
            **json.loads(self.context_json),
        }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _context(context: Any, replay_id: str) -> None:
    from src.shared.python.shadow_tracker import check_sha256
    from .project_store import validate_workspace_id
    from .necromatcher_impact_contracts import finite_array, nonempty

    keys = {
        "recorded_time_s",
        "recorded_sample_index",
        "parents",
        "replay_clock_policy",
        "scientific_qualified",
        "physical_source_time_qualified",
        "qualification",
        "assumptions",
        "source_to_target_rotation",
        "framepolicy",
    }
    if not isinstance(context, dict) or set(context) != keys:
        raise ValueError("Research context must contain exactly whitelist fields")
    finite_array(context["recorded_time_s"], (), "Authored recorded time")
    if (
        context["recorded_time_s"] < 0
        or type(context["recorded_sample_index"]) is not int
        or context["recorded_sample_index"] < 0
    ):
        raise ValueError("Authored sample/time must be nonnegative")
    if context["replay_clock_policy"] != "authored_simulation_seconds":
        raise ValueError("Research context requires authored simulation seconds")
    _context_rotation(context)
    parents = context["parents"]
    expected = {
        name + suffix
        for name in ("replay", "profile", "fit", "model", "capture")
        for suffix in ("_id", "_hash")
    }
    if (
        not isinstance(parents, dict)
        or set(parents) != expected
        or parents["replay_id"] != replay_id
    ):
        raise ValueError("Research parent whitelist or replay identity differs")
    for name in ("replay", "profile", "fit", "model", "capture"):
        validate_workspace_id(parents[name + "_id"], "Research parent")
        digest = parents[name + "_hash"]
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError("Research parent hash requires canonical prefix")
        check_sha256(digest[7:], "Research parent hash")
    assumptions = context["assumptions"]
    if not isinstance(assumptions, dict) or set(assumptions) != {
        "geometry",
        "selection",
        "environment",
        "ball_assumptions",
        "impact_params",
        "impact_method",
        "flight_settings",
    }:
        raise ValueError("Research assumptions must contain exactly saved settings")
    nonempty(assumptions["geometry"], "Geometry assumption")
    nonempty(assumptions["selection"], "Selection assumption")
    if _canonical(_settings(assumptions)) != _canonical(
        {k: v for k, v in assumptions.items() if k not in {"geometry", "selection"}}
    ):
        raise ValueError("Research settings contain uncontrolled fields")


def _context_rotation(context: dict[str, Any]) -> None:
    from .necromatcher_impact_contracts import finite_array
    from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID

    if context["framepolicy"] != {
        "source_frame_id": FLIGHT_FRAME_ID,
        "target_frame_id": "target_local_xyz",
        "operation": "proper_rotation_vectors",
    }:
        raise ValueError("Research vector frame policy differs")
    rotation = finite_array(
        context["source_to_target_rotation"], (3, 3), "Source-to-target rotation"
    )
    AimContext(
        cast(
            tuple[
                tuple[float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ],
            tuple(tuple(float(v) for v in row) for row in rotation),
        )
    )


def _qualification() -> ShotQualification:
    return ShotQualification(
        ContactStatus.UNVERIFIED,
        NumericalStatus.UNVERIFIED,
        ScientificStatus.UNVERIFIED,
    )


def _caller(metadata: ShotMetadata) -> None:
    if (
        not isinstance(metadata, ShotMetadata)
        or metadata.source_kind is not SourceKind.MODEL_CONTACT
    ):
        raise ValueError("Research impact requires explicit MODEL_CONTACT metadata")
    if not isinstance(metadata.aim_context, AimContext):
        raise ValueError("Research impact requires explicit AimContext")
    if metadata.qualification is not None and (
        not isinstance(metadata.qualification, ShotQualification)
        or metadata.qualification.contact is not ContactStatus.UNVERIFIED
        or metadata.qualification.numerical is not NumericalStatus.UNVERIFIED
        or metadata.qualification.scientific is not ScientificStatus.UNVERIFIED
    ):
        raise ValueError("Caller cannot promote research qualification")


def _post_impact(record: Any) -> PostImpactState:
    from src.shared.python.physics.impact_model.types import PostImpactState
    from .necromatcher_impact_contracts import finite_array

    shapes = {
        "ball_velocity": (3,),
        "ball_angular_velocity": (3,),
        "clubhead_velocity": (3,),
        "clubhead_angular_velocity": (3,),
        "contact_duration": (),
        "energy_transfer": (),
        "impact_location": (2,),
    }
    if not isinstance(record, dict) or set(record) != set(shapes):
        raise ValueError("Saved post-impact state requires exact canonical fields")
    values = {
        key: finite_array(record[key], shape, key) for key, shape in shapes.items()
    }
    for name in ("contact_duration", "energy_transfer"):
        values[name] = values[name].item()
    return PostImpactState(**values)


def _target_impact(record: Any, aim: AimContext) -> PostImpactState:
    from src.shared.python.spatial_algebra.pose6dof import Transform6DOF
    from .necromatcher_impact_contracts import finite_array

    source = _post_impact(record)
    transform = Transform6DOF(
        rotation=finite_array(aim.source_to_target_rotation, (3, 3), "Aim rotation")
    )
    vectors = {
        name: transform.transform_vector(getattr(source, name))
        for name in (
            "ball_velocity",
            "ball_angular_velocity",
            "clubhead_velocity",
            "clubhead_angular_velocity",
        )
    }
    return replace(source, **vectors)


def _read_bundle(archive: Path) -> tuple[dict[str, Any], SwingState, dict[str, str]]:
    from .necromatcher_impact_receipt import load_replay_impact_receipt

    names = {"request.json", "result.json", "impact-receipt.json", "trajectory.json"}
    digest = compute_file_sha256(archive)
    with (
        ZipFile(archive) as zipped,
        TemporaryDirectory(prefix="research-impact-read-") as directory,
    ):
        if len(zipped.namelist()) != 4 or set(zipped.namelist()) != names:
            raise ValueError("Research impact bundle requires exactly four members")
        root = Path(directory)
        for name in names:
            (root / name).write_bytes(zipped.read(name))
        result = json.loads((root / "result.json").read_bytes())
        request = json.loads((root / "request.json").read_bytes())
        _canonical(result)
        _canonical(request)
        state = load_replay_impact_receipt(
            root / "impact-receipt.json", root / "trajectory.json"
        )
        for name in ("parents", "geometry", "selection", "execution_stamp"):
            if _canonical(result.get(name)) != _canonical(request.get(name)):
                raise ValueError("Saved result and request identity differ")
        for name in ("scientific_qualified", "physical_source_time_qualified"):
            if result.get(name) is not False:
                raise ValueError("Saved research qualification must be false")
        if _canonical(result.get("extraction_metadata")) != _canonical(state.metadata):
            raise ValueError("Saved extraction context differs from its receipt")
        for name in ("geometry", "selection"):
            if _canonical(state.metadata[name]) != _canonical(request[name]):
                raise ValueError("Receipt declarations differ from owned request")
        if any(state.metadata.get(k) != v for k, v in request["parents"].items()):
            raise ValueError("Receipt parents differ from owned request")
        hashes = {name: compute_file_sha256(root / name) for name in names}
    if compute_file_sha256(archive) != digest:
        raise ValueError("Authenticated bundle changed while reading")
    return {"result": result, "request": request}, state, hashes


def _settings(result: dict[str, Any]) -> dict[str, Any]:
    """Retain finite saved settings, not local-adapter defaults or arbitrary paths."""
    from .necromatcher_impact_contracts import finite_array

    fields = {
        "environment": (
            "air_density",
            "altitude",
            "gravity",
            "relative_humidity",
            "sea_level_pressure_pa",
            "temperature",
            "wind_velocity",
        ),
        "ball_assumptions": (
            "cd0",
            "cd1",
            "cd2",
            "cl0",
            "cl1",
            "cl2",
            "diameter",
            "mass",
            "spin_decay_rate",
        ),
        "impact_params": (
            "contact_damping",
            "contact_duration",
            "contact_stiffness",
            "cor",
            "friction_coefficient",
            "gear_effect_factor",
            "gear_effect_h_scale",
            "gear_effect_v_scale",
        ),
        "flight_settings": ("dt_s", "max_time_s"),
    }
    settings = {}
    for name, keys in fields.items():
        value = result.get(name)
        if not isinstance(value, dict) or not set(keys).issubset(value):
            raise ValueError("Saved impact settings are incomplete")
        settings[name] = {key: value[key] for key in keys}
        for key in keys:
            if key == "sea_level_pressure_pa" and value[key] is None:
                continue
            finite_array(value[key], (3,) if key == "wind_velocity" else (), key)
    if result.get("impact_method") != "RIGID_BODY":
        raise ValueError("Unsupported saved impact method")
    return {**settings, "impact_method": result["impact_method"]}


def _convert(
    bundle: dict[str, Any],
    state: SwingState,
    hashes: dict[str, str],
    metadata: ShotMetadata,
) -> ResearchImpactShot:
    from src.shared.python.golf_simulator.launch_bridge import (
        post_impact_state_to_shot_envelope,
    )

    request = bundle["request"]
    result = bundle["result"]
    context = state.metadata
    actual = {
        "model_run_id": request["run_id"],
        "trace_digest": context["replay_hash"],
        "impact_id": request["run_id"],
        "impact_time_s": context["recorded_time_s"],
    }
    for name, value in actual.items():
        if getattr(metadata, name) is not None and getattr(metadata, name) != value:
            raise ValueError("Caller impact lineage differs from authenticated result")
    evidence = tuple(
        hashes[name]
        for name in ("result.json", "impact-receipt.json", "trajectory.json")
    )
    qualification = replace(_qualification(), evidence_refs=evidence)
    updated = replace(metadata, qualification=qualification, **actual)
    shot = post_impact_state_to_shot_envelope(
        _target_impact(result.get("impact_state"), metadata.aim_context), updated
    )
    from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID

    safe = {
        "recorded_time_s": context["recorded_time_s"],
        "recorded_sample_index": context["recorded_sample_index"],
        "parents": request["parents"],
        "source_to_target_rotation": [
            list(row) for row in metadata.aim_context.source_to_target_rotation
        ],
        "framepolicy": {
            "source_frame_id": FLIGHT_FRAME_ID,
            "target_frame_id": "target_local_xyz",
            "operation": "proper_rotation_vectors",
        },
        "replay_clock_policy": context["replay_clock_policy"],
        "scientific_qualified": False,
        "physical_source_time_qualified": False,
        "qualification": {
            "contact": "unverified",
            "numerical": "unverified",
            "scientific": "unverified",
        },
        "assumptions": {
            "geometry": request["geometry"]["assumption_description"],
            "selection": request["selection"]["selection_description"],
            **_settings(result),
        },
    }
    return ResearchImpactShot(
        shot,
        request["replay_id"],
        request["run_id"],
        hashes["result.json"],
        hashes["impact-receipt.json"],
        hashes["trajectory.json"],
        _canonical(safe),
    )


def load_research_impact_shot(
    library: NecromatcherLibrary, replay_id: str, run_id: str, metadata: ShotMetadata
) -> ResearchImpactShot:
    """Authenticate saved bytes/parents and preserve exact ball vectors.

    The caller supplies shot/session/aim/timestamp only; model lineage and authored
    sample time derive from the saved result. All qualification remains unverified.
    No simulation is rerun. Canonical download creates its exclusive checked ZIP.
    """
    from .necromatcher_impact_jobs import NativeImpactSession

    _caller(metadata)
    session = NativeImpactSession(library)
    error = None
    try:
        archive = session.download(replay_id, run_id)
        bundle, state, hashes = _read_bundle(archive)
        if (
            bundle["request"].get("run_id") != run_id
            or bundle["request"].get("replay_id") != replay_id
        ):
            raise ValueError("Downloaded bundle has a foreign replay/run identity")
        value = _convert(bundle, state, hashes, metadata)
        fresh = session.download(replay_id, run_id)
        with ZipFile(fresh) as verified:
            from hashlib import sha256

            if (
                len(verified.namelist()) != 4
                or set(verified.namelist()) != set(hashes)
                or any(
                    "sha256:" + sha256(verified.read(name)).hexdigest() != digest
                    for name, digest in hashes.items()
                )
            ):
                raise ValueError(
                    "Read bundle differs from fresh authenticated download"
                )
        if not session.view(replay_id, run_id)["execution_verified"]:
            raise ValueError("Saved impact changed during research conversion")
        return value
    except Exception as exc:
        error = exc
        raise
    finally:
        try:
            session.close()
        except Exception as cleanup:
            if error is None:
                raise
            error.add_note(f"Research impact session cleanup also failed: {cleanup}")
