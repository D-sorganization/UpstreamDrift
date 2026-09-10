"""Read-only evidence inspection for guided capture; safe to run off the UI thread."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from src.motion_capture.coaching.storage import load_layer
from src.motion_capture.reference import (
    load_comparison_session,
    session_camera,
    session_clock,
)
from src.motion_capture.reference.storage import ReferenceLibrary
from src.motion_capture.rig.bundle import PLAN_FILE, load_bundle
from src.motion_capture.rig.capture_notes import read_notes
from src.motion_capture.rig.edits import EDITS_FILE, load_edits, validate_edits
from src.motion_capture.rig.equipment import load_capture_club

from .capture_evidence import inspect_observations
from .calibration_profiles import validate_profile_set
from .goal_catalog import workflow_evidence
from .goal_planner import CaptureRoute, Readiness, evaluate
from .session import SessionMedia
from .wizard_storage import input_revision, read_document


@dataclass(frozen=True)
class CalibrationReview:
    """A current-window confirmation, discarded on restart or changed lens evidence."""

    capture_id: str
    path: Path
    calibration_sha256: str
    plan_sha256: str

    @classmethod
    def confirmed(cls, root: Path, path: Path) -> CalibrationReview:
        # Called only after CalibrationDialog validated all ProfileAssignments.
        data = read_document(path)
        validate_profile_set(data, _camera_set(root), capture_root=root)
        return cls(
            read_notes(root).capture_id,
            path.resolve(),
            sha256(data).hexdigest(),
            sha256(read_document(root / PLAN_FILE)).hexdigest(),
        )

    def matches(self, media: SessionMedia, start: Path | None) -> bool:
        matches = (
            start is not None
            and start.resolve() == self.path
            and self.capture_id == read_notes(media.root).capture_id
            and self.calibration_sha256 == sha256(read_document(self.path)).hexdigest()
            and self.plan_sha256
            == sha256(read_document(media.root / PLAN_FILE)).hexdigest()
        )
        if matches:
            validate_profile_set(
                read_document(self.path),
                _camera_set(media.root),
                capture_root=media.root,
            )
        return matches


def _camera_set(root: Path) -> dict[str, tuple[str, tuple[int, int]]]:
    plan, index, _ = load_bundle(root)
    identities = {camera.view: camera.identity for camera in plan.cameras}
    if any(not entry.width or not entry.height for entry in index.recordings):
        raise ValueError("Recorded camera sizes are unknown; review the capture")
    return {
        entry.view: (
            identities[entry.view],
            (int(entry.width or 0), int(entry.height or 0)),
        )
        for entry in index.recordings
    }


@dataclass(frozen=True)
class WizardEvidence:
    capture_id: str | None
    identity: str
    input_revision: str | None
    states: dict[str, Readiness]


def _drawings(media: SessionMedia) -> bool:
    _, index, _ = load_bundle(media.root)
    return any(
        bool(
            load_layer(
                media.root, entry.view, entry.width, entry.height, entry.frames
            ).shapes
        )
        for entry in index.recordings
        if entry.ok and entry.width and entry.height and entry.frames
    )


def _references(media: SessionMedia, library: ReferenceLibrary) -> dict[str, Readiness]:
    scan = library.scan()
    assets = {asset.id: asset for asset in scan.assets}
    evidence = {
        "capture.experts": Readiness(
            "done" if assets else "ready",
            "Choose a reference in Expert Comparison."
            if assets
            else "Import an expert video, C3D or model motion.",
        ),
        "compare.video": Readiness(
            "ready", "Align an expert video and save the comparison."
        ),
        "compare.projected": Readiness(
            "ready", "Place a model in the camera scene, align events and save."
        ),
    }
    for path in sorted((media.root / "comparisons").glob("*.json")):
        if ".before-review-" in path.name:
            continue
        saved = load_comparison_session(path)
        key = (
            "compare.video" if saved.reference_kind == "video" else "compare.projected"
        )
        try:
            asset = assets.get(saved.reference_id)
            registration = saved.registration
            if (
                asset is None
                or registration is None
                or Path(saved.session_root).resolve() != media.root.resolve()
            ):
                raise ValueError("Reference or capture association needs review")
            try:
                camera = session_camera(media.root, "", saved.view)
            except (ValueError, OSError, KeyError):
                if saved.reference_kind == "motion":
                    raise
                camera = None
            registration.validate_binding(
                asset, camera, session_clock(media.timing, saved.view)
            )
            evidence[key] = Readiness(
                "done",
                "Saved alignment is associated with this reference and camera; review its quality.",
            )
        except (ValueError, OSError, KeyError) as exc:
            if evidence[key].status != "done":
                evidence[key] = Readiness(
                    "ready", f"Review the saved comparison: {exc}"
                )
    return evidence


def inspect_capture(
    route: CaptureRoute,
    media: SessionMedia | None,
    library_root: Path,
    *,
    review: CalibrationReview | None = None,
    start_file: Path | None = None,
    skipped: frozenset[str] = frozenset(),
    model_name: str = "golfer",
) -> WizardEvidence:
    """Inspect real artifacts; stale prerequisites block downstream completion."""
    if media is None:
        return WizardEvidence(
            None,
            "No Capture Selected",
            None,
            {
                step.id: Readiness("ready", "Open Library to import or select a swing.")
                if step.id == "capture.library"
                else Readiness("blocked", "Select a capture first.")
                for step in route.steps
            },
        )
    notes = read_notes(media.root)
    revision = input_revision(media)
    playable = sum(
        view.recording is not None and view.recording.is_file() for view in media.views
    )
    navigation = {
        "capture.library": Readiness(
            "done" if playable else "ready",
            "Recordings selected." if playable else "Import or record playable video.",
        )
    }
    if "capture.selection" in route.step_ids:
        validate_edits(media.root, load_edits(media.root))
        saved = (media.root / EDITS_FILE).is_file()
        navigation["capture.selection"] = Readiness(
            "done" if saved else "ready",
            "Saved selection uses original video coordinates."
            if saved
            else "Mark the swing and Save. Keep the full range to use the entire recording.",
        )
    if "capture.references" in route.step_ids:
        navigation["capture.references"] = Readiness(
            "done" if _drawings(media) else "ready",
            "Save lines or shapes in the drawing editor.",
        )
    if "capture.experts" in route.step_ids:
        navigation.update(
            _references(media, ReferenceLibrary(library_root / "references"))
        )
    if "clubs.bag" in route.step_ids:
        club = load_capture_club(media.root)
        navigation["clubs.bag"] = Readiness(
            "done" if club else "ready",
            "Club assigned to this capture."
            if club
            else "Assign a club, or skip if its details are unknown.",
        )
    for step in route.steps:
        if step.optional and step.id in skipped:
            navigation[step.id] = Readiness(
                "skipped", "You skipped this optional step."
            )
    compatible = bool(review and review.matches(media, start_file))
    invalid = _invalidated(media, route, compatible, model_name)
    current = replace(media, intrinsics=start_file) if compatible else media
    evidence = workflow_evidence(route, current, navigation, invalidated=invalid)
    states = evaluate(
        route, evidence, view_count=playable, calibration_compatible=compatible
    )
    if input_revision(media) != revision:
        raise ValueError("Capture changed while checking status; refresh again")
    return WizardEvidence(
        notes.capture_id,
        f"{notes.title}\nCapture: {notes.capture_id}",
        revision,
        dict(zip(route.step_ids, states, strict=True)),
    )


def _invalidated(
    media: SessionMedia, route: CaptureRoute, compatible: bool, model: str
) -> dict[str, str]:
    invalid: dict[str, str] = {}
    if "step.intrinsics" in route.step_ids and not compatible:
        invalid["intrinsics"] = (
            "Review the lens, optical zoom, focus and image size for this capture. Reconfirm after reopening the wizard."
        )
    current_views = 0
    for view in media.views:
        if view.observations is not None:
            evidence = inspect_observations(media.root, view.observations)
            if evidence.edits_match is True:
                current_views += 1
            else:
                invalid["detect"] = (
                    f"{view.view}: {evidence.description}. Run pose detection again or review its metadata."
                )
    if "step.reconstruct" in route.step_ids and current_views < 2:
        invalid["reconstruct"] = (
            "Detect joints in at least two camera views with matching swing edits."
        )
    if media.model_fit is not None:
        provenance = media.model_fit.get("provenance", {})
        parameters = (
            provenance.get("parameters", {}) if isinstance(provenance, dict) else {}
        )
        if not isinstance(parameters, dict) or parameters.get("model") != model:
            invalid["fit_model"] = (
                "The saved fit uses another or unverified model. Select its model or fit the currently selected model."
            )
    return invalid
