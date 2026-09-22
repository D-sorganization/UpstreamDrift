"""Fail-closed bridge to the Tools mocap session contract.

ADR-0041: the camera, capture, timebase and session *contracts* are owned by
Tools ``sidekick.lab.mocap`` and reach UpstreamDrift through the pinned vendor
tree, as ``shared.python.sidekick.lab.mocap`` — the same family path the rig's
preview seam and the capture-rig calibration worker already consume. This
module reports exactly what that pin offers and, when it offers the schema
this adapter was written against, projects a rig capture session onto the one
canonical ``MocapSessionManifest`` through the Tools builders and serializer
(#9422). It never invents a mapping: an absent or different schema is reported
as data and the export is refused.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from src.shared.python.core.contracts import StateError, ensure, require
from src.shared.python.logging_pkg.logging_config import get_logger

from .plan import CaptureMode, RigPlan
from .session import CameraStats, CaptureOutcome, SessionManifest
from .sources import HOST_MONOTONIC

logger = get_logger(__name__)

TOOLS_MOCAP_MODULE = "shared.python.sidekick.lab.mocap"
REQUIRED_SUBMODULES = ("devices", "enums", "geometry", "serialization", "session")
# The Tools schema this adapter maps onto; a pin that ships another version is
# ``incompatible`` until the mapping is re-verified against it.
PINNED_SESSION_SCHEMA_VERSION = "mocap-session/1.0.0"
MOCAP_SESSION_FILE = "mocap_session.json"

RIG_PROVIDER_ID = "upstreamdrift.motion_capture.rig"
RIG_CAPTURE_METHOD_ID = f"{RIG_PROVIDER_ID}.capture"
RIG_TRANSPORT = "usb"
_CLOCK_KINDS = {HOST_MONOTONIC: "host-monotonic"}

SchemaStatus = Literal["ready", "incompatible", "unavailable"]
ExportStatus = Literal["written", "rejected", "unavailable"]


@dataclass(frozen=True)
class SchemaProbe:
    """What the pinned Tools tree offers for mocap sessions."""

    status: SchemaStatus
    reason: str | None
    module: str = TOOLS_MOCAP_MODULE
    version: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "module": self.module,
            "status": self.status,
            "reason": self.reason,
            "version": self.version,
        }


@dataclass(frozen=True)
class RecordingTerms:
    """What the operator recorded about consent and retention (ADR-0041).

    The Tools policy contract decides what these terms permit; this record
    only carries them. ``no_store`` is derived: nothing retained, nothing kept.
    """

    consent_recorded: bool = False
    raw_video_retained: bool = False
    retention_days: int = 0

    @property
    def no_store(self) -> bool:
        return not self.raw_video_retained and self.retention_days == 0


def _spec_present(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def probe_tools_schema(module_name: str = TOOLS_MOCAP_MODULE) -> SchemaProbe:
    """Report whether the Tools mocap schema is importable and the pinned one.

    Never raises for a missing or partial installation; the answer is data.
    """
    if not _spec_present(module_name):
        return SchemaProbe(
            "unavailable",
            f"{module_name} is not in the pinned vendor tree (Tools #4706 / PR #4734)",
            module_name,
        )
    missing = [
        s for s in REQUIRED_SUBMODULES if not _spec_present(f"{module_name}.{s}")
    ]
    if missing:
        return SchemaProbe(
            "incompatible",
            f"{module_name} lacks expected submodules: {', '.join(missing)}",
            module_name,
        )
    module = importlib.import_module(module_name)
    version = getattr(module, "MOCAP_SESSION_SCHEMA_VERSION", None)
    if version != PINNED_SESSION_SCHEMA_VERSION:
        return SchemaProbe(
            "incompatible",
            f"{module_name} ships session schema {version!r}; this adapter maps "
            f"{PINNED_SESSION_SCHEMA_VERSION!r}",
            module_name,
            None if version is None else str(version),
        )
    return SchemaProbe("ready", None, module_name, str(version))


@dataclass(frozen=True)
class CameraRecord:
    """One rig camera as Tools records: identity plus advertised capabilities.

    ``identity`` is a Tools ``CameraIdentity`` and ``capabilities`` a Tools
    ``CameraCapabilities``; ``view`` keeps the rig's name for the camera.
    """

    view: str
    identity: Any
    capabilities: Any


def _ready_tools() -> Any:
    """The pinned Tools mocap module, or :class:`StateError` if it is not ready."""
    probe = probe_tools_schema()
    if probe.status != "ready":
        raise StateError(f"Tools mocap schema {probe.status}: {probe.reason}")
    return importlib.import_module(probe.module)


def _clock_kind(domain: str) -> str:
    kind = _CLOCK_KINDS.get(domain)
    if kind is None:
        raise ValueError(f"clock domain {domain!r} has no Tools ClockKind")
    return kind


def _advertised_mode(stats: CameraStats) -> CaptureMode:
    """The negotiated mode, else the requested one when none was negotiated."""
    return stats.effective_mode or stats.requested_mode


def _unsupported(mocap: Any, reason: str) -> Any:
    return mocap.FeatureSupport(mocap.SupportLevel.UNSUPPORTED, reason)


def _capabilities(mocap: Any, stats: CameraStats) -> Any:
    """Map one rig camera onto a Tools ``CameraCapabilities``.

    UVC rig cameras free-run on the host clock and declare neither a shutter
    kind nor an exposure range in microseconds (UVC exposure units are
    driver-specific), so those are reported as unknown/unsupported/absent
    rather than guessed.
    """
    mode = _advertised_mode(stats)
    clock = _clock_kind(stats.clock_domain)
    return mocap.CameraCapabilities(
        resolutions_px=((mode.width, mode.height),),
        frame_rates_hz=(float(mode.fps),),
        pixel_formats=(mode.fourcc,),
        shutter=mocap.ShutterKind.UNKNOWN,
        hardware_trigger=_unsupported(
            mocap, "UVC rig cameras free-run; no hardware trigger input"
        ),
        device_timestamps=_unsupported(
            mocap, f"frames are stamped on the {clock} host clock, not the device"
        ),
    )


def _camera_records(
    mocap: Any, manifest: SessionManifest, plan: RigPlan
) -> tuple[CameraRecord, ...]:
    views = [stats.view for stats in manifest.cameras]
    if len(set(views)) != len(views):
        raise ValueError(f"manifest camera views must be unique: {views}")
    bindings = {binding.view: binding for binding in plan.cameras}
    records = []
    for stats in manifest.cameras:
        binding = bindings.get(stats.view)
        if binding is None:
            raise ValueError(f"camera view {stats.view!r} is not bound by the plan")
        identity = mocap.CameraIdentity(
            provider_id=RIG_PROVIDER_ID,
            device_id=stats.identity,
            transport=RIG_TRANSPORT,
            serial_number=binding.serial,
        )
        records.append(CameraRecord(stats.view, identity, _capabilities(mocap, stats)))
    return tuple(records)


def map_camera_records(
    manifest: SessionManifest, plan: RigPlan
) -> tuple[CameraRecord, ...]:
    """Tools ``CameraIdentity`` + ``CameraCapabilities`` for every rig camera.

    Preconditions: ``manifest`` is a :class:`SessionManifest` and ``plan`` a
    :class:`RigPlan` (else ``ValueError``); the pinned schema probes ``ready``
    (else :class:`StateError`); camera views are unique, bound by ``plan`` and
    on a clock domain with a Tools ``ClockKind`` (else ``ValueError``).

    Postconditions: exactly one record per manifest camera, in manifest order,
    with each view and device id preserved; the identities are the ones
    :func:`export_session_manifest` writes.
    """
    require(isinstance(manifest, SessionManifest), "manifest must be a SessionManifest")
    require(isinstance(plan, RigPlan), "plan must be a RigPlan")
    records = _camera_records(_ready_tools(), manifest, plan)
    ensure(
        [(r.view, r.identity.device_id) for r in records]
        == [(s.view, s.identity) for s in manifest.cameras],
        "one camera record per manifest camera, ids preserved",
    )
    return records


def _clocks(mocap: Any, manifest: SessionManifest) -> tuple[Any, ...]:
    domains = dict.fromkeys(stats.clock_domain for stats in manifest.cameras)
    clocks = []
    for domain in domains:
        clocks.append(
            mocap.ClockDomain(
                clock_id=domain,
                kind=mocap.ClockKind(_clock_kind(domain)),
                tick_period_seconds=1e-9,
                monotonic=True,
            )
        )
    return tuple(clocks)


def export_session_manifest(
    manifest: SessionManifest,
    plan: RigPlan,
    terms: RecordingTerms = RecordingTerms(),
    *,
    calibration_ids: Sequence[str] = (),
) -> str:
    """Canonical Tools ``mocap-session`` JSON for one rig capture session.

    Preconditions: the pinned schema probes ``ready`` (else :class:`StateError`)
    and every manifest camera is bound by ``plan`` (else ``ValueError``). The
    Tools contract validates everything else and raises ``ValueError`` or
    ``TypeError`` for what it rejects — retained raw video without recorded
    consent, duplicate camera identities, a non-UTC start time.

    Postconditions: the text round-trips through Tools ``load_session_manifest``
    and is byte-stable for equal inputs. The session is ``finalized`` only when
    the capture was ``supported``, consent was recorded and a calibration is
    named; otherwise it is ``incomplete`` and the warnings say why.
    """
    require(isinstance(manifest, SessionManifest), "manifest must be a SessionManifest")
    require(isinstance(plan, RigPlan), "plan must be a RigPlan")
    require(isinstance(terms, RecordingTerms), "terms must be RecordingTerms")
    mocap = _ready_tools()

    warnings = list(dict.fromkeys(manifest.reasons))
    if not terms.consent_recorded:
        warnings.append("no consent recorded")
    if not calibration_ids:
        warnings.append("no calibration recorded")
    finalized = manifest.outcome is CaptureOutcome.SUPPORTED and not warnings
    state = mocap.SessionState.FINALIZED if finalized else mocap.SessionState.INCOMPLETE
    session = mocap.MocapSessionManifest(
        session_id=f"{manifest.plan_name}@{manifest.started_utc}",
        created_at_utc=manifest.started_utc,
        state=state,
        world_frame=mocap.CoordinateFrame.affinedrift_world_v1(),
        cameras=tuple(r.identity for r in _camera_records(mocap, manifest, plan)),
        clocks=_clocks(mocap, manifest),
        methods=(
            mocap.MethodDescriptor(
                method_id=RIG_CAPTURE_METHOD_ID,
                version=manifest.schema_version.rsplit("/", 1)[-1],
                implementation="src.motion_capture.rig.session.CaptureSession",
                license_spdx="MIT",
            ),
        ),
        recording_policy=mocap.RecordingPolicy(
            consent_recorded=terms.consent_recorded,
            raw_video_retained=terms.raw_video_retained,
            retention_days=terms.retention_days,
            no_store=terms.no_store,
        ),
        calibration_ids=tuple(calibration_ids),
        warnings=tuple(warnings),
    )
    return mocap.dumps_canonical(session)


def export_to_bundle(
    bundle_dir: Path,
    manifest: SessionManifest,
    plan: RigPlan,
    terms: RecordingTerms = RecordingTerms(),
    *,
    calibration_ids: Sequence[str] = (),
) -> dict[str, str | None]:
    """Write :data:`MOCAP_SESSION_FILE` into the bundle; report the outcome as data.

    Never raises for an absent schema or a rejected policy — the rig manifest
    records ``{"status": ..., "reason": ..., "path": ...}`` so a later stage
    can see exactly why no canonical session exists beside it.
    """
    try:
        text = export_session_manifest(
            manifest, plan, terms, calibration_ids=calibration_ids
        )
    except StateError as exc:
        return {"status": "unavailable", "reason": str(exc), "path": None}
    except (ValueError, TypeError) as exc:
        logger.warning("Tools mocap session export rejected: %s", exc)
        return {"status": "rejected", "reason": str(exc), "path": None}
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / MOCAP_SESSION_FILE
    path.write_text(text, encoding="utf-8")
    return {"status": "written", "reason": None, "path": path.name}
