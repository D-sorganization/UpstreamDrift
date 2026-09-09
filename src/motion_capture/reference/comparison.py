"""Comparison workspace models, saved layers and reproducible export sidecars (#9866).

Defines:
- ComparisonLayer: Appearance and visibility settings for comparison overlays.
- ComparisonSession: Saved comparison state associating user capture with a reference.
- ExportSidecar and functions for creating reproducible export documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from src.motion_capture.provenance import now_utc, sha256_of, stamp
from src.motion_capture.reference.model import Asset
from src.motion_capture.reference.registration import ReferenceRegistration, TimeMapping
from src.motion_capture.rig.edits import CropRect
from src.shared.python.core.contracts import require

COMPARISON_SESSION_SCHEMA = "comparison-session/1.0.0"
COMPARISON_EXPORT_SCHEMA = "comparison-export/1.0.0"
DEFAULT_REFERENCE_COLOUR = "#00dcff"
MAX_COMPARISON_BYTES = 256_000
ComparisonView = Annotated[
    str, Field(min_length=1, max_length=64, pattern=r"^[\w -]+$")
]
_VIEW_ADAPTER = TypeAdapter(ComparisonView)


class ComparisonLayer(BaseModel):
    """Appearance and visibility settings for a reference overlay."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    colour: str = Field(default=DEFAULT_REFERENCE_COLOUR, pattern=r"^#[0-9a-fA-F]{6}$")
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    visible: bool = True
    draw_skeleton: bool = True
    draw_joints: bool = True
    line_width: int = Field(default=2, ge=1, le=20)

    @property
    def colour_bgr(self) -> tuple[int, int, int]:
        """Convert #rrggbb hex colour to BGR tuple for OpenCV rendering."""
        clean = self.colour.lstrip("#")
        r, g, b = (int(clean[i : i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)


class ComparisonSession(BaseModel):
    """Saved comparison session linking a capture recording to an expert reference."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["comparison-session/1.0.0"] = "comparison-session/1.0.0"
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_root: str = Field(min_length=1)
    view: ComparisonView
    reference_id: str
    reference_kind: Literal["motion", "video"]
    registration: ReferenceRegistration | None = None
    layer: ComparisonLayer = Field(default_factory=ComparisonLayer)
    notes: str = Field(default="", max_length=50_000)
    created_utc: str = Field(default_factory=now_utc)

    @model_validator(mode="after")
    def validate_session(self) -> Self:
        if str(UUID(self.id)) != self.id:
            raise ValueError("Comparison session id must be a canonical UUID")
        if str(UUID(self.reference_id)) != self.reference_id:
            raise ValueError("Reference id must be a canonical UUID")
        if self.registration and self.registration.reference_id != self.reference_id:
            raise ValueError("Registration must belong to the comparison reference")
        return self

    def changed(self, **values: object) -> Self:
        return type(self).model_validate(self.model_dump() | values)

    @property
    def layer_visible(self) -> bool:
        return self.layer.visible

    @property
    def layer_opacity(self) -> float:
        return self.layer.opacity

    @property
    def layer_colour(self) -> str:
        return self.layer.colour

    @property
    def layer_line_width(self) -> int:
        return self.layer.line_width

    def with_layer(
        self,
        *,
        colour: str | None = None,
        opacity: float | None = None,
        visible: bool | None = None,
        line_width: int | None = None,
    ) -> Self:
        current = self.layer
        new_layer = current.model_validate(
            current.model_dump()
            | {
                "colour": colour if colour is not None else current.colour,
                "opacity": opacity if opacity is not None else current.opacity,
                "visible": visible if visible is not None else current.visible,
                "line_width": line_width
                if line_width is not None
                else current.line_width,
            }
        )
        return self.changed(layer=new_layer)


def comparison_session_path(root: Path, view: str, reference_id: str) -> Path:
    """Conventional path for saved comparison session sidecars."""
    _VIEW_ADAPTER.validate_python(view)
    if str(UUID(reference_id)) != reference_id:
        raise ValueError("Reference id must be a canonical UUID")
    return root / "comparisons" / f"{view}_{reference_id}.json"


def save_comparison_session(session: ComparisonSession, root: Path) -> Path:
    """Save comparison session to root/comparisons/<view>_<reference_id>.json."""
    from src.motion_capture.rig.documents import write_document

    out = comparison_session_path(root, session.view, session.reference_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_document(out, session.model_dump(mode="json"))
    return out


def load_comparison_session(path: Path) -> ComparisonSession:
    """Load comparison session from a JSON sidecar."""
    require(path.is_file(), "Comparison file does not exist", str(path))
    if path.stat().st_size > MAX_COMPARISON_BYTES:
        raise ValueError("Comparison file is too large")
    return ComparisonSession.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ComparisonExportSidecarSpec:
    """Metadata parameters for reproducible comparison export sidecars."""

    video_out: Path
    source_media: Path
    reference_asset: Asset
    view: str
    fps: float
    frame_count: int
    output_frame_times: list[float]
    registration: ReferenceRegistration | None = None
    time_mapping: TimeMapping | None = None
    crop: CropRect | None = None
    layer: ComparisonLayer | None = None


def build_comparison_sidecar(spec: ComparisonExportSidecarSpec) -> dict[str, Any]:
    """Assemble reproducible comparison export metadata sidecar."""
    source_sha = sha256_of(spec.source_media)
    ref_asset = spec.reference_asset
    ref_source = ref_asset.source
    ref_source_sha = ref_source.sha256
    ref_path = ref_source.path

    registration = spec.registration
    is_3d = ref_asset.kind == "motion"
    has_calib = bool(registration and registration.is_calibrated)

    if is_3d:
        alignment_status = (
            "calibrated_3d_projection" if has_calib else "uncalibrated_3d_projection"
        )
    else:
        alignment_status = "manual_2d_homography_no_3d_claim"

    time_mapping_dict: dict[str, Any] | None = None
    if spec.time_mapping:
        time_mapping_dict = spec.time_mapping.model_dump(mode="json")
    elif registration and registration.time_mapping:
        reg_tm = registration.time_mapping
        time_mapping_dict = reg_tm.model_dump(mode="json")

    sidecar: dict[str, Any] = {
        "schema_version": COMPARISON_EXPORT_SCHEMA,
        "video_file": spec.video_out.name,
        "created_utc": now_utc(),
        "source": {
            "path": str(spec.source_media),
            "sha256": source_sha,
            "view": spec.view,
        },
        "reference": {
            "id": ref_asset.id,
            "title": ref_asset.title,
            "kind": ref_asset.kind,
            "source_path": ref_path,
            "source_sha256": ref_source_sha,
            "alignment_status": alignment_status,
            "is_3d": is_3d,
            "is_calibrated": has_calib,
            "missing_alignment_evidence": not has_calib,
        },
        "registration": (
            registration.model_dump(mode="json") if registration else None
        ),
        "time_mapping": time_mapping_dict,
        "crop": spec.crop.model_dump(mode="json") if spec.crop else None,
        "layer_appearance": (
            spec.layer.model_dump(mode="json")
            if spec.layer
            else ComparisonLayer().model_dump()
        ),
        "playback": {
            "fps": spec.fps,
            "frame_count": spec.frame_count,
            "output_frame_times": spec.output_frame_times,
        },
    }
    inputs = [spec.source_media]
    if Path(ref_path).is_file():
        inputs.append(Path(ref_path))
    return stamp(
        sidecar,
        schema_version=COMPARISON_EXPORT_SCHEMA,
        module="src.motion_capture.reference.comparison",
        inputs=inputs,
        parameters={
            "view": spec.view,
            "fps": spec.fps,
            "alignment_status": alignment_status,
        },
    )
