"""Local historical-player library routes shared by web and desktop hosts."""

from __future__ import annotations

from contextlib import contextmanager, asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Iterator, Literal, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from src.api.routes.matched_swings import require_local_client
from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace import (
    DatasetMetadata,
    NecromatcherLibrary,
    project_fit_frame,
    NativeRefitSession,
    NativeVideoSession,
    NativeRefitOptions,
    refit_plan,
)
from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ShaftAxisEvidence,
)
from src.shared.python.motion_matching.pipeline.plant import EngineUnavailableError
from src.shared.python.workspace.necromatcher import default_necromatcher_library
from src.shared.python.workspace.necromatcher_review import CaptureReview
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from src.shared.python.workspace.necromatcher_caption import CaptionOverlayOptions


@asynccontextmanager
async def refits_lifespan(_app: object) -> AsyncIterator[None]:
    try:
        yield
    finally:
        if get_refits.cache_info().currsize:
            get_refits().close()
            get_refits.cache_clear()
        if get_video_exports.cache_info().currsize:
            get_video_exports().close()
            get_video_exports.cache_clear()


router = APIRouter(
    prefix="/necromatcher",
    tags=["necromatcher"],
    dependencies=[Depends(require_local_client)],
    lifespan=refits_lifespan,
)


@lru_cache(maxsize=1)
def get_library() -> NecromatcherLibrary:
    """Resolve a configurable library beneath the canonical user-state root."""
    return default_necromatcher_library()


Library = Annotated[NecromatcherLibrary, Depends(get_library)]


@lru_cache(maxsize=1)
def get_refits() -> NativeRefitSession:
    return NativeRefitSession(get_library())


Refits = Annotated[NativeRefitSession, Depends(get_refits)]


@lru_cache(maxsize=1)
def get_video_exports() -> NativeVideoSession:
    """Share owned export jobs and verified artifact recall across API requests."""
    return NativeVideoSession(get_library())


VideoExports = Annotated[NativeVideoSession, Depends(get_video_exports)]


class RefitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    new_fit_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    frame_indices: list[int] = Field(min_length=2, max_length=1000)
    knot_count: int = Field(ge=2, le=1000)
    coordinate_scales: list[float] = Field(min_length=1, max_length=256)
    max_iterations: int = Field(default=100, ge=1, le=10000)
    prior_weight: float = Field(default=0.1, ge=0)
    smoothness_weight: float = Field(default=0.01, ge=0)
    closure_weight: float = Field(default=100.0, ge=0)
    unknown_visibility_weight: float = Field(default=0.5, ge=0, le=1)
    budget_wall_s: float = Field(default=600.0, gt=0, le=3600)
    config: dict[str, Any] | None = None
    shaft_evidence: dict[str, Any] | None = None
    operation: Literal["fit", "author_initialization"] = "fit"
    initialization_source: Literal["sampled_parent", "preserved_spline"] = (
        "sampled_parent"
    )

    def options(self) -> NativeRefitOptions:
        legacy = {
            "max_iterations",
            "prior_weight",
            "smoothness_weight",
            "closure_weight",
        }
        if self.config is not None and self.model_fields_set & legacy:
            raise ValueError("Nested config cannot be mixed with legacy scalar options")
        config = (
            ImageFitConfig.from_record(self.config)
            if self.config is not None
            else ImageFitConfig(
                self.max_iterations,
                self.prior_weight,
                self.smoothness_weight,
                self.closure_weight,
            )
        )
        return NativeRefitOptions(
            tuple(self.frame_indices),
            self.knot_count,
            tuple(self.coordinate_scales),
            config,
            self.unknown_visibility_weight,
            self.budget_wall_s,
            self.operation,
            self.initialization_source,
        )


class VideoExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    shaft_evidence: dict[str, Any] | None = None
    shape_overlay: dict[str, Any] | None = None
    caption_overlay: dict[str, Any] | None = None


class IdentityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)


class SwingRequest(IdentityRequest):
    player_id: str = Field(min_length=1, max_length=128)


class AssetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    source_path: str = Field(min_length=1)


class ModelRequest(AssetRequest):
    engine: Literal["mujoco", "drake", "pinocchio", "opensim", "simscape"]
    dofs: tuple[str, ...] = Field(min_length=1)


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except StateError as exc:
        raise HTTPException(409, str(exc)) from exc
    except EngineUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, "Unknown library identity") from exc
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, "Referenced evidence is missing") from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc


def _asset_response(asset: DatasetMetadata) -> dict[str, Any]:
    """Expose identities and provenance without server filesystem paths."""
    return {
        "dataset_id": asset.dataset_id,
        "session_id": asset.session_id,
        "kind": asset.kind,
        "metadata": asset.metadata,
    }


@router.get("/fits/{fit_id}/refit-plan")
def get_refit_plan(fit_id: str, library: Library) -> dict[str, Any]:
    with _errors():
        return refit_plan(library, fit_id)


@router.post("/fits/{fit_id}/refits", status_code=202)
def submit_refit(fit_id: str, request: RefitRequest, refits: Refits) -> dict[str, Any]:
    with _errors():
        try:
            options = request.options()
            if request.shaft_evidence is None:
                return refits.submit(fit_id, request.new_fit_id, options)
            evidence = ShaftAxisEvidence.from_record(request.shaft_evidence)
            return refits.submit(fit_id, request.new_fit_id, options, evidence)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.get("/refits/{run_id}")
def view_refit(run_id: str, refits: Refits) -> dict[str, Any]:
    with _errors():
        return refits.view(run_id)


@router.post("/refits/{run_id}/cancel")
def cancel_refit(run_id: str, refits: Refits) -> dict[str, Any]:
    with _errors():
        return refits.cancel(run_id)


@router.post("/fits/{fit_id}/video-exports", status_code=202)
def submit_video_export(
    fit_id: str, exports: VideoExports, request: VideoExportRequest | None = None
) -> dict[str, Any]:
    """Queue a source-bound video review without accepting a host output path."""
    with _errors():
        try:
            if request is None:
                return exports.submit(fit_id)
            evidence = (
                ShaftAxisEvidence.from_record(request.shaft_evidence)
                if request.shaft_evidence is not None
                else None
            )
            options: dict[str, Any] = (
                {
                    "shape_overlay": ShapeOverlayOptions.from_record(
                        request.shape_overlay
                    )
                }
                if request.shape_overlay is not None
                else {}
            )
            if request.caption_overlay is not None:
                options["caption_overlay"] = CaptionOverlayOptions.from_record(
                    request.caption_overlay
                )
            if evidence is None:
                return exports.submit(fit_id, **options)
            return exports.submit(fit_id, evidence, **options)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc


@router.get("/video-exports/{run_id}")
def view_video_export(run_id: str, exports: VideoExports) -> dict[str, Any]:
    with _errors():
        return exports.view(run_id)


@router.post("/video-exports/{run_id}/cancel")
def cancel_video_export(run_id: str, exports: VideoExports) -> dict[str, Any]:
    with _errors():
        return exports.cancel(run_id)


@router.get("/video-exports/{run_id}/download")
def download_video_export(run_id: str, exports: VideoExports) -> FileResponse:
    """Serve only the session's revalidated completed review ZIP."""
    with _errors():
        try:
            path = exports.download(run_id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return FileResponse(
            path,
            media_type="application/zip",
            filename=f"necromatcher-overlay-{run_id}.zip",
        )


@router.get("/players")
def players(library: Library) -> dict[str, Any]:
    with _errors():
        return {"players": [asdict(x) for x in library.players()]}


@router.post("/players", status_code=201)
def add_player(request: IdentityRequest, library: Library) -> dict[str, Any]:
    with _errors():
        return asdict(library.add_player(request.id, request.name))


@router.get("/swings")
def swings(library: Library, player_id: str | None = None) -> dict[str, Any]:
    with _errors():
        return {"swings": [asdict(x) for x in library.swings(player_id)]}


@router.post("/swings", status_code=201)
def add_swing(request: SwingRequest, library: Library) -> dict[str, Any]:
    with _errors():
        return asdict(library.add_swing(request.id, request.player_id, request.name))


@router.get("/swings/{swing_id}/assets")
def assets(swing_id: str, library: Library) -> dict[str, Any]:
    with _errors():
        return {
            "assets": [
                _asset_response(library.load_asset(x.dataset_id))
                for x in library.assets(swing_id)
            ]
        }


@router.post("/swings/{swing_id}/models", status_code=201)
def add_model(swing_id: str, request: ModelRequest, library: Library) -> dict[str, Any]:
    with _errors():
        return _asset_response(
            library.add_model(
                request.id,
                swing_id,
                Path(request.source_path),
                engine=request.engine,
                dofs=request.dofs,
            )
        )


@router.post("/swings/{swing_id}/profiles", status_code=201)
def add_profile(
    swing_id: str, request: AssetRequest, library: Library
) -> dict[str, Any]:
    with _errors():
        return _asset_response(
            library.add_profile(request.id, swing_id, Path(request.source_path))
        )


@router.post("/swings/{swing_id}/captures", status_code=201)
def add_capture(
    swing_id: str, request: AssetRequest, library: Library
) -> dict[str, Any]:
    with _errors():
        return _asset_response(
            library.add_capture(request.id, swing_id, Path(request.source_path))
        )


@router.post("/swings/{swing_id}/fits", status_code=201)
def add_fit(swing_id: str, request: AssetRequest, library: Library) -> dict[str, Any]:
    """Import an immutable source-bound research trajectory through the library."""
    with _errors():
        return _asset_response(
            library.add_fit(request.id, swing_id, Path(request.source_path))
        )


@router.post("/swings/{swing_id}/replays", status_code=201)
def add_replay(
    swing_id: str, request: AssetRequest, library: Library
) -> dict[str, Any]:
    """Import a checked authored trace using shared immutable library admission."""
    with _errors():
        return _asset_response(
            library.add_replay(request.id, swing_id, Path(request.source_path))
        )


@router.get("/replays/{replay_id}")
def replay_summary(replay_id: str, library: Library) -> dict[str, Any]:
    """Expose checked replay provenance without host filesystem paths."""
    with _errors():
        trace = library.load_replay(replay_id)
        return {
            "replay_id": replay_id,
            "sample_count": len(trace.t),
            "dt_s": trace.dt,
            "backend": trace.backend,
            "metadata": dict(trace.meta),
        }


@router.get("/replays/{replay_id}/data")
def replay_data(replay_id: str, library: Library) -> FileResponse:
    """Download the canonical HDF5 after revalidating replay and parent versions."""
    with _errors():
        library.load_replay(replay_id)
        asset = library.load_asset(replay_id)
        return FileResponse(
            asset.path, media_type="application/x-hdf5", filename=f"{replay_id}.h5"
        )


@router.get("/fits/{fit_id}")
def fit_summary(fit_id: str, library: Library) -> dict[str, Any]:
    """Return verified fit identity and selectable source indices without paths."""
    with _errors():
        fit = library.load_fit(fit_id)
        return {
            "fit_id": fit_id,
            "frame_count": len(fit["frame_indices"]),
            **{
                key: fit[key]
                for key in (
                    "model_id",
                    "model_hash",
                    "capture_id",
                    "capture_hash",
                    "coordinate_order",
                    "coordinate_units",
                    "frame_indices",
                    "qualification",
                    "physical_time_qualified",
                    "dynamics_replayed",
                )
            },
        }


@router.get("/fits/{fit_id}/frames/{frame_index}")
def fit_frame(fit_id: str, frame_index: int, library: Library) -> dict[str, Any]:
    """Recall one native sample by original capture index, including exact PTS."""
    with _errors():
        fit = library.load_fit(fit_id)
        try:
            position = fit["frame_indices"].index(frame_index)
        except ValueError as exc:
            raise IndexError("Source frame has no stored fit sample") from exc
        return {
            "fit_id": fit_id,
            "frame_index": frame_index,
            "frame": fit["frames"][position],
            "q": fit["q"][position],
            **{
                key: fit[key]
                for key in (
                    "model_id",
                    "capture_id",
                    "coordinate_order",
                    "coordinate_units",
                    "qualification",
                    "physical_time_qualified",
                    "dynamics_replayed",
                )
            },
        }


@router.get("/fits/{fit_id}/frames/{frame_index}/projection")
def fit_projection(fit_id: str, frame_index: int, library: Library) -> dict[str, Any]:
    """Project verified native samples with their explicit research camera."""
    with _errors():
        return project_fit_frame(library, fit_id, frame_index)


@router.get("/swings/{swing_id}/export")
def export_swing(swing_id: str, library: Library) -> FileResponse:
    temporary = TemporaryDirectory(prefix="necromatcher-export-")
    try:
        destination = Path(temporary.name) / "swing.zip"
        with _errors():
            library.export_swing(swing_id, destination)
        return FileResponse(
            destination,
            media_type="application/zip",
            filename=f"{swing_id}.zip",
            background=BackgroundTask(temporary.cleanup),
        )
    except (HTTPException, OSError):
        temporary.cleanup()
        raise


@lru_cache(maxsize=4)
def _capture_review(root: str, capture_id: str) -> CaptureReview:
    """Verify an archive once per active review rather than hash it per frame."""
    return CaptureReview(NecromatcherLibrary(root), capture_id)


@contextmanager
def _review_access(
    library: NecromatcherLibrary, capture_id: str
) -> Iterator[CaptureReview]:
    """Invalidate changed evidence so a retry must verify the archive anew."""
    with _errors():
        try:
            yield _capture_review(str(library.root), capture_id)
        except StateError:
            _capture_review.cache_clear()
            raise


@router.get("/captures/{capture_id}/frames/{frame_index}")
def capture_frame(
    capture_id: str, frame_index: int, library: Library
) -> dict[str, Any]:
    with _review_access(library, capture_id) as review:
        return review.frame(frame_index)


@router.get("/captures/{capture_id}/frames/{frame_index}/image")
def capture_image(capture_id: str, frame_index: int, library: Library) -> Response:
    with _review_access(library, capture_id) as review:
        image = review.image(frame_index)
        return Response(image, media_type="image/png")
