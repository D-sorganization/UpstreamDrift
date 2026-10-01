"""Local historical-player library routes shared by web and desktop hosts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Iterator, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from src.api.routes.matched_swings import require_local_client
from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace import DatasetMetadata, NecromatcherLibrary
from src.shared.python.workspace.necromatcher import default_necromatcher_library
from src.shared.python.workspace.necromatcher_review import CaptureReview

router = APIRouter(
    prefix="/necromatcher",
    tags=["necromatcher"],
    dependencies=[Depends(require_local_client)],
)


@lru_cache(maxsize=1)
def get_library() -> NecromatcherLibrary:
    """Resolve a configurable library beneath the canonical user-state root."""
    return default_necromatcher_library()


Library = Annotated[NecromatcherLibrary, Depends(get_library)]


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
