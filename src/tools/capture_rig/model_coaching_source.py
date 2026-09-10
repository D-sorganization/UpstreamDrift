"""Saved model analysis through the common coaching source and export pipeline."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from src.motion_capture.coaching import DrawingLayer, render_layer
from src.motion_capture.reference.evidence import asset_identity
from src.motion_capture.reference.model import MAX_REFERENCE_BYTES
from src.motion_capture.rig.documents import write_document

from .clips import ClipRange, ClipRendering, verify_frame_clip, write_frame_clip
from .coaching_export import publish_still
from .model_frame_source import ModelFrameSource, ModelViewRecipe
from .swing_export import publish_export
from .swing_export_actions import ExportJob

MAX_MODEL_ANALYSIS_BYTES = MAX_REFERENCE_BYTES + 4_000_000


class ModelAnalysisDocument(BaseModel):
    """One portable model-view recipe and its original-image drawing layer."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["model-analysis/1.0.0"] = "model-analysis/1.0.0"
    recipe: ModelViewRecipe
    drawings: DrawingLayer

    @model_validator(mode="after")
    def matching_drawings(self) -> Self:
        recipe = self.recipe
        width, height = recipe.camera.image_size_px
        expected = (recipe.asset.id, width, height, recipe.frame_count)
        actual = (
            self.drawings.view,
            self.drawings.width,
            self.drawings.height,
            self.drawings.frames,
        )
        if actual != expected:
            raise ValueError(
                "Drawings do not match the model view and display timeline"
            )
        return self


class ModelCoachingSource:
    """A model workspace containing analysis data, never a capture bundle."""

    def __init__(self, recipe: ModelViewRecipe, root: Path) -> None:
        self.root = root
        self.path = root / "model-analysis.json"
        width, height = recipe.camera.image_size_px
        document = ModelAnalysisDocument(
            recipe=recipe,
            drawings=DrawingLayer(
                view=recipe.asset.id,
                width=width,
                height=height,
                frames=recipe.frame_count,
            ),
        )
        if self.path.is_file():
            with self.path.open("rb") as stream:
                content = stream.read(MAX_MODEL_ANALYSIS_BYTES + 1)
            if len(content) > MAX_MODEL_ANALYSIS_BYTES:
                raise ValueError("Model analysis document is too large")
            document = ModelAnalysisDocument.model_validate_json(content)
            if asset_identity(document.recipe.asset) != asset_identity(recipe.asset):
                raise ValueError(
                    "Saved analysis belongs to a changed or different model"
                )
        self.reader = ModelFrameSource(document.recipe)
        self.drawings = document.drawings
        self._saved_recipe = document.recipe

    @property
    def dirty(self) -> bool:
        return self.reader.recipe != self._saved_recipe

    @property
    def recipe(self) -> ModelViewRecipe:
        """Expose the analysis contract without exposing decoder internals."""
        return self.reader.recipe

    def replace_recipe(self, recipe: ModelViewRecipe) -> None:
        """Replace display settings without changing source identity or pixel grid."""
        current = self.reader.recipe
        if (
            asset_identity(current.asset) != asset_identity(recipe.asset)
            or current.fps != recipe.fps
        ):
            raise ValueError(
                "Changing the model source or clock requires a new analysis"
            )
        ModelAnalysisDocument(recipe=recipe, drawings=self.drawings)
        replacement = ModelFrameSource(recipe)
        self.reader.close()
        self.reader = replacement

    def time_at(self, index: int) -> float:
        """Return the model's actual source-clock time."""
        return self.reader.time_at(index)

    def save(self, drawings: DrawingLayer) -> None:
        """Atomically persist the actual model, camera, appearance and drawings."""
        document = ModelAnalysisDocument(recipe=self.reader.recipe, drawings=drawings)
        payload = document.model_dump(mode="json")
        if (
            len(
                json.dumps(
                    payload, ensure_ascii=False, allow_nan=False, indent=2
                ).encode("utf-8")
            )
            + 1
            > MAX_MODEL_ANALYSIS_BYTES
        ):
            raise ValueError("Model analysis document is too large")
        self.root.mkdir(parents=True, exist_ok=True)
        write_document(self.path, payload)
        self.drawings = drawings
        self._saved_recipe = self.reader.recipe

    def still(self, drawings: DrawingLayer, index: int, out: Path) -> None:
        """Publish the same model pixels and drawing renderer used by playback."""
        document = ModelAnalysisDocument(recipe=self.reader.recipe, drawings=drawings)
        image = self.reader.read(index)
        if image is None:
            raise ValueError("Frame is outside the model display timeline")
        metadata = self._metadata(document) | {
            "frame": index,
            "source_seconds": self.time_at(index),
        }
        publish_still(render_layer(image, drawings, index), out, metadata)

    @staticmethod
    def _metadata(document: ModelAnalysisDocument) -> dict[str, Any]:
        return {
            "schema_version": "model-analysis-export/1.0.0",
            "source_kind": "virtual_model",
            "recipe": document.recipe.model_dump(mode="json"),
            "drawings": document.drawings.model_dump(mode="json"),
        }

    def export_job(self, drawings: DrawingLayer) -> ExportJob:
        """Snapshot a cancellable model export independently of later view edits."""
        document = ModelAnalysisDocument(recipe=self.reader.recipe, drawings=drawings)

        def run(
            out: Path,
            cancelled: Callable[[], bool],
            progress: Callable[[int, int], None],
        ) -> None:
            if cancelled():
                raise InterruptedError("Model export cancelled")
            if out.suffix.lower() not in (".avi", ".mp4"):
                raise ValueError("Choose an AVI or MP4 filename")
            if out.exists() or out.with_suffix(".json").exists():
                raise FileExistsError("Choose a new filename for model export")
            source = ModelFrameSource(document.recipe)
            try:
                with TemporaryDirectory(
                    prefix=".model-analysis-", dir=out.parent
                ) as temporary:
                    staged = Path(temporary) / out.name
                    rendering = ClipRendering(
                        drawings=drawings,
                        strict=True,
                        clock=False,
                        cancelled=cancelled,
                        progress=progress,
                    )
                    result = write_frame_clip(
                        source,
                        ClipRange(0, source.frame_count - 1),
                        staged,
                        rendering=rendering,
                    )
                    verify_frame_clip(
                        staged,
                        source.frame_count,
                        rendering.size(source.width, source.height),
                        cancelled,
                    )
                    write_document(
                        staged.with_suffix(".json"),
                        self._metadata(document)
                        | {
                            "frames": result["frames"],
                            "fps": result["fps"],
                            "source_times_s": [
                                source.time_at(i) for i in range(source.frame_count)
                            ],
                        },
                    )
                    if cancelled():
                        raise InterruptedError("Model export cancelled")
                    publish_export(staged, out)
            finally:
                source.close()

        return run
