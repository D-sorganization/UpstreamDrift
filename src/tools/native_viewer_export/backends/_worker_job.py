"""Job description exchanged with subprocess workers (OpenSim, MyoSuite)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.force_overlay.glyphs import GlyphSet


@dataclass(frozen=True)
class WorkerJob:
    """Everything a render worker needs; frames are written as ``.npy`` files."""

    bundle_path: str
    q_path: str
    indices: list[int]
    views: list[str]
    width: int
    height: int
    lookat_m: list[float]
    distance_m: float | None
    out_dir: str
    glyphs_path: str | None = None  # JSON list of GlyphSet dicts aligned to indices

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> WorkerJob:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def frame_path(self, view: str, position: int) -> Path:
        return Path(self.out_dir) / f"{view}_{position:05d}.npy"

    def load_glyph_sets(self) -> list[GlyphSet] | None:
        """Per-frame glyph sets staged for this job, or ``None`` without overlays."""
        if not self.glyphs_path:
            return None
        from src.shared.python.force_overlay.glyphs import GlyphSet

        docs = json.loads(Path(self.glyphs_path).read_text(encoding="utf-8"))
        return [GlyphSet.from_dict(d) for d in docs]
