"""Portable reference catalog with atomic metadata updates and stable identities."""

from pathlib import Path
from dataclasses import dataclass
import json
from uuid import UUID

from src.motion_capture.rig.documents import write_document

from .model import ASSET_ADAPTER, MAX_REFERENCE_BYTES, Asset


@dataclass(frozen=True)
class ReferenceScan:
    assets: tuple[Asset, ...]
    problems: tuple[str, ...]


class ReferenceLibrary:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def _path(self, identity: str) -> Path:
        if str(UUID(identity)) != identity:
            raise ValueError("Reference identity must be a canonical UUID")
        return self.root / f"{identity}.json"

    def save(self, asset: Asset) -> None:
        validated = ASSET_ADAPTER.validate_python(asset.model_dump())
        content = (
            json.dumps(
                validated.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        )
        if len(content.encode("utf-8")) > MAX_REFERENCE_BYTES:
            raise ValueError("Reference is too large; trim the source before importing")
        self.root.mkdir(parents=True, exist_ok=True)
        write_document(self._path(validated.id), validated.model_dump(mode="json"))

    def load(self, identity: str) -> Asset:
        path = self._path(identity)
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError("Reference document exceeds the supported size")
        asset = ASSET_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
        if asset.id != identity:
            raise ValueError(
                "Reference document identity differs from its catalog filename"
            )
        return asset

    def list(self, *, archived: bool = False) -> list[Asset]:
        assets = [self.load(path.stem) for path in sorted(self.root.glob("*.json"))]
        return sorted(
            (a for a in assets if a.archived == archived),
            key=lambda a: a.title.casefold(),
        )

    def scan(self, *, archived: bool = False) -> ReferenceScan:
        assets: list[Asset] = []
        problems: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                asset = self.load(path.stem)
                if asset.archived == archived:
                    assets.append(asset)
            except (ValueError, OSError) as exc:
                problems.append(f"{path.name}: {exc}")
        return ReferenceScan(
            tuple(sorted(assets, key=lambda a: a.title.casefold())), tuple(problems)
        )
