"""OpenCap session layout discovery and subject metadata (#11403).

An OpenCap session, as downloaded by ``opencap-processing`` or written by
``opencap-core``, is laid out as::

    <session>/sessionMetadata.yaml               subject + model name
    <session>/MarkerData/<trial>.trc             augmented markers
    <session>/OpenSimData/Model/<model>_scaled.osim
    <session>/OpenSimData/Kinematics/<trial>.mot IK result (inDegrees=yes)

The ``neutral`` trial is the static pose OpenCap scales the model from; it is
never chosen implicitly when a session also holds motion trials.

This module only discovers files and reads metadata. It never parses marker
or motion data, so both the source adapter and the session loader can depend
on it without depending on each other.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

NEUTRAL_TRIAL = "neutral"

_YAML_METADATA_FILENAMES = ("sessionMetadata.yaml", "sessionMetadata.yml")
# Pre-#11403 fixtures and hand-assembled sessions used JSON metadata.
_JSON_METADATA_FILENAMES = (
    "sessionMetadata.json",
    "session_metadata.json",
    "metadata.json",
)


def read_session_metadata(session_dir: Path) -> dict[str, Any]:
    """Return the session's metadata mapping, or ``{}`` when none exists.

    ``sessionMetadata.yaml`` (what OpenCap writes) wins over the legacy JSON
    names. Raises ``ValueError`` when a metadata file is not a mapping.
    """
    for filename in _YAML_METADATA_FILENAMES + _JSON_METADATA_FILENAMES:
        path = session_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if isinstance(data, dict):
            return data
        raise ValueError(f"OpenCap metadata {path} must contain a mapping")
    return {}


def _optional_positive(metadata: Mapping[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"OpenCap metadata {key!r} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"OpenCap metadata {key!r} must be > 0, got {value!r}")
    return number


@dataclass(frozen=True)
class OpenCapSubject:
    """Subject anthropometry and model choice recorded by OpenCap.

    Units are SI (kg, m). Fields absent from the metadata are ``None``.
    """

    mass_kg: float | None
    height_m: float | None
    sex: str | None
    opensim_model: str | None
    subject_id: str | None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> OpenCapSubject:
        """Build from a ``sessionMetadata`` mapping.

        Raises:
            TypeError: mass or height is not numeric.
            ValueError: mass or height is not finite and positive.
        """
        sex = metadata.get("gender_mf")
        model = metadata.get("openSimModel")
        subject_id = metadata.get("subjectID")
        return cls(
            mass_kg=_optional_positive(metadata, "mass_kg"),
            height_m=_optional_positive(metadata, "height_m"),
            sex=str(sex) if sex is not None else None,
            opensim_model=str(model) if model is not None else None,
            subject_id=str(subject_id) if subject_id is not None else None,
        )


def _marker_files(session_dir: Path) -> dict[str, Path]:
    marker_dir = session_dir / "MarkerData"
    files = sorted(marker_dir.glob("*.trc")) if marker_dir.is_dir() else []
    if not files:
        # Legacy/hand-assembled sessions: any TRC below the session root.
        files = sorted(
            p for p in session_dir.rglob("*.trc") if "Settings" not in p.parts
        )
    trials: dict[str, Path] = {}
    for path in files:
        if path.stem in trials:
            raise ValueError(
                f"OpenCap session {session_dir} has two marker files for trial "
                f"{path.stem!r}: {trials[path.stem]} and {path}"
            )
        trials[path.stem] = path
    return trials


@dataclass(frozen=True)
class OpenCapSessionLayout:
    """Files that make up one OpenCap session directory."""

    session_dir: Path
    metadata: dict[str, Any] = field(repr=False)
    marker_files: dict[str, Path]
    kinematics_files: dict[str, Path]
    model_candidates: tuple[Path, ...]

    @classmethod
    def discover(cls, session_dir: Path) -> OpenCapSessionLayout:
        """Scan ``session_dir``.

        Raises:
            NotADirectoryError: ``session_dir`` is not a directory.
            ValueError: two marker files claim the same trial name.
        """
        root = Path(session_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"OpenCap session is not a directory: {root}")
        kin_dir = root / "OpenSimData" / "Kinematics"
        model_dir = root / "OpenSimData" / "Model"
        return cls(
            session_dir=root,
            metadata=read_session_metadata(root),
            marker_files=_marker_files(root),
            kinematics_files={
                p.stem: p for p in sorted(kin_dir.glob("*.mot")) if p.is_file()
            },
            model_candidates=tuple(sorted(model_dir.glob("*.osim"))),
        )

    @property
    def trials(self) -> list[str]:
        """Trial names that have marker data, in sorted order."""
        return list(self.marker_files)

    @property
    def subject(self) -> OpenCapSubject:
        return OpenCapSubject.from_metadata(self.metadata)

    def resolve_trial(self, trial: str | None = None) -> str:
        """Return ``trial`` if present, else the session's only motion trial.

        Raises:
            FileNotFoundError: the session has no marker data at all.
            KeyError: ``trial`` is not in the session.
            ValueError: ``trial`` is ``None`` and several motion trials exist.
        """
        if not self.marker_files:
            raise FileNotFoundError(
                f"OpenCap session has no TRC marker file: {self.session_dir}"
            )
        if trial is not None:
            if trial not in self.marker_files:
                raise KeyError(
                    f"Trial {trial!r} not in OpenCap session {self.session_dir}; "
                    f"available: {', '.join(self.trials)}"
                )
            return trial
        motion = [t for t in self.trials if t.lower() != NEUTRAL_TRIAL]
        if len(motion) == 1:
            return motion[0]
        if not motion:
            return self.trials[0]
        raise ValueError(
            f"OpenCap session {self.session_dir} has {len(motion)} trials "
            f"({', '.join(motion)}); pass trial= to choose one"
        )

    def model_file(self) -> Path | None:
        """Return the scaled OpenSim model, or ``None`` if the session has none.

        Prefers ``<openSimModel>_scaled.osim`` named by the metadata, then a
        lone ``.osim``. Raises ``ValueError`` when several models remain and
        the metadata does not say which one is the scaled model.
        """
        if not self.model_candidates:
            return None
        model_name = self.metadata.get("openSimModel")
        if model_name:
            for path in self.model_candidates:
                if path.name == f"{model_name}_scaled.osim":
                    return path
        if len(self.model_candidates) == 1:
            return self.model_candidates[0]
        raise ValueError(
            f"OpenCap session {self.session_dir} has several models "
            f"({', '.join(p.name for p in self.model_candidates)}) and its "
            "metadata does not name the scaled one"
        )
