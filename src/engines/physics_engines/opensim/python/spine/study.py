"""Deterministic, hash-pinned spine study: request -> receipt (SPN-13a, #12766).

``run_spine_study`` chains the merged pipeline

    golf motion -> segmental drive (+ .mot) -> spinal inverse dynamics
    -> trunk static optimisation -> joint reactions -> context bands

and returns a :class:`SpineStudyReceipt`.  The receipt records the SHA-256 of
every input (motion, golf model, segmental model, external loads, configuration
JSON), the code version, per-stage provenance, the admission status of a
licensed asset, the hashed output artifacts and headline residual, reserve and
context-band numbers.  There is no timestamp and no absolute path, so the same
request on the same inputs gives byte-identical receipt JSON.

Status is fail-closed: a licensed asset that is not ``QUALIFIED`` (including
``MISSING``) gives ``unqualified`` and never ``pass``; a stage exception gives
``failed``.  ``pass`` is a software statement only (``scientific_status`` stays
``unqualified``).  Context-band comparisons are reported, never acceptance.

CLI: ``python3 -m src.engines.physics_engines.opensim.python.spine.study
--request request.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy

from src.engines.physics_engines.opensim.python.spine import study_stages
from src.engines.physics_engines.opensim.python.spine.assets import (
    DEFAULT_MANIFEST_PATH,
    SpineAssetReceipt,
    SpineAssetRequest,
    SpineAssetStatus,
    admit_spine_asset,
    resolve_spine_asset,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    DEFAULT_LEVEL_MAP_PATH,
)
from src.engines.physics_engines.opensim.python.spine import study_variant
from src.engines.physics_engines.opensim.python.spine.study_receipt import (
    FileRecord,
    SpineStudyReceipt,
    load_receipt,
    receipt_to_json,
    write_receipt,
)
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.pose_estimation.model_files import sha256_of
from src.shared.python.spine.context_bands import DEFAULT_BANDS_PATH
from src.shared.python.spine.coupling_data import DEFAULT_PATH as COUPLING_LAWS_PATH
from src.shared.python.spine.movement_variants import DEFAULT_VARIANTS_PATH

logger = logging.getLogger(__name__)

__all__ = [
    "SpineStudyReceipt",
    "SpineStudyRequest",
    "load_receipt",
    "main",
    "request_fingerprint",
    "run_spine_study",
    "write_receipt",
]

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_DATAFILE_RE = re.compile(r"<datafile>\s*([^<]+?)\s*</datafile>")
_EXPONENTS = (2, 3)
_PELVIS_KINDS = ("golf", "fixture")
_OPTIONAL_PATHS = ("model_path", "external_loads")


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def _optional_count(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int or None")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _events(value: object) -> tuple[tuple[str, float], ...]:
    if isinstance(value, Mapping):
        pairs: Any = value.items()
    elif isinstance(value, Iterable):
        pairs = value
    else:
        raise TypeError("events must be a mapping or a sequence of pairs")
    out = []
    for name, time in pairs:
        if not isinstance(name, str) or not name:
            raise ValueError("events names must be non-empty strings")
        if not math.isfinite(time):
            raise ValueError(f"events[{name!r}] must be finite, got {time!r}")
        out.append((name, float(time)))
    return tuple(sorted(out))


@dataclass(frozen=True)
class SpineStudyRequest:
    """Everything that defines one spine study (frozen).

    Attributes:
        study_id: file-name-safe identifier (letters, digits, ``_.-``).
        motion_path / golf_model_path: golf swing ``.sto``/``.mot`` and the
            golf humanoid ``.osim`` that gives its trunk rotation.
        model_id: level-map registry key of the segmental model.
        model_path: segmental ``.osim``; ``None`` only with ``asset_id``, which
            is then resolved through the asset manifest environment variable.
        asset_id: manifest id of a licensed asset; the model must be admitted
            by SHA-256 or the study is ``unqualified``.
        coupling_law_id: ``coupling_data`` law id, or ``None`` to extract the
            law from the model's own couplers.
        restrict_law_to_model: restrict the named law to the model's mapped
            level range (SPN-14a, ``coupling.restrict_to_levels``); requires
            ``coupling_law_id``.  Recorded in the receipt's law version.
        cost_exponent: static-optimisation exponent (2 or 3).
        external_loads: ExternalLoads XML or ``None`` (gravity only).
        events / stride / max_frames: static-optimisation frame selection.
        pelvis: ground-pelvis encoding, ``golf`` or ``fixture``.
        drive_tolerance_rad / filter_hz: drive residual flag, ID/JR filter.
        residual_force_fraction_bw / residual_moment_fraction: pelvis residual
            budget of the inverse dynamics.
        reserve_limit_nm / reserve_max_share: reserve-actuator limits.
        variant_id / variant_parameter / variant_handedness: optional shipped
            movement variant (SPN-13d) applied to the golf trunk input, its
            parameter (``None`` = the variant default; checked against the
            declared bounds) and the golfer's handedness.  ``None`` runs the
            unvaried study, byte-identical to a request without these fields.
        output_dir: where artifacts and the receipt are written (never
            recorded in the receipt).

    Preconditions (``TypeError``/``ValueError``): see the field notes; numeric
    tolerances finite and positive, paths ``pathlib.Path``.
    """

    study_id: str
    motion_path: Path
    golf_model_path: Path
    model_id: str
    output_dir: Path
    model_path: Path | None = None
    asset_id: str | None = None
    coupling_law_id: str | None = None
    cost_exponent: int = 3
    external_loads: Path | None = None
    events: Any = field(default=())
    stride: int | None = None
    max_frames: int | None = None
    pelvis: str = "golf"
    drive_tolerance_rad: float | None = None
    filter_hz: float | None = None
    residual_force_fraction_bw: float = 0.05
    residual_moment_fraction: float = 0.01
    reserve_limit_nm: float = 1.0
    reserve_max_share: float = 0.05
    restrict_law_to_model: bool = False
    variant_id: str | None = None
    variant_parameter: float | None = None
    variant_handedness: str = "right"

    def __post_init__(self) -> None:
        if not isinstance(self.study_id, str) or not _ID_RE.fullmatch(self.study_id):
            raise ValueError(f"study_id must match {_ID_RE.pattern}: {self.study_id!r}")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        for name in ("motion_path", "golf_model_path", "output_dir"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a pathlib.Path")
        for name in _OPTIONAL_PATHS:
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be a pathlib.Path or None")
        if self.model_path is None and self.asset_id is None:
            raise ValueError("model_path or asset_id is required")
        if self.asset_id is not None and not (
            isinstance(self.asset_id, str) and self.asset_id.strip()
        ):
            raise ValueError("asset_id must be a non-empty string or None")
        if self.coupling_law_id is not None and not isinstance(
            self.coupling_law_id, str
        ):
            raise TypeError("coupling_law_id must be a str or None")
        if not isinstance(self.restrict_law_to_model, bool):
            raise TypeError("restrict_law_to_model must be a bool")
        if self.restrict_law_to_model and self.coupling_law_id is None:
            raise ValueError("restrict_law_to_model requires a coupling_law_id")
        if isinstance(self.cost_exponent, bool) or self.cost_exponent not in _EXPONENTS:
            raise ValueError(
                f"cost_exponent must be one of {_EXPONENTS}, got {self.cost_exponent!r}"
            )
        if self.pelvis not in _PELVIS_KINDS:
            raise ValueError(f"pelvis must be one of {_PELVIS_KINDS}")
        _optional_count(self.stride, "stride")
        _optional_count(self.max_frames, "max_frames")
        for name in ("drive_tolerance_rad", "filter_hz"):
            if getattr(self, name) is not None:
                _positive(getattr(self, name), name)
        for name in (
            "residual_force_fraction_bw",
            "residual_moment_fraction",
            "reserve_limit_nm",
            "reserve_max_share",
        ):
            _positive(getattr(self, name), name)
        study_variant.validate_request_variant(
            self.variant_id, self.variant_parameter, self.variant_handedness
        )
        object.__setattr__(self, "events", _events(self.events))

    def canonical_dict(self) -> dict[str, Any]:
        """Path-free, JSON-compatible definition (basenames, no output_dir).

        ``restrict_law_to_model`` appears only when set, so the fingerprints of
        requests that predate it are unchanged.
        """

        def base(path: Path | None) -> str | None:
            return None if path is None else path.name

        data = {
            "study_id": self.study_id,
            "motion": base(self.motion_path),
            "golf_model": base(self.golf_model_path),
            "model": base(self.model_path),
            "model_id": self.model_id,
            "asset_id": self.asset_id,
            "coupling_law_id": self.coupling_law_id,
            "cost_exponent": self.cost_exponent,
            "external_loads": base(self.external_loads),
            "events": dict(self.events),
            "stride": self.stride,
            "max_frames": self.max_frames,
            "pelvis": self.pelvis,
            "drive_tolerance_rad": self.drive_tolerance_rad,
            "filter_hz": self.filter_hz,
            "residual_force_fraction_bw": self.residual_force_fraction_bw,
            "residual_moment_fraction": self.residual_moment_fraction,
            "reserve_limit_nm": self.reserve_limit_nm,
            "reserve_max_share": self.reserve_max_share,
            **({"restrict_law_to_model": True} if self.restrict_law_to_model else {}),
        }
        if self.variant_id is not None:
            data["variant"] = study_variant.request_record(
                self.variant_id, self.variant_parameter, self.variant_handedness
            )
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base_dir: Path) -> SpineStudyRequest:
        """Build from a JSON object; relative paths resolve against ``base_dir``."""
        values = dict(data)
        for name in (
            "motion_path",
            "golf_model_path",
            "model_path",
            "external_loads",
            "output_dir",
        ):
            if values.get(name) is not None:
                values[name] = (base_dir / Path(values[name])).resolve()
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"invalid request: {exc}") from exc


# ------------------------------------------------------------------- hashing


def _config_files(request: SpineStudyRequest) -> dict[str, Path]:
    files = {
        "spine_coupling_laws.json": COUPLING_LAWS_PATH,
        "spine_model_level_maps.json": DEFAULT_LEVEL_MAP_PATH,
        "spine_golf_context_bands.json": DEFAULT_BANDS_PATH,
    }
    if request.asset_id is not None:
        files["spine_model_assets.json"] = DEFAULT_MANIFEST_PATH
    if request.variant_id is not None:
        files[study_variant.VARIANTS_CONFIG_NAME] = Path(DEFAULT_VARIANTS_PATH)
    return files


def _external_load_files(path: Path) -> list[FileRecord]:
    records = [FileRecord("external_loads", path.name, sha256_of(path))]
    for name in _DATAFILE_RE.findall(path.read_text(encoding="utf-8")):
        data = path.parent / name
        if data.is_file():
            records.append(
                FileRecord("external_loads_data", data.name, sha256_of(data))
            )
    return records


def _input_records(request: SpineStudyRequest, model: Path | None) -> list[FileRecord]:
    """Hash every existing input in a fixed role order."""
    records = [
        FileRecord("motion", request.motion_path.name, sha256_of(request.motion_path)),
        FileRecord(
            "golf_model",
            request.golf_model_path.name,
            sha256_of(request.golf_model_path),
        ),
    ]
    if model is not None and model.is_file():
        records.append(FileRecord("segmental_model", model.name, sha256_of(model)))
    if request.external_loads is not None:
        records.extend(_external_load_files(request.external_loads))
    for name, path in sorted(_config_files(request).items()):
        records.append(FileRecord(f"config:{name}", name, sha256_of(path)))
    return records


def _require_inputs(request: SpineStudyRequest) -> None:
    for label, path in (
        ("motion", request.motion_path),
        ("golf model", request.golf_model_path),
        ("segmental model", request.model_path),
        ("external loads", request.external_loads),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path.name}")


def _fingerprint(request: SpineStudyRequest, records: Sequence[FileRecord]) -> str:
    payload = {
        "request": request.canonical_dict(),
        "inputs": [r.as_dict() for r in records],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_fingerprint(request: SpineStudyRequest) -> str:
    """SHA-256 over the path-free request and the hash of every input file.

    Precondition: the motion and golf model files exist
    (``FileNotFoundError``).  Postcondition: independent of ``output_dir`` and
    of the directories the inputs live in; changes if any input byte changes.
    """
    if not isinstance(request, SpineStudyRequest):
        raise TypeError("request must be a SpineStudyRequest")
    _require_inputs(request)
    return _fingerprint(request, _input_records(request, request.model_path))


# --------------------------------------------------------------------- misc


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[6]
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = done.stdout.strip()
    return (
        sha
        if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha)
        else ("unknown")
    )


def _admit(
    request: SpineStudyRequest, env: Mapping[str, str] | None
) -> tuple[SpineAssetReceipt | None, Path | None]:
    """Asset admission and the segmental model path to use (or ``None``)."""
    if request.asset_id is None:
        return None, request.model_path
    source = request.model_path
    if source is None:
        resolved = resolve_spine_asset(request.asset_id, env)
        source = None if resolved is None else resolved.source_path
    probe = source if source is not None else Path(f"{request.asset_id}.unset")
    receipt = admit_spine_asset(SpineAssetRequest(request.asset_id, probe))
    return receipt, (source if receipt.status is not SpineAssetStatus.MISSING else None)


def _judge(
    asset: SpineAssetReceipt | None, outcome: study_stages.StageOutcome
) -> tuple[str, tuple[str, ...]]:
    reasons = []
    if asset is not None and asset.status is not SpineAssetStatus.QUALIFIED:
        reasons.append(f"asset {asset.asset_id} is {asset.status.value}")
    if outcome.failed:
        return "failed", (*reasons, *outcome.reasons)
    reasons.extend(outcome.reasons)
    return ("unqualified" if reasons else "pass"), tuple(reasons)


def _variant_record(
    request: SpineStudyRequest, outcome: study_stages.StageOutcome
) -> dict[str, Any] | None:
    """Variant provenance: the outcome's record, else the request's description."""
    if request.variant_id is None:
        return None
    if outcome.variant is not None:
        return outcome.variant
    return study_variant.request_record(
        request.variant_id, request.variant_parameter, request.variant_handedness
    )


def _scrubber(request: SpineStudyRequest) -> Any:
    dirs = {request.output_dir}
    for path in (
        request.motion_path,
        request.golf_model_path,
        request.model_path,
        request.external_loads,
    ):
        if path is not None:
            dirs.update({path.parent, path.resolve().parent})
    texts = sorted((str(d) for d in dirs), key=len, reverse=True)
    texts.append(str(Path.home()))

    def scrub(text: str) -> str:
        for item in texts:
            if item not in ("", ".", "/"):
                text = text.replace(item, "<dir>")
        return text

    return scrub


def run_spine_study(
    request: SpineStudyRequest,
    *,
    code_version: str | None = None,
    env: Mapping[str, str] | None = None,
) -> SpineStudyReceipt:
    """Execute the pipeline for ``request`` and write ``<study_id>_receipt.json``.

    Args:
        request: the study definition.
        code_version: git SHA recorded in the receipt (injectable for tests);
            ``None`` reads ``git rev-parse HEAD`` and falls back to
            ``unknown``.
        env: environment used to resolve a licensed asset (default
            ``os.environ``).

    Returns:
        The receipt.  Postconditions: it holds basenames and hashes only; a
        licensed asset that is not QUALIFIED never gives ``pass``; a stage
        exception gives ``failed`` with the failing stage last; artifacts are
        listed with their SHA-256.

    Raises:
        TypeError: ``request`` is not a :class:`SpineStudyRequest`.
        FileNotFoundError: a named input file is absent.
    """
    if not isinstance(request, SpineStudyRequest):
        raise TypeError(
            f"request must be a SpineStudyRequest, got {type(request).__name__}"
        )
    _require_inputs(request)
    asset, model = _admit(request, env)
    records = _input_records(request, model)
    fingerprint = _fingerprint(request, records)
    out_dir = request.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if model is None:
        outcome = study_stages.StageOutcome.skipped("segmental model asset is absent")
    else:
        outcome = study_stages.run_stages(request, model, _scrubber(request))
    status, reasons = _judge(asset, outcome)
    receipt = SpineStudyReceipt(
        study_id=request.study_id,
        status=status,  # type: ignore[arg-type]
        status_reasons=reasons,
        request=request.canonical_dict(),
        request_sha256=fingerprint,
        inputs=tuple(records),
        code_version=_git_sha() if code_version is None else code_version,
        opensim_version=(
            "not-run" if not outcome.ran else str(import_opensim().GetVersionAndDate())
        ),
        scipy_version=scipy.__version__,
        numpy_version=np.__version__,
        law=outcome.law,
        distribution_source=outcome.distribution_source,
        asset=None if asset is None else json.loads(json.dumps(asset.as_dict())),
        stages=tuple(outcome.stages),
        artifacts=tuple(outcome.artifacts),
        residual=outcome.residual,
        reserve=outcome.reserve,
        context_comparisons=tuple(outcome.context),
        variant=_variant_record(request, outcome),
    )
    write_receipt(receipt, out_dir / f"{request.study_id}_receipt.json")
    ensure(
        receipt.status != "pass" or asset is None or asset.status.value == "qualified",
        "a non-qualified asset must never give pass",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: run the study of a JSON request file; exit 0 only on ``pass``."""
    parser = argparse.ArgumentParser(prog="spine.study", description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    raw = json.loads(args.request.read_text(encoding="utf-8"))
    request = SpineStudyRequest.from_dict(raw, args.request.resolve().parent)
    if args.out is not None:
        request = SpineStudyRequest.from_dict(
            {**raw, "output_dir": str(args.out.resolve())},
            args.request.resolve().parent,
        )
    receipt = run_spine_study(request, env=os.environ)
    sys.stdout.write(receipt_to_json(receipt))
    return 0 if receipt.status == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
