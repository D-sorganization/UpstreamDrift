"""Silhouette segmentation model provenance, dev-only inference harness, and bounded benchmark (MMR-13, #11099).

Provides:
- ``SegmentationModelCard`` metadata (architecture, license, hardware budgets, failure modes).
- Checkpoint verification that fails closed against operator-supplied SHA-256 pins and
  rejects corrupt or arbitrary weight files without hidden network downloads.
- ``RealSegmentationAdapter``: a dev-only checkpoint-validation harness. It accepts real
  decoded frame pixels, lazily verifies and loads the pinned checkpoint through an
  optional neural runtime (torch or onnxruntime), and executes the checkpoint against
  those pixels when a calibrated body/club postprocess decoder is supplied. Without a
  runnable checkpoint or decode stage it raises :class:`SegmentationUnavailableError`;
  it never fabricates geometry and never reports successful inference from dimensions.
- ``evaluate_segmentation_benchmark``: scores adapter output against independently
  recorded gold label artifacts loaded from an evidence directory, and emits a typed
  blocked result (no numeric metrics) when that evidence or a runnable runtime is
  absent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Final

import numpy as np

from ._validation import (
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    check_id,
    check_sha256,
)
from .contracts import SegmentationRequest, SegmentationResult
from .ingestion import compute_frame_hash
from .mask_records import MaskFrame
from .segmentation import (
    ManualMaskProvider,
    compute_mask_dice,
    compute_mask_iou,
    track_occlusion_and_identity,
)
from .source_records import FrameIdentity

logger = logging.getLogger(__name__)

BENCHMARK_NAME: Final[str] = "shadow_tracker_silhouette_segmentation"

DEFAULT_DECODER_NAME: Final[str] = "unspecified"
DEFAULT_PIXEL_FORMAT: Final[str] = "rgb24"
_UNKNOWN_TIMING_REASON: Final[str] = "unknown_timing_not_fabricated"


class SegmentationUnavailableError(RuntimeError):
    """Real neural inference is not runnable in this configuration.

    Raised when the pinned checkpoint is missing, fails hash verification,
    cannot be loaded or executed by any available optional runtime, or when no
    calibrated body/club postprocess decoder is configured. Callers must fall
    back to :class:`ManualMaskProvider` reviewed annotations; synthetic masks
    and fabricated success counts are never emitted as a substitute.
    """


@dataclass(frozen=True, slots=True)
class SegmentationModelCard:
    """Model card recording architecture, licensing, and hardware budgets.

    ``checkpoint_sha256`` records the trusted weight pin for a model when the
    project has verified one. This repository ships none: pins are per-deployment
    operator configuration supplied to :func:`verify_checkpoint`, which rejects
    malformed and mismatched pins (fail closed).
    """

    model_name: str
    architecture: str
    version: str
    license: str
    checkpoint_sha256: str | None
    parameter_count_m: float
    input_resolution: tuple[int, int]
    hardware_requirements: dict[str, str]
    redistribution_terms: str
    known_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PINNED_MODELS: Final[dict[str, SegmentationModelCard]] = {
    "sam-vit-b-golf": SegmentationModelCard(
        model_name="sam-vit-b-golf",
        architecture="Segment Anything Model (ViT-B)",
        version="1.0.0",
        license="Apache-2.0",
        checkpoint_sha256=None,
        parameter_count_m=91.0,
        input_resolution=(1024, 1024),
        hardware_requirements={
            "min_ram_gb": "8",
            "min_vram_gb": "4",
            "gpu_recommended": "True",
            "cpu_fallback": "Supported",
        },
        redistribution_terms="Permitted under Apache-2.0; offline checkpoint required.",
        known_limitations=(
            "Motion blur in high-speed swing phases (>120 deg/frame) may diffuse clubhead boundaries",
            "Thin steel/graphite shafts (<3 pixels) require contrast guidance or manual correction",
            "Spectator, tree, or golf bag overlaps trigger partial occlusion flags",
        ),
    ),
    "mobilesam-golf": SegmentationModelCard(
        model_name="mobilesam-golf",
        architecture="MobileSAM (TinyViT)",
        version="1.0.0",
        license="Apache-2.0",
        checkpoint_sha256=None,
        parameter_count_m=9.66,
        input_resolution=(1024, 1024),
        hardware_requirements={
            "min_ram_gb": "4",
            "min_vram_gb": "2",
            "gpu_recommended": "False",
            "cpu_fallback": "Supported",
        },
        redistribution_terms="Permitted under Apache-2.0; offline checkpoint required.",
        known_limitations=(
            "Lower boundary precision on thin clubheads compared to ViT-B",
            "Extreme motion blur in historical archive clips requires contrast normalization",
        ),
    ),
}


def verify_checkpoint(
    model_name: str,
    checkpoint_path: Path | str,
    *,
    expected_sha256: str,
) -> SegmentationModelCard:
    """Validate a checkpoint against an operator-supplied SHA-256 pin (fail closed).

    The pin is REQUIRED per-deployment configuration; the repository ships no
    trusted placeholder hashes. ``expected_sha256`` must be exactly 64 lowercase
    hex characters. Missing checkpoints raise an actionable
    :class:`FileNotFoundError`; a computed hash that differs from the pin raises
    :class:`RuntimeError`. No hidden network downloads are initiated.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found for {model_name}: {path}. "
            "Hidden network downloads are disallowed by repository policy. "
            "Please install the verified weights offline per "
            "docs/development/matched_swing_program/evidence/segmentation/README.md "
            "or use ManualMaskProvider for reviewed annotations."
        )

    check_sha256(expected_sha256, "expected_sha256")

    model_card = PINNED_MODELS.get(model_name)
    if model_card is None:
        raise ValueError(
            f"Unknown model_name {model_name!r}. Registered models: {sorted(PINNED_MODELS)}"
        )

    computed_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if computed_sha != expected_sha256:
        raise RuntimeError(
            f"not a valid model checkpoint: hash mismatch for {model_name!r}. "
            f"Expected operator pin {expected_sha256}, got {computed_sha}. "
            "Arbitrary or corrupt checkpoints cannot pass segmentation validation."
        )
    return model_card


def _load_frame_pixels(pixels: np.ndarray) -> np.ndarray:
    """Validate a decoded frame pixel array for hashing and inference."""
    if not isinstance(pixels, np.ndarray):
        raise TypeError(f"pixels must be a numpy.ndarray, got {type(pixels).__name__}")
    if pixels.dtype != np.dtype(np.uint8):
        raise ValueError(f"pixels must be uint8 decoded pixels, got {pixels.dtype}")
    if pixels.ndim != 3 or pixels.shape[2] not in (1, 3):
        raise ValueError(
            f"pixels must have shape (height, width, 1|3), got {pixels.shape}"
        )
    if pixels.shape[0] <= 0 or pixels.shape[1] <= 0:
        raise ValueError(f"pixels must be non-empty, got {pixels.shape}")
    return np.ascontiguousarray(pixels)


def build_frame_identity(
    frame_meta: Mapping[str, Any],
    pixels: np.ndarray,
    *,
    decoder_name: str = DEFAULT_DECODER_NAME,
    pixel_format: str = DEFAULT_PIXEL_FORMAT,
    asset_id: str = "asset-unspecified",
    shot_id: str = "shot-unspecified",
    swing_id: str = "swing-unspecified",
    camera_id: str = "camera-unspecified",
) -> FrameIdentity:
    """Build a :class:`FrameIdentity` from decoded pixels and caller-supplied timing.

    ``frame_sha256`` is always the SHA-256 of the decoded pixel payload bound to
    the decoder/pixel-format provenance (never a hash of the frame ID string).
    Physical timing is recorded only when the caller supplies both
    ``physical_time_s`` (finite float) and its ``physical_time_reason``; missing
    timing stays ``None`` with an explicit unknown-time reason. Ticks and
    timebase are REQUIRED and never fabricated.
    """
    _load_frame_pixels(pixels)
    if not isinstance(frame_meta, Mapping):
        raise TypeError(
            f"frame_meta must be a mapping, got {type(frame_meta).__name__}"
        )

    frame_id = check_id(frame_meta.get("frame_id"), "frame_id")
    pts_ticks = frame_meta.get("pts_ticks")
    if pts_ticks is None:
        raise ValueError(
            "pts_ticks is required for every segmented frame; timing is never fabricated"
        )
    if isinstance(pts_ticks, bool) or not isinstance(pts_ticks, int):
        raise TypeError(f"pts_ticks must be an int, got {type(pts_ticks).__name__}")

    numerator = frame_meta.get("timebase_numerator")
    denominator = frame_meta.get("timebase_denominator")
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or numerator <= 0
        or denominator <= 0
    ):
        raise ValueError(
            "timebase_numerator and timebase_denominator are required "
            "(positive reduced fraction); timing units are never invented"
        )

    physical_time: float | None = None
    physical_time_reason = ""
    supplied_time = frame_meta.get("physical_time_s")
    if supplied_time is not None:
        if isinstance(supplied_time, bool) or not isinstance(supplied_time, float):
            raise TypeError(
                f"physical_time_s must be float, got {type(supplied_time).__name__}"
            )
        supplied_reason = frame_meta.get("physical_time_reason")
        if (
            not isinstance(supplied_reason, str)
            or not supplied_reason.strip()
            or supplied_reason.strip() != supplied_reason
        ):
            raise ValueError(
                "physical_time_reason must be supplied, non-empty, and trimmed "
                "whenever physical_time_s is recorded"
            )
        physical_time = supplied_time
        physical_time_reason = supplied_reason
    if not physical_time_reason:
        physical_time_reason = _UNKNOWN_TIMING_REASON

    pixels_digest = compute_frame_hash(
        pixels.tobytes(), decoder_name=decoder_name, pixel_format=pixel_format
    )
    return FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id=asset_id,
        shot_id=shot_id,
        swing_id=swing_id,
        camera_id=camera_id,
        frame_id=frame_id,
        pts_ticks=pts_ticks,
        timebase_numerator=numerator,
        timebase_denominator=denominator,
        physical_time_s=physical_time,
        physical_time_reason=physical_time_reason,
        frame_sha256=pixels_digest,
        timing_mode=frame_meta.get("timing_mode", "container_pts"),
        is_timing_exact=bool(frame_meta.get("is_timing_exact", False)),
        clock_evidence=frame_meta.get("clock_evidence", "unverified_legacy_record"),
        decoder_name=decoder_name,
        pixel_format=pixel_format,
    )


def derive_revision_id(
    model_name: str,
    *,
    expected_sha256: str,
    frame: FrameIdentity,
    body: bytes,
    club: bytes,
    valid: bytes,
    config: Mapping[str, Any],
) -> str:
    """Derive a content-addressed revision ID for a generated mask frame.

    The ID binds the model name, checkpoint pin, full frame scope, the source
    frame's decoded-pixel content hash, the complete inference configuration
    (dimensions, conditions, options), and the generated mask bytes, so
    different outputs for the same frame never collide-conflict in
    :class:`ManualMaskProvider`.
    """
    config_payload = json.dumps(
        dict(config), sort_keys=True, separators=(",", ":"), default=str
    )
    scope_payload = json.dumps(
        [
            frame.asset_id,
            frame.shot_id,
            frame.swing_id,
            frame.camera_id,
            frame.frame_id,
            frame.pts_ticks,
            frame.frame_sha256,
        ],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        b"rev\x00"
        + model_name.encode("utf-8")
        + b"\x00"
        + expected_sha256.encode("ascii")
        + b"\x00"
        + config_payload.encode("utf-8")
        + b"\x00"
        + scope_payload.encode("utf-8")
        + b"\x00"
        + body
        + club
        + valid
    ).hexdigest()[:16]
    return f"rev-{model_name}-{frame.frame_id}-{digest}"


class _CheckpointExecutor:
    """Executed pinned checkpoint bound to a concrete neural runtime."""

    def __init__(self, runtime_label: str, callable_model: Callable[[np.ndarray], Any]):
        self.runtime_label = runtime_label
        self._callable_model = callable_model

    def run(self, pixels: np.ndarray) -> np.ndarray:
        return np.asarray(self._callable_model(pixels))


def _torch_forward(pixels: np.ndarray, model: Any) -> Any:
    """Forward a (1, C, H, W) float tensor through the executed torch model."""
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).float() / 255.0
    if tensor.ndim == 3 and tensor.shape[2] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
    if isinstance(output, torch.Tensor):
        return output.detach().cpu().numpy()
    return output


def _onnx_forward(pixels: np.ndarray, session: Any, input_name: str) -> Any:
    height, width = pixels.shape[:2]
    batch = (
        np.ascontiguousarray(pixels.transpose(2, 0, 1))[None, ...].astype(np.float32)
        / 255.0
    )
    try:
        output = session.run(None, {input_name: batch})[0]
    except Exception as exc:
        raise SegmentationUnavailableError(
            f"Model forward pass failed under onnxruntime for a "
            f"{height}x{width} frame: {exc}. The adapter never substitutes "
            "synthetic masks for a failed run (checkpoint input size or "
            "dynamic axis configuration is the usual cause)."
        ) from exc
    return output


def _load_executable_checkpoint(
    model_name: str,
    path: Path,
    postprocess: Callable[..., Any] | None,
) -> _CheckpointExecutor:
    """Lazily import a neural runtime and load the checkpoint as an executor."""
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        ort = None
    try:
        import torch as torch_mod  # type: ignore[import-not-found]
    except ImportError:
        torch_mod = None

    if ort is None and torch_mod is None:
        raise SegmentationUnavailableError(
            f"No neural runtime (torch or onnxruntime) is installed for "
            f"{model_name!r}; the pinned checkpoint cannot be executed. "
            "RealSegmentationAdapter never synthesizes masks from dimensions. "
            "Provision an offline runtime, or use ManualMaskProvider."
        )

    runtime = (
        ort
        if (ort is not None and (path.suffix.lower() == ".onnx" or torch_mod is None))
        else None
    )

    if runtime is not None:
        try:
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
        except Exception as exc:
            raise SegmentationUnavailableError(
                f"Checkpoint {path} could not be loaded by onnxruntime: {exc}"
            ) from exc
        if postprocess is None:
            raise SegmentationUnavailableError(_NO_POSTPROCESS_MESSAGE)
        return _CheckpointExecutor(
            "onnxruntime", lambda px: _onnx_forward(px, session, input_name)
        )

    return _load_torch_executor(model_name, path, postprocess, torch_mod)


_NO_POSTPROCESS_MESSAGE = (
    "no body/club postprocess decoder is configured for this model; the "
    "checkpoint may be verified and loadable, but its raw output cannot be "
    "honestly converted into body/club masks. RealSegmentationAdapter is a "
    "dev-only checkpoint-validation harness; register a calibrated postprocess "
    "callable (operator-supplied decode stage) or use ManualMaskProvider."
)


def _load_torch_executor(
    model_name: str,
    path: Path,
    postprocess: Callable[..., Any] | None,
    torch_mod: Any,
) -> _CheckpointExecutor:
    if postprocess is None:
        raise SegmentationUnavailableError(_NO_POSTPROCESS_MESSAGE)
    try:
        model = torch_mod.jit.load(str(path), map_location="cpu")
    except Exception:
        try:
            model = torch_mod.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SegmentationUnavailableError(
                f"Checkpoint for {model_name!r} could not be loaded by torch: {exc}"
            ) from exc
    if isinstance(model, torch_mod.nn.Module):
        model = model.float().cpu().eval()
        return _CheckpointExecutor("torch", lambda px: _torch_forward(px, model))
    raise SegmentationUnavailableError(
        f"Checkpoint for {model_name!r} loaded as a weights-only object "
        f"({type(model).__name__}): weights-only state dicts cannot be executed "
        "without model architecture code. Supply an executable TorchScript or "
        "ONNX export of the pinned weights."
    )


class RealSegmentationAdapter:
    """Dev-only checkpoint-validation harness for pinned silhouette models.

    Contract: every requested frame must arrive as real decoded pixels; the
    pinned checkpoint is lazily verified (SHA-256 against the operator pin),
    loaded, and executed under an optional runtime (torch or onnxruntime), and
    masks are decoded from the executed output only when a calibrated
    body/club postprocess decoder is supplied. Without any of these the adapter
    raises :class:`SegmentationUnavailableError`, and no mask is registered:
    it never reports successful model inference from dimensions alone.
    """

    __slots__ = (
        "_model_name",
        "_checkpoint_path",
        "_expected_sha256",
        "_decoder_name",
        "_pixel_format",
        "_postprocess",
        "_manual_provider",
        "_card",
        "_executor",
    )

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Path | str,
        *,
        expected_sha256: str,
        manual_provider: ManualMaskProvider | None = None,
        decoder_name: str = DEFAULT_DECODER_NAME,
        pixel_format: str = DEFAULT_PIXEL_FORMAT,
        postprocess: Callable[..., Any] | None = None,
    ) -> None:
        if model_name not in PINNED_MODELS:
            raise ValueError(
                f"Unknown model_name {model_name!r}. "
                f"Registered models: {sorted(PINNED_MODELS)}"
            )
        check_sha256(expected_sha256, "expected_sha256")
        check_id(decoder_name, "decoder_name")
        self._model_name = model_name
        self._checkpoint_path = Path(checkpoint_path)
        self._expected_sha256 = expected_sha256
        self._decoder_name = decoder_name
        self._pixel_format = pixel_format
        self._postprocess = postprocess
        self._card = PINNED_MODELS[model_name]
        self._executor: _CheckpointExecutor | None = None
        if not self._checkpoint_path.is_file():
            raise SegmentationUnavailableError(
                f"Model checkpoint not found for {model_name}: "
                f"{self._checkpoint_path}. Segmentation cannot run without "
                "operator-provisioned weights pinned by SHA-256 (no hidden "
                "downloads); use ManualMaskProvider for reviewed annotations."
            )
        self._manual_provider = manual_provider or ManualMaskProvider()

    @property
    def model_card(self) -> SegmentationModelCard:
        return self._card

    @property
    def manual_provider(self) -> ManualMaskProvider:
        return self._manual_provider

    def _ensure_executor(self) -> _CheckpointExecutor:
        if self._executor is None:
            try:
                self._card = verify_checkpoint(
                    self._model_name,
                    self._checkpoint_path,
                    expected_sha256=self._expected_sha256,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                raise SegmentationUnavailableError(
                    f"Checkpoint for {self._model_name!r} failed pinned-hash "
                    f"verification: {exc}"
                ) from exc
            if (
                self._postprocess is None
            ):  # decode stage is mandatory before any model executes
                raise SegmentationUnavailableError(_NO_POSTPROCESS_MESSAGE)
            self._executor = _load_executable_checkpoint(
                self._model_name,
                self._checkpoint_path,
                self._postprocess,
            )
        return self._executor

    def infer_frame(
        self,
        frame: FrameIdentity,
        pixels: np.ndarray,
        *,
        adverse_conditions: list[str] | None = None,
    ) -> MaskFrame:
        """Run the executed checkpoint on decoded pixels, or raise typed.

        The frame identity must carry the same pixel content hash as the
        decoded pixels; masks are never synthesized from dimensions or
        condition labels.
        """
        if not isinstance(frame, FrameIdentity):
            raise TypeError(f"Expected FrameIdentity, got {type(frame).__name__}")
        pixels = _load_frame_pixels(pixels)
        height, width = pixels.shape[:2]
        expected_hash = compute_frame_hash(
            pixels.tobytes(),
            decoder_name=self._decoder_name,
            pixel_format=self._pixel_format,
        )
        if frame.frame_sha256 != expected_hash:
            raise ValueError(
                f"frame_sha256 {frame.frame_sha256!r} does not match the decoded "
                f"pixel payload hash for frame {frame.frame_id!r}"
            )

        executor = self._ensure_executor()
        conditions = sorted(set(adverse_conditions or []))
        config: dict[str, Any] = {
            "model": self._model_name,
            "checkpoint_sha8": self._expected_sha256[:8],
            "width_px": width,
            "height_px": height,
            "adverse_conditions": conditions,
            "decoder_name": self._decoder_name,
            "pixel_format": self._pixel_format,
            "executor": executor.runtime_label,
        }

        try:
            raw_output = executor.run(pixels)
        except SegmentationUnavailableError:
            raise

        body_np, club_np, valid_np = self._decode_masks(raw_output, pixels)
        body = bytes(body_np)
        club = bytes(club_np)
        valid = bytes(valid_np)
        revision_id = derive_revision_id(
            self._model_name,
            expected_sha256=self._expected_sha256,
            frame=frame,
            body=body,
            club=club,
            valid=valid,
            config=config,
        )
        mask_frame = MaskFrame(
            schema_version=MASK_SCHEMA_VERSION,
            frame=frame,
            width_px=width,
            height_px=height,
            body=body,
            club=club,
            valid=valid,
            revision_id=revision_id,
            parent_revision_id=None,
            producer_id=f"model:{self._model_name}:{self._expected_sha256[:8]}",
            correction_note="Unreviewed draft from dev-only checkpoint-validation harness",
        )
        self._manual_provider.register_mask(mask_frame)
        return mask_frame

    def _decode_masks(
        self, raw_output: np.ndarray, pixels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode the executed output into flat body/club/valid channels."""
        if self._postprocess is None:
            raise SegmentationUnavailableError(
                f"{_NO_POSTPROCESS_MESSAGE[0].upper()}{_NO_POSTPROCESS_MESSAGE[1:]}"
            )
        decoded = self._postprocess(np.asarray(raw_output), pixels)
        if not isinstance(decoded, tuple) or len(decoded) != 3:
            raise ValueError(
                "postprocess must return (body, club, valid) uint8 numpy arrays"
            )
        height, width = pixels.shape[:2]
        size = width * height
        channels: list[np.ndarray] = []
        for channel_name, arr in zip(("body", "club", "valid"), decoded, strict=True):
            if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
                raise ValueError(
                    f"postprocess {channel_name} channel must be a uint8 numpy array"
                )
            flat = np.ascontiguousarray(arr).reshape(-1)
            if flat.size != size or frozenset(np.unique(flat)) - {0, 1}:
                raise ValueError(
                    f"postprocess {channel_name} channel must contain exactly "
                    f"{width * height} 0/1 values, got shape {arr.shape}"
                )
            channels.append(flat)
        body_np, club_np, valid_np = channels
        return (
            (body_np * valid_np).astype(np.uint8),
            (club_np * valid_np).astype(np.uint8),
            valid_np.astype(np.uint8),
        )

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        """Fulfill a :class:`SegmentationRequest` with executed model inference.

        Requires a ``frames`` mapping in ``request.options`` carrying decoded
        pixels plus caller-supplied PTS/timebase (and any real physical times
        and their recorded reasons). Missing pixels, missing checkpoints, or an
        unavailable runtime raise :class:`SegmentationUnavailableError` instead
        of reporting success.
        """
        if not isinstance(request, SegmentationRequest):
            raise TypeError(
                f"Expected SegmentationRequest, got {type(request).__name__}"
            )

        frames = request.options.get("frames")
        if not isinstance(frames, Mapping):
            raise SegmentationUnavailableError(
                "SegmentationRequest must provide per-frame decoded pixels under "
                "options['frames']; got "
                f"{type(frames).__name__ if frames is not None else 'nothing'}. "
                "Masks are never synthesized from dimensions or condition labels."
            )

        adverse = request.options.get("adverse_conditions")
        decoder_name = request.options.get("decoder_name", self._decoder_name)
        pixel_format = request.options.get("pixel_format", self._pixel_format)
        asset_id = request.options.get("asset_id", "asset-unspecified")
        swing_id = request.options.get("swing_id", "swing-unspecified")
        camera_id = request.options.get("camera_id", "camera-unspecified")

        count = 0
        for frame_id in request.frame_ids:
            entry = frames.get(frame_id)
            if not isinstance(entry, Mapping):
                raise SegmentationUnavailableError(
                    f"No real pixels provided for frame {frame_id!r}; provide "
                    "options['frames'][frame_id] with decoded pixels and timing."
                )
            meta = dict(entry)
            meta.setdefault("frame_id", frame_id)
            pixels = meta.pop("pixels", None)
            if pixels is None:
                raise SegmentationUnavailableError(
                    f"Frame {frame_id!r} carries no 'pixels' payload; inference "
                    "would be fabricated from dimensions and is refused."
                )
            identity = build_frame_identity(
                meta,
                pixels,
                decoder_name=decoder_name,
                pixel_format=pixel_format,
                asset_id=asset_id,
                shot_id=request.shot_id,
                swing_id=swing_id,
                camera_id=camera_id,
            )
            self.infer_frame(identity, pixels, adverse_conditions=adverse)
            count += 1

        return SegmentationResult(
            shot_id=request.shot_id,
            mask_count=count,
            provenance=f"model_inference:{self._model_name}@{self._expected_sha256[:8]}",
        )


# ---------------------------------------------------------------------------
# Bounded benchmark scoring against independently recorded gold artifacts
# ---------------------------------------------------------------------------

GoldLabel = tuple[bytes, bytes, bytes]
"""Gold (body, club, valid) byte masks decoded from a recorded label image."""


def _clip_manifest(evidence_dir: Path, manifest_name: str) -> list[dict[str, Any]]:
    """Parse clip_manifest.json; raises KeyError when absent (typed blocker)."""
    path = evidence_dir / manifest_name
    if not path.is_file():
        raise KeyError(manifest_name)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_name} is not valid JSON: {exc}") from exc
    clips = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(clips, list):
        raise ValueError(f"{manifest_name} must define a clips list")
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get("clip_id"), str):
            raise ValueError(f"{manifest_name} clips entries need a clip_id")
    return clips


def _gold_label_arrays(label: np.ndarray) -> tuple[bytes, bytes, bytes]:
    """Decode a recorded gold label image (0 bg, 1 body, 2 club, 3 invalid)."""
    if not isinstance(label, np.ndarray) or label.dtype != np.dtype(np.uint8):
        raise ValueError("gold label file must be a uint8 .npy label array")
    valid_np = np.where(label == 3, np.uint8(0), np.uint8(1))
    body_np = np.where(label == 1, np.uint8(1), np.uint8(0)) * valid_np
    club_np = np.where(label == 2, np.uint8(1), np.uint8(0)) * valid_np
    return (
        bytes(body_np.reshape(-1)),
        bytes(club_np.reshape(-1)),
        bytes(valid_np.reshape(-1)),
    )


def _clip_dimensions(clip: Mapping[str, Any]) -> tuple[int, int]:
    width, height = clip.get("width_px"), clip.get("height_px")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"clip {clip.get('clip_id')!r} must record width_px/height_px")
    return width, height


def _load_gold_for_clip(
    evidence_dir: Path, clip: Mapping[str, Any]
) -> list[tuple[Mapping[str, Any], GoldLabel]]:
    """Load and verify every recorded gold mask row for one clip.

    Returns (manifest_row, (gold_body, gold_club, gold_valid)) tuples. Raises
    KeyError for absent artifacts and ValueError for hash or label failures
    (fail closed: recorded evidence is never replaced).
    """
    gold_rows = clip.get("gold_masks")
    if not isinstance(gold_rows, list) or not gold_rows:
        raise KeyError("independent_gold_masks")
    verified: list[tuple[Mapping[str, Any], GoldLabel]] = []
    for row in gold_rows:
        if not isinstance(row, Mapping):
            raise ValueError("gold_masks entries must be mappings")
        label_path = evidence_dir / str(row.get("file", ""))
        if not label_path.is_file():
            raise KeyError(str(label_path.name))
        payload = label_path.read_bytes()
        expected_sha = row.get("sha256")
        if not isinstance(expected_sha, str) or (
            hashlib.sha256(payload).hexdigest() != expected_sha
        ):
            raise ValueError(f"gold mask {label_path.name} hash mismatch")
        label = np.load(label_path)
        verified.append((row, _gold_label_arrays(label)))
    return verified


def _blocked_report(
    adapter: Any,
    *,
    reason_code: str,
    missing: list[str],
) -> dict[str, Any]:
    card = getattr(adapter, "model_card", None)
    if card is not None and getattr(card, "checkpoint_sha256", "present") is None:
        missing.append("pinned_checkpoint_weights")
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "model": _model_summary(adapter),
        "status": "blocked",
        "reason_code": reason_code,
        "missing_evidence": sorted(set(missing)),
        "metrics_reported": False,
        "clips": [],
    }


def _model_summary(adapter: Any) -> dict[str, Any]:
    """Render the adapter's model card as a JSON-ready dict (never fake pins)."""
    card = getattr(adapter, "model_card", None)
    if card is None:
        return {}
    if hasattr(card, "to_dict"):
        return card.to_dict()
    try:
        return asdict(card)
    except TypeError:
        return dict(getattr(card, "__dict__", {}))


def _clip_skip(clip: Mapping[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "clip_id": clip.get("clip_id"),
        "clip_type": clip.get("clip_type"),
        "status": status,
        "reason": reason,
    }


def _evaluate_clip(
    adapter: Any,
    clip: Mapping[str, Any],
    gold_rows: list[tuple[Mapping[str, Any], GoldLabel]],
    frames: list[tuple[np.ndarray, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Run the adapter over a clip with evidence and score against gold."""
    clip_id = str(clip["clip_id"])
    conditions = clip.get("adverse_conditions") or []
    latencies_ms: list[float] = []
    body_ious: list[float] = []
    club_recalls: list[float] = []
    boundary_f1s: list[float] = []
    correction_edits = 0
    occlusion_any = False

    tracemalloc.start()
    try:
        for index, (pixels, meta) in enumerate(frames):
            row, (gold_body, gold_club, gold_valid) = gold_rows[index]
            meta = dict(meta)
            meta.setdefault("frame_id", row.get("frame_id"))
            meta.setdefault("pts_ticks", row.get("pts_ticks"))
            meta.setdefault("timebase_numerator", row.get("timebase_numerator"))
            meta.setdefault("timebase_denominator", row.get("timebase_denominator"))
            physical_time = row.get("physical_time_s")
            if physical_time is not None:
                meta.setdefault("physical_time_s", float(physical_time))
                meta.setdefault(
                    "physical_time_reason",
                    row.get("physical_time_reason", "clip_manifest"),
                )
            start = time.perf_counter()
            inferred = adapter.infer_frame(
                _build_identity(
                    meta,
                    pixels,
                    asset_id=str(clip.get("asset_id", "asset-unspecified")),
                    shot_id=str(clip.get("shot_id", f"shot-{clip_id}")),
                    swing_id=str(clip.get("swing_id", "swing-unspecified")),
                    camera_id=str(clip.get("camera_id", "camera-unspecified")),
                ),
                pixels,
                adverse_conditions=list(conditions),
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            valid = _combine_valid(inferred.valid, gold_valid)
            body_ious.append(compute_mask_iou(inferred.body, gold_body, valid))
            boundary_f1s.append(compute_mask_dice(inferred.club, gold_club, valid))
            club_recalls.append(_club_recall(inferred.club, gold_club, valid))
            correction_edits += _correction_edits(
                inferred.body, gold_body, inferred.club, gold_club, valid
            )
            occlusion_any = occlusion_any or _occlusion_detected(inferred, gold_body)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    metrics = {
        "clip_id": clip_id,
        "clip_type": str(clip.get("clip_type", "unspecified")),
        "status": "evaluated",
        "body_iou": round(float(np.mean(body_ious)), 4),
        "club_recall": round(float(np.mean(club_recalls)), 4),
        "boundary_f1": round(float(np.mean(boundary_f1s)), 4),
        "correction_effort_edits": int(correction_edits),
        "latency_ms_per_frame": round(float(np.mean(latencies_ms)), 2),
        "peak_memory_mb": round(peak_bytes / (1024.0 * 1024.0), 2),
        "occlusion_detected": bool(occlusion_any),
        "frames_evaluated": len(frames),
    }
    return metrics


def _build_identity(
    meta: Mapping[str, Any],
    pixels: np.ndarray,
    *,
    asset_id: str,
    shot_id: str,
    swing_id: str,
    camera_id: str,
) -> FrameIdentity:
    return build_frame_identity(
        meta,
        pixels,
        asset_id=asset_id,
        shot_id=shot_id,
        swing_id=swing_id,
        camera_id=camera_id,
    )


def _club_recall(club: bytes, gold_club: bytes, valid: bytes) -> float:
    pred = np.frombuffer(club, dtype=np.uint8)
    gold = np.frombuffer(gold_club, dtype=np.uint8)
    val = np.frombuffer(valid, dtype=np.uint8)
    gold_pos = int(np.sum((gold == 1) & (val == 1)))
    if gold_pos == 0:
        return 1.0
    true_pos = int(np.sum((pred == 1) & (gold == 1) & (val == 1)))
    return float(true_pos / gold_pos)


def _correction_edits(
    body: bytes, gold_body: bytes, club: bytes, gold_club: bytes, valid: bytes
) -> int:
    val = np.frombuffer(valid, dtype=np.uint8)
    diffs = np.frombuffer(body, dtype=np.uint8) != np.frombuffer(
        gold_body, dtype=np.uint8
    )
    diffs |= np.frombuffer(club, dtype=np.uint8) != np.frombuffer(
        gold_club, dtype=np.uint8
    )
    return int(np.sum((diffs) & (val == 1)))


def _occlusion_detected(inferred: MaskFrame, gold_body: bytes) -> bool:
    expected_body_area_px = max(1, int(gold_body.count(1)))
    report = track_occlusion_and_identity(
        inferred, expected_body_area_px=expected_body_area_px
    )
    return bool(report.is_partially_occluded or report.is_identity_lost)


def _combine_valid(pred_valid: bytes, gold_valid: bytes) -> bytes:
    """Intersect predicted and recorded-gold valid regions for honest scoring."""
    pred = np.frombuffer(pred_valid, dtype=np.uint8)
    gold = np.frombuffer(gold_valid, dtype=np.uint8)
    if pred.size != gold.size:
        raise ValueError("predicted and gold valid masks must share the same grid")
    return bytes(np.minimum(pred, gold).astype(np.uint8))


def evaluate_segmentation_benchmark(
    adapter: Any,
    *,
    evidence_dir: Path | str,
    frame_source: Callable[[int], tuple[np.ndarray, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score adapter output against independently recorded gold evidence.

    Gold masks are loaded from ``evidence_dir``: a ``clip_manifest.json``
    describing the held-out clips plus recorded gold label artifacts (``.npy``
    label images; 0=background, 1=body, 2=club, 3=invalid/occluded, each bound
    by SHA-256 in the manifest) and a caller-provided ``frame_source(index)``
    returning the decoded pixels plus frame metadata for each gold frame. The
    adapter under test is never used to generate gold. When the manifest, gold
    artifacts, a runnable runtime, or decoded clips are absent, the result is a
    typed insufficiency report with ``status='blocked'`` and no numeric
    metrics.
    """
    evidence = Path(evidence_dir)
    try:
        clips = _clip_manifest(evidence, "clip_manifest.json")
    except KeyError:
        return _blocked_report(
            adapter,
            reason_code="missing_evidence",
            missing=["clip_manifest.json", "independent_gold_masks"],
        )

    clip_results: list[dict[str, Any]] = []
    missing: set[str] = set()
    dominant: list[str] = []
    evaluated = 0
    for clip in clips:
        clip_id = clip.get("clip_id", "<unnamed>")
        try:
            gold_rows = _load_gold_for_clip(evidence, clip)
            frames = _frame_inputs(clip, gold_rows, frame_source)
        except KeyError as exc:
            missing.update(str(exc.args[0]))
            clip_results.append(
                _clip_skip(
                    clip,
                    "skipped_missing_evidence",
                    str(exc.args[0]) if exc.args else "absent",
                )
            )
            dominant.append("missing_evidence")
            continue
        except (ValueError, OSError) as exc:
            clip_results.append(_clip_skip(clip, "invalid_evidence", str(exc)))
            dominant.append("invalid_evidence")
            continue

        try:
            clip_results.append(_evaluate_clip(adapter, clip, gold_rows, frames))
            evaluated += 1
        except SegmentationUnavailableError as exc:
            missing.add("runnable_model_runtime")
            clip_results.append(_clip_skip(clip, "model_unavailable", str(exc)))
            dominant.append("model_unavailable")
        except (ValueError, OSError, KeyError) as exc:
            clip_results.append(_clip_skip(clip, "invalid_evidence", str(exc)))
            dominant.append("invalid_evidence")

    metrics_reported = evaluated > 0
    if metrics_reported and evaluated == len(clips):
        status, reason_code = "ok", "none"
    elif metrics_reported:
        status, reason_code = "partial", dominant[0] if dominant else "missing_evidence"
    else:
        status = "blocked"
        reason_code = dominant[0] if dominant else "missing_evidence"

    card = getattr(adapter, "model_card", None)
    if card is not None and getattr(card, "checkpoint_sha256", "present") is None:
        missing.add("pinned_checkpoint_weights")

    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "model": _model_summary(adapter),
        "status": status,
        "reason_code": reason_code,
        "missing_evidence": sorted(missing),
        "metrics_reported": metrics_reported,
        "clips": clip_results,
    }


def _frame_inputs(
    clip: Mapping[str, Any],
    gold_rows: list[tuple[Mapping[str, Any], GoldLabel]],
    frame_source: Callable[[int], tuple[np.ndarray, Mapping[str, Any]]] | None,
) -> list[tuple[np.ndarray, Mapping[str, Any]]]:
    """Fetch decoded pixels for every gold frame dimension-verified."""
    if frame_source is None:
        raise KeyError("decoded_clip_pixels")
    frames: list[tuple[np.ndarray, Mapping[str, Any]]] = []
    for index in range(len(gold_rows)):
        pixels, meta = frame_source(index)
        pixels = _load_frame_pixels(pixels)
        width, height = _clip_dimensions(clip)
        if pixels.shape[0] != height or pixels.shape[1] != width:
            raise ValueError(
                f"frame_source pixels {pixels.shape} do not match recorded clip "
                f"grid {width}x{height} for {clip.get('clip_id')!r}"
            )
        frames.append((pixels, meta))
    return frames
