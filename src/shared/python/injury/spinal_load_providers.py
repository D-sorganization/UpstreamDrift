"""Spinal load providers: heuristic and model-based behind one interface (SPN-12).

``SpinalLoadAnalyzer`` historically computed lumbar loads from a closed-form
heuristic (body-weight share above the level, torque over a fixed 5 cm moment
arm, a centrifugal term). Those numbers are a screening estimate, not a joint
reaction. This module separates the two sources of per-level loads:

* :class:`HeuristicSpinalLoadProvider` wraps the unchanged heuristic and labels
  every output ``heuristic`` (provenance ``solver``, ``distribution_source``
  and ``provider_label``), so a heuristic series cannot be mistaken for a model
  load;
* :class:`ModelSpinalLoadProvider` wraps a precomputed
  :class:`~src.shared.python.spine.loads.SpinalLoadSeries` (for example the
  OpenSim pipeline output) and passes it through with provenance intact.

Both satisfy :class:`SpinalLoadProvider`. Sign convention is that of
``spine.loads`` (ADR-0055 item 5): force on the superior vertebra by the
inferior one, inferior ISB frame, compression ``+F_y``, AP shear ``+F_x``,
ML shear ``+F_z``. The heuristic risk thresholds apply to heuristic output
only; the analyzer does not apply them to model output.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.spine.levels import SpinalLevel, parse_level
from src.shared.python.spine.loads import (
    REQUIRED_PROVENANCE_KEYS,
    LoadObservable,
    SpinalLoadSeries,
)

__all__ = [
    "HEURISTIC_PROVIDER_ID",
    "HeuristicSpinalLoadProvider",
    "ModelSpinalLoadProvider",
    "SpinalLoadProvider",
    "model_spinal_load_provider_from_receipt",
]

#: Identifier of the heuristic provider; also surfaced by the analyzer result.
HEURISTIC_PROVIDER_ID = "heuristic_v1"

_TIME_MATCH_TOL_S = 1e-9
_ArrayDict = Mapping[str, np.ndarray]


@runtime_checkable
class SpinalLoadProvider(Protocol):
    """Source of per-level spinal loads.

    Attributes:
        provider_id: Stable identifier surfaced on analysis results.
        is_model_based: ``True`` only for loads from a musculoskeletal model.
    """

    provider_id: str
    is_model_based: bool

    def loads(
        self,
        *,
        time_s: np.ndarray,
        joint_angles: _ArrayDict | None = None,
        joint_velocities: _ArrayDict | None = None,
        joint_torques: _ArrayDict | None = None,
        levels: tuple[SpinalLevel, ...] | None = None,
    ) -> SpinalLoadSeries:
        """Return the per-level load series for the given time base."""
        ...


def _heuristic_provenance() -> dict[str, str]:
    provenance = dict.fromkeys(REQUIRED_PROVENANCE_KEYS, "n/a (heuristic)")
    provenance["solver"] = HEURISTIC_PROVIDER_ID
    provenance["cost_exponent"] = "n/a"
    provenance["distribution_source"] = (
        "heuristic closed-form estimate, not a distributed model load"
    )
    provenance["external_loads"] = "not modelled (heuristic)"
    provenance["provider_label"] = "heuristic"
    return provenance


class HeuristicSpinalLoadProvider:
    """The legacy analyzer heuristic exposed as a labelled load series.

    ``analyzer`` supplies ``lumbar_segments`` (level names) and
    ``compute_segment_loads``; the numbers are exactly the analyzer's. Torsion
    is reported as the axial moment ``M_y``.

    Raises (from :meth:`loads`): ``ValueError`` if an analyzed segment name is
    not a known :class:`SpinalLevel` or ``time_s`` is not a valid time base.
    """

    provider_id = HEURISTIC_PROVIDER_ID
    is_model_based = False

    def __init__(self, analyzer: Any) -> None:
        for attr in ("lumbar_segments", "compute_segment_loads"):
            if not hasattr(analyzer, attr):
                raise TypeError(f"analyzer must provide {attr!r}")
        self._analyzer = analyzer

    def loads(
        self,
        *,
        time_s: np.ndarray,
        joint_angles: _ArrayDict | None = None,
        joint_velocities: _ArrayDict | None = None,
        joint_torques: _ArrayDict | None = None,
        levels: tuple[SpinalLevel, ...] | None = None,
    ) -> SpinalLoadSeries:
        """Return the heuristic loads as a series labelled ``heuristic``."""
        time = np.asarray(time_s, dtype=float)
        angles = {} if joint_angles is None else dict(joint_angles)
        velocities = {} if joint_velocities is None else dict(joint_velocities)
        torques = {} if joint_torques is None else dict(joint_torques)
        names = list(self._analyzer.lumbar_segments)
        parsed = tuple(parse_level(name) for name in names)
        keep = [i for i, lv in enumerate(parsed) if levels is None or lv in levels]
        if not keep:
            raise ValueError("no analyzed segment matches the requested levels")
        force = np.zeros((time.size, len(keep), 3))
        moment = np.zeros_like(force)
        for out, i in enumerate(keep):
            seg = self._analyzer.compute_segment_loads(
                names[i], angles, velocities, torques, time
            )
            force[:, out, 0] = seg.ap_shear
            force[:, out, 1] = seg.compression
            force[:, out, 2] = seg.lateral_shear
            moment[:, out, 1] = seg.torsion
        return SpinalLoadSeries(
            time_s=time,
            levels=tuple(parsed[i] for i in keep),
            observable=LoadObservable.GENERALIZED_EFFORT,
            force_n=force,
            moment_nm=moment,
            provenance=_heuristic_provenance(),
        )


class ModelSpinalLoadProvider:
    """Pass-through of a precomputed model :class:`SpinalLoadSeries`.

    Preconditions: ``series`` is a ``SpinalLoadSeries``; ``provider_id`` is a
    non-empty string different from :data:`HEURISTIC_PROVIDER_ID`.

    Postcondition of :meth:`loads`: the returned series carries the original
    arrays and provenance unchanged (the same object when no level subset is
    requested). A requested time base must equal the series' time base and
    every requested level must be present, otherwise ``ValueError``.
    """

    is_model_based = True

    def __init__(
        self, series: SpinalLoadSeries, provider_id: str = "model_precomputed"
    ) -> None:
        if not isinstance(series, SpinalLoadSeries):
            raise TypeError("series must be a SpinalLoadSeries")
        if not isinstance(provider_id, str) or not provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")
        if provider_id == HEURISTIC_PROVIDER_ID:
            raise ValueError("provider_id must not reuse the heuristic provider id")
        self._series = series
        self.provider_id = provider_id

    def loads(
        self,
        *,
        time_s: np.ndarray,
        joint_angles: _ArrayDict | None = None,
        joint_velocities: _ArrayDict | None = None,
        joint_torques: _ArrayDict | None = None,
        levels: tuple[SpinalLevel, ...] | None = None,
    ) -> SpinalLoadSeries:
        """Return the stored series (optionally a level subset)."""
        time = np.asarray(time_s, dtype=float)
        stored = self._series.time_s
        if time.shape != stored.shape or not np.allclose(
            time, stored, rtol=0, atol=_TIME_MATCH_TOL_S
        ):
            raise ValueError(
                f"requested time base {time.shape} does not match the model "
                f"series time base {stored.shape}"
            )
        if levels is None:
            return self._series
        missing = [lv.name for lv in levels if lv not in self._series.levels]
        if missing:
            raise ValueError(f"levels not in the model series: {missing}")
        stored_levels = self._series.levels
        idx = [stored_levels.index(lv) for lv in levels]
        return SpinalLoadSeries(
            time_s=stored,
            levels=tuple(levels),
            observable=self._series.observable,
            force_n=self._series.force_n[:, idx, :],
            moment_nm=self._series.moment_nm[:, idx, :],
            provenance=dict(self._series.provenance),
            force_unit=self._series.force_unit,
        )

    @property
    def series(self) -> SpinalLoadSeries:
        """The underlying :class:`SpinalLoadSeries`."""
        return self._series

    @classmethod
    def from_study_receipt(
        cls,
        receipt: str | Path | Any,
        artifact_dir: str | Path | None = None,
        provider_id: str | None = None,
    ) -> ModelSpinalLoadProvider:
        """Construct a provider from a finished spine study receipt, run, or view.

        Args:
            receipt: A filesystem path (`str` or `Path`) pointing to a
                `*_receipt.json` file, a `VerifiedRun`, a `SpineStudyReceipt`,
                or a `SpineStudyView`.
            artifact_dir: Optional folder containing the loads `.npz` artifact
                (defaults to receipt directory).
            provider_id: Optional custom provider identifier.

        Returns:
            A :class:`ModelSpinalLoadProvider` configured with the verified series.

        Raises:
            TypeError: if receipt is not of a supported type.
            ValueError: if receipt path is empty or malformed.
        """
        return model_spinal_load_provider_from_receipt(
            receipt, artifact_dir=artifact_dir, provider_id=provider_id
        )


def _provenance_from_receipt(receipt: Any) -> dict[str, str]:
    request = getattr(receipt, "request", {}) or {}
    model_id = str(request.get("model_id") or "not recorded in receipt")
    model_sha = str(request.get("model_sha256") or "not recorded in receipt")
    dist = str(getattr(receipt, "distribution_source", "") or "not recorded in receipt")
    ext = str(request.get("external_loads") or "not recorded in receipt")
    solver = str(request.get("solver") or "JointReaction")
    return {
        "model_id": model_id,
        "model_sha256": model_sha,
        "solver": solver,
        "cost_exponent": "n/a",
        "distribution_source": dist,
        "external_loads": ext,
    }


def _series_from_run(run: Any) -> SpinalLoadSeries:
    receipt = getattr(run, "receipt", None)
    return SpinalLoadSeries(
        time_s=run.time_s,
        levels=run.levels,
        observable=LoadObservable.JOINT_REACTION,
        force_n=run.joint_reaction_force_n,
        moment_nm=run.joint_reaction_moment_nm,
        provenance=_provenance_from_receipt(receipt),
    )


def model_spinal_load_provider_from_receipt(
    receipt: str | Path | Any,
    artifact_dir: str | Path | None = None,
    provider_id: str | None = None,
) -> ModelSpinalLoadProvider:
    """Create a :class:`ModelSpinalLoadProvider` from a spine study receipt, run, or view.

    Preconditions:
        receipt: A filesystem path (`str` or `Path`) pointing to a
            `*_receipt.json` file, a `VerifiedRun`, a `SpineStudyReceipt`,
            or a `SpineStudyView`.
        artifact_dir: Optional folder containing the loads `.npz` artifact
            (defaults to receipt directory).
        provider_id: Optional custom provider identifier.

    Returns:
        A `ModelSpinalLoadProvider` wrapping the study's `SpinalLoadSeries`.

    Raises:
        TypeError: if receipt is None or an unsupported type.
        ValueError: if receipt path is empty or malformed.
    """
    if receipt is None:
        raise TypeError("receipt must not be None")

    if isinstance(receipt, (str, Path)):
        receipt_path = Path(receipt)
        if not str(receipt_path).strip():
            raise ValueError("receipt path must be non-empty")
        from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
            load_verified_run,
        )

        run = load_verified_run(receipt_path, artifact_dir=artifact_dir)
        series = _series_from_run(run)
        study_id = getattr(run.receipt, "study_id", "spine_study")
    elif hasattr(receipt, "series") and isinstance(receipt.series, SpinalLoadSeries):
        series = receipt.series
        study_id = getattr(receipt, "study_id", "spine_study")
    elif hasattr(receipt, "joint_reaction_force_n") and hasattr(receipt, "receipt"):
        series = _series_from_run(receipt)
        study_id = getattr(receipt.receipt, "study_id", "spine_study")
    else:
        raise TypeError(
            f"expected receipt path (str/Path), VerifiedRun, or SpineStudyView, "
            f"got {type(receipt).__name__}"
        )

    pid = provider_id or (
        f"spine_study_{study_id}"
        if study_id != HEURISTIC_PROVIDER_ID
        else "model_spine_study"
    )
    return ModelSpinalLoadProvider(series, provider_id=pid)
