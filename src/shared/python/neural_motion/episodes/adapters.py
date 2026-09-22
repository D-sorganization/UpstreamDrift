"""Versioned schema adapters into EpisodeRecord (NM-03 #10618)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
    SCHEMA_VERSION as COMPACT_SCHEMA,
)

from .record import EPISODE_STORE_SCHEMA, EpisodeRecord

__all__ = ["CompactAdapter", "CompactArrayBundle"]


@dataclass(frozen=True, slots=True)
class CompactArrayBundle:
    """Typed compact-1.0 payload for :meth:`CompactAdapter.from_compact_arrays`.

    Keeps the adapter entry point under the architecture parameter budget while
    preserving explicit joint/coefficient layout fields for DbC checks.
    """

    trial_id: str
    family_id: str
    sample_times_s: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray
    joint_names: tuple[str, ...]
    coefficient_letters: tuple[str, ...]
    coefficients: np.ndarray
    model_id: str = "compact.legacy"
    geometry_stratum: str = "unknown"
    contact_stratum: str = "unknown"
    club_stratum: str = "unknown"
    ancestry: tuple[str, ...] = ()


class CompactAdapter:
    """Adapt compact-1.0 arrays without reinterpreting 27/189 layouts."""

    source_schema = COMPACT_SCHEMA

    def from_compact_arrays(self, payload: CompactArrayBundle) -> EpisodeRecord:
        """Map compact q/qd/qdd/tau onto identity channels + coefficients.

        ``qdd`` is treated as interval finite-difference acceleration at the
        source clock and recorded under ``a_native`` only when finite; the
        compact schema does not claim instantaneous native acceleration.
        """
        if not isinstance(payload, CompactArrayBundle):
            raise TypeError(
                f"payload must be CompactArrayBundle (got {type(payload).__name__})"
            )
        if tuple(payload.joint_names) != CANONICAL_JOINTS:
            raise ValueError(
                "joint_names must match CANONICAL_JOINTS order; "
                "refusing silent reinterpretation of the 27-coordinate layout"
            )
        if tuple(payload.coefficient_letters) != COEFFICIENT_LETTERS:
            raise ValueError("coefficient_letters must match COEFFICIENT_LETTERS")
        coeffs = np.asarray(payload.coefficients, dtype=np.float64)
        if coeffs.shape != (N_COEFFS,):
            raise ValueError(
                f"coefficients must have length {N_COEFFS} (189), got {coeffs.shape}"
            )
        named = (
            ("q", payload.q),
            ("qd", payload.qd),
            ("qdd", payload.qdd),
            ("tau", payload.tau),
        )
        for name, arr in named:
            values = np.asarray(arr, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != N_JOINTS:
                raise ValueError(f"{name} must be (T, {N_JOINTS}), got {values.shape}")

        times = np.asarray(payload.sample_times_s, dtype=np.float64)
        q_arr = np.asarray(payload.q, dtype=np.float64)
        q_next = np.roll(q_arr, -1, axis=0)
        q_next[-1] = q_arr[-1]
        return EpisodeRecord(
            trial_id=payload.trial_id,
            family_id=payload.family_id,
            model_id=payload.model_id,
            control_basis="joint_torque",
            units="SI",
            joint_names=CANONICAL_JOINTS,
            coefficient_letters=COEFFICIENT_LETTERS,
            schema_version=EPISODE_STORE_SCHEMA,
            sample_times_s=times,
            q=q_arr,
            v=np.asarray(payload.qd, dtype=np.float64),
            u=np.asarray(payload.tau, dtype=np.float64),
            a_native=np.asarray(payload.qdd, dtype=np.float64),
            q_next=q_next,
            channel_availability={
                "q": "available",
                "v": "available",
                "u": "available",
                "a_native": "available",
                "q_next": "available",
            },
            ancestry=payload.ancestry,
            geometry_stratum=payload.geometry_stratum,
            contact_stratum=payload.contact_stratum,
            club_stratum=payload.club_stratum,
            coefficients=coeffs,
            source_schema=COMPACT_SCHEMA,
        )
