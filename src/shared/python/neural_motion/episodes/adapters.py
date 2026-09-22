"""Versioned schema adapters into EpisodeRecord (NM-03 #10618)."""

from __future__ import annotations

import numpy as np

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
    SCHEMA_VERSION as COMPACT_SCHEMA,
)

from .record import EPISODE_STORE_SCHEMA, EpisodeRecord

__all__ = ["CompactAdapter"]


class CompactAdapter:
    """Adapt compact-1.0 arrays without reinterpreting 27/189 layouts."""

    source_schema = COMPACT_SCHEMA

    def from_compact_arrays(
        self,
        *,
        trial_id: str,
        family_id: str,
        sample_times_s: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        tau: np.ndarray,
        joint_names: tuple[str, ...],
        coefficient_letters: tuple[str, ...],
        coefficients: np.ndarray,
        model_id: str = "compact.legacy",
        geometry_stratum: str = "unknown",
        contact_stratum: str = "unknown",
        club_stratum: str = "unknown",
        ancestry: tuple[str, ...] = (),
    ) -> EpisodeRecord:
        """Map compact q/qd/qdd/tau onto identity channels + coefficients.

        ``qdd`` is treated as interval finite-difference acceleration at the
        source clock and recorded under ``a_native`` only when finite; the
        compact schema does not claim instantaneous native acceleration.
        """
        if tuple(joint_names) != CANONICAL_JOINTS:
            raise ValueError(
                "joint_names must match CANONICAL_JOINTS order; "
                "refusing silent reinterpretation of the 27-coordinate layout"
            )
        if tuple(coefficient_letters) != COEFFICIENT_LETTERS:
            raise ValueError("coefficient_letters must match COEFFICIENT_LETTERS")
        coeffs = np.asarray(coefficients, dtype=np.float64)
        if coeffs.shape != (N_COEFFS,):
            raise ValueError(
                f"coefficients must have length {N_COEFFS} (189), got {coeffs.shape}"
            )
        for name, arr in (("q", q), ("qd", qd), ("qdd", qdd), ("tau", tau)):
            values = np.asarray(arr, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != N_JOINTS:
                raise ValueError(f"{name} must be (T, {N_JOINTS}), got {values.shape}")

        times = np.asarray(sample_times_s, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
        q_next = np.roll(q_arr, -1, axis=0)
        q_next[-1] = q_arr[-1]
        return EpisodeRecord(
            trial_id=trial_id,
            family_id=family_id,
            model_id=model_id,
            control_basis="joint_torque",
            units="SI",
            joint_names=CANONICAL_JOINTS,
            coefficient_letters=COEFFICIENT_LETTERS,
            schema_version=EPISODE_STORE_SCHEMA,
            sample_times_s=times,
            q=q_arr,
            v=np.asarray(qd, dtype=np.float64),
            u=np.asarray(tau, dtype=np.float64),
            a_native=np.asarray(qdd, dtype=np.float64),
            q_next=q_next,
            channel_availability={
                "q": "available",
                "v": "available",
                "u": "available",
                "a_native": "available",
                "q_next": "available",
            },
            ancestry=ancestry,
            geometry_stratum=geometry_stratum,
            contact_stratum=contact_stratum,
            club_stratum=club_stratum,
            coefficients=coeffs,
            source_schema=COMPACT_SCHEMA,
        )
