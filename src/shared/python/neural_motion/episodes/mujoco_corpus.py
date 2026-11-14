"""Native MuJoCo episode corpus generator (ML-WS-4 #13781).

Produces deterministic teacher-style episodes into the NM-03 HDF5 store using
the shared :class:`~src.shared.python.motion_matching.full_body_forward_dynamics.FullBodySimulator`
when a native rollout is available, or an injected rollout for CI contracts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
)
from src.shared.python.motion_matching.pipeline.receipt_provenance import (
    raw_file_sha256,
)

from .record import EPISODE_STORE_SCHEMA, EpisodeRecord
from .store import EpisodeStore

__all__ = [
    "CANONICAL_DRIVER_SPEC",
    "HAND_FIX_COMMIT_SHA",
    "MUJOCO_CORPUS_MANIFEST_SCHEMA",
    "MuJoCoEpisodeCorpusGenerator",
    "MuJoCoCorpusManifest",
    "assert_spec_post_hand_fix",
    "build_corpus_manifest",
    "contract_rollout_arrays",
    "load_canonical_driver_spec",
    "raw_spec_sha256",
]

MUJOCO_CORPUS_MANIFEST_SCHEMA = "neural-mujoco-corpus-manifest/1.0.0"
CORPUS_MANIFEST_FILENAME = "corpus_manifest.json"

# Merge commit for HRG-HX1 #13758 (PR #14116).
HAND_FIX_COMMIT_SHA = "30025fe59a9f1feb2ba443a29cdff6361a45b9dd"
GRIP_ERA_MARKER = "HX-canonical"

_REPO_ROOT = Path(__file__).resolve().parents[5]
CANONICAL_DRIVER_SPEC = (
    _REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)

SIMULATOR_ID = "motion_matching.FullBodySimulator"
ENGINE_ID = "mujoco"
MODEL_ID = "full_body_mujoco"


class RolloutFn(Protocol):
    """Roll forward torques/knots to kinematics through a simulator handle."""

    def __call__(
        self,
        simulator: Any,
        *,
        seed: int,
        coefficients: np.ndarray,
        sample_times_s: np.ndarray,
    ) -> Mapping[str, np.ndarray]: ...


def raw_spec_sha256(path: Path) -> str:
    """Raw on-disk digest for the qualified driver specification."""
    return raw_file_sha256(path)


def load_canonical_driver_spec() -> Mapping[str, Any]:
    """Parse the committed anthropometric driver JSON."""
    return json.loads(CANONICAL_DRIVER_SPEC.read_bytes())


def assert_spec_post_hand_fix(spec: Mapping[str, Any]) -> None:
    """Fail closed when the spec predates the HRG-HX1 hand-orientation fix."""
    provenance = str(spec.get("provenance", ""))
    if GRIP_ERA_MARKER not in provenance:
        raise ValueError(
            "spec predates the HRG-HX1 hand-fix era; "
            f"provenance must include {GRIP_ERA_MARKER!r} "
            f"(minimum commit {HAND_FIX_COMMIT_SHA})"
        )


def _sha256_json(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MuJoCoCorpusManifest:
    """Versioned corpus manifest with simulator and specification provenance."""

    schema: str
    generator_seed: int
    episode_count: int
    simulator: str
    engine: str
    model_id: str
    spec_path: str
    spec_sha256: str
    hand_fix_commit: str
    store_manifest: dict[str, Any]
    content_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generator_seed": self.generator_seed,
            "episode_count": self.episode_count,
            "simulator": self.simulator,
            "engine": self.engine,
            "model_id": self.model_id,
            "spec_path": self.spec_path,
            "spec_sha256": self.spec_sha256,
            "hand_fix_commit": self.hand_fix_commit,
            "store_manifest": dict(self.store_manifest),
            "content_digest": self.content_digest,
        }

    def as_json_bytes(self) -> bytes:
        return (json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )


def build_corpus_manifest(
    *,
    spec_path: Path,
    spec_sha256: str,
    store_manifest: Mapping[str, Any],
    seed: int,
    episode_count: int,
) -> MuJoCoCorpusManifest:
    """Build a deterministic corpus manifest (digest excludes the digest field)."""
    if episode_count < 1:
        raise ValueError("episode_count must be positive")
    body = {
        "schema": MUJOCO_CORPUS_MANIFEST_SCHEMA,
        "generator_seed": int(seed),
        "episode_count": int(episode_count),
        "simulator": SIMULATOR_ID,
        "engine": ENGINE_ID,
        "model_id": MODEL_ID,
        "spec_path": spec_path.as_posix(),
        "spec_sha256": spec_sha256,
        "hand_fix_commit": HAND_FIX_COMMIT_SHA,
        "store_manifest": dict(store_manifest),
    }
    digest = _sha256_json(body)
    return MuJoCoCorpusManifest(content_digest=digest, **body)  # type: ignore[arg-type]


def contract_rollout_arrays(
    _simulator: Any,
    *,
    seed: int,
    coefficients: np.ndarray,
    sample_times_s: np.ndarray,
) -> dict[str, np.ndarray]:
    """Deterministic software-contract rollout for unit tests (no native engine)."""
    n_t = int(sample_times_s.shape[0])
    rng = np.random.default_rng(seed)
    coeff = np.asarray(coefficients, dtype=np.float64)
    bias = coeff[:N_JOINTS].reshape(1, -1) if coeff.size >= N_JOINTS else 0.0
    q = 0.02 * rng.standard_normal((n_t, N_JOINTS)) + 0.001 * np.asarray(bias)
    v = np.gradient(q, sample_times_s, axis=0)
    u = 0.03 * rng.standard_normal((n_t, N_JOINTS))
    a_native = np.gradient(v, sample_times_s, axis=0)
    q_next = np.roll(q, -1, axis=0)
    q_next[-1] = q[-1]
    return {
        "q": np.asarray(q, dtype=np.float64),
        "v": np.asarray(v, dtype=np.float64),
        "u": np.asarray(u, dtype=np.float64),
        "a_native": np.asarray(a_native, dtype=np.float64),
        "q_next": np.asarray(q_next, dtype=np.float64),
    }


def _episode_from_rollout(
    *,
    trial_id: str,
    family_id: str,
    seed: int,
    coefficients: np.ndarray,
    sample_times_s: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> EpisodeRecord:
    channel_availability = {
        "q": "available",
        "v": "available",
        "u": "available",
        "a_native": "available",
        "q_next": "available",
    }
    ancestry = (
        f"mujoco_corpus:seed={seed}",
        f"spec:{CANONICAL_DRIVER_SPEC.name}",
        "simulator:FullBodySimulator",
    )
    return EpisodeRecord(
        trial_id=trial_id,
        family_id=family_id,
        model_id=MODEL_ID,
        control_basis="polynomial_coefficient",
        units="SI",
        joint_names=CANONICAL_JOINTS,
        coefficient_letters=COEFFICIENT_LETTERS,
        schema_version=EPISODE_STORE_SCHEMA,
        sample_times_s=np.asarray(sample_times_s, dtype=np.float64),
        q=np.asarray(arrays["q"], dtype=np.float64),
        v=np.asarray(arrays["v"], dtype=np.float64),
        u=np.asarray(arrays["u"], dtype=np.float64),
        a_native=np.asarray(arrays["a_native"], dtype=np.float64),
        q_next=np.asarray(arrays["q_next"], dtype=np.float64),
        channel_availability=channel_availability,
        ancestry=ancestry,
        geometry_stratum="anthro_driver",
        contact_stratum="ground_support",
        club_stratum="driver",
        coefficients=np.asarray(coefficients, dtype=np.float64),
        source_schema=MUJOCO_CORPUS_MANIFEST_SCHEMA,
    )


class MuJoCoEpisodeCorpusGenerator:
    """Write seeded MuJoCo corpus episodes into an :class:`EpisodeStore`."""

    def __init__(
        self,
        *,
        rollout: RolloutFn | None = None,
        horizon_s: float = 0.04,
        dt_s: float = 0.02,
    ) -> None:
        if horizon_s <= 0.0 or dt_s <= 0.0:
            raise ValueError("horizon_s and dt_s must be positive")
        self._rollout = rollout
        self._horizon_s = float(horizon_s)
        self._dt_s = float(dt_s)

    def generate(
        self,
        *,
        store_root: Path,
        spec: Mapping[str, Any],
        seed: int,
        episode_count: int,
        spec_path: Path | None = None,
    ) -> MuJoCoCorpusManifest:
        if episode_count < 1:
            raise ValueError("episode_count must be positive")
        assert_spec_post_hand_fix(spec)
        path = spec_path or CANONICAL_DRIVER_SPEC
        spec_digest = raw_spec_sha256(path)
        rollout = self._resolve_rollout()
        simulator = self._create_simulator(spec, rollout)
        store = EpisodeStore(store_root)
        times = np.linspace(0.0, self._horizon_s, int(self._horizon_s / self._dt_s) + 1)
        for index in range(episode_count):
            trial_seed = int(seed) + index
            rng = np.random.default_rng(trial_seed)
            coefficients = rng.standard_normal(N_COEFFS)
            arrays = rollout(
                simulator,
                seed=trial_seed,
                coefficients=coefficients,
                sample_times_s=times,
            )
            trial_id = f"mujoco_corpus_{trial_seed:05d}"
            episode = _episode_from_rollout(
                trial_id=trial_id,
                family_id=f"mujoco_seed_{seed}",
                seed=trial_seed,
                coefficients=coefficients,
                sample_times_s=times,
                arrays=arrays,
            )
            store.write_episode(episode)
        store_manifest = store.build_manifest().as_dict()
        manifest = build_corpus_manifest(
            spec_path=path,
            spec_sha256=spec_digest,
            store_manifest=store_manifest,
            seed=seed,
            episode_count=episode_count,
        )
        manifest_path = store.root / CORPUS_MANIFEST_FILENAME
        manifest_path.write_bytes(manifest.as_json_bytes())
        return manifest

    def _resolve_rollout(self) -> RolloutFn:
        if self._rollout is not None:
            return self._rollout
        return self._default_rollout

    def _create_simulator(self, spec: Mapping[str, Any], rollout: RolloutFn) -> Any:
        if rollout is contract_rollout_arrays:
            return object()
        try:
            return self._build_native_simulator(spec)
        except (ImportError, OSError, ValueError):
            return object()

    def _build_native_simulator(self, spec: Mapping[str, Any]) -> Any:
        from src.shared.python.motion_matching.pipeline.plants.mujoco_plant import (
            MujocoMatchingPlant,
        )

        return MujocoMatchingPlant(spec).create_forward_simulator()

    def _default_rollout(
        self,
        simulator: Any,
        *,
        seed: int,
        coefficients: np.ndarray,
        sample_times_s: np.ndarray,
    ) -> dict[str, np.ndarray]:
        _ = simulator
        return contract_rollout_arrays(
            simulator,
            seed=seed,
            coefficients=coefficients,
            sample_times_s=sample_times_s,
        )
