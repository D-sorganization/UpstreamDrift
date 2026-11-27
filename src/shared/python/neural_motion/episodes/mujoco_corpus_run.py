"""MuJoCo corpus generation run receipt and trainer-facing loader (ML-WS-5 #13782)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.neural_motion.episodes.record import EpisodeRecord
from src.shared.python.neural_motion.episodes.splits import (
    FamilySplitPlan,
    build_family_splits,
)
from src.shared.python.neural_motion.episodes.store import EpisodeStore
from src.shared.python.neural_motion.json_io import (
    SortedJsonWritableMixin,
    write_sorted_json,
)

from .mujoco_corpus import (
    CORPUS_MANIFEST_FILENAME,
    MUJOCO_CORPUS_MANIFEST_SCHEMA,
    MuJoCoCorpusManifest,
    MuJoCoEpisodeCorpusGenerator,
    RolloutFn,
    load_canonical_driver_spec,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]

__all__ = [
    "CORPUS_RUN_RECEIPT_FILENAME",
    "MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA",
    "MuJoCoCorpusGenerationRequest",
    "MuJoCoCorpusRunReceipt",
    "MuJoCoCorpusTrainingBundle",
    "load_corpus_run_receipt",
    "load_mujoco_corpus_for_training",
    "read_corpus_manifest",
    "resolve_repo_head_sha",
    "run_mujoco_corpus_generation",
    "verify_corpus_manifest_digest",
]

CORPUS_RUN_RECEIPT_FILENAME = "corpus_run_receipt.json"
MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA = "neural-mujoco-corpus-run/1.0.0"
_DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
_DEFAULT_HELD_OUT: Mapping[str, tuple[str, ...]] = {
    "geometry": ("anthro_iron",),
    "contact": ("no_contact",),
    "club": ("iron7",),
}


def _sha256_json(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _manifest_body(data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": data["schema"],
        "generator_seed": int(data["generator_seed"]),
        "episode_count": int(data["episode_count"]),
        "simulator": str(data["simulator"]),
        "engine": str(data["engine"]),
        "model_id": str(data["model_id"]),
        "spec_path": str(data["spec_path"]),
        "spec_sha256": str(data["spec_sha256"]),
        "hand_fix_commit": str(data["hand_fix_commit"]),
        "store_manifest": dict(data["store_manifest"]),
    }


def _manifest_from_mapping(data: Mapping[str, Any]) -> MuJoCoCorpusManifest:
    return MuJoCoCorpusManifest(
        schema=str(data["schema"]),
        generator_seed=int(data["generator_seed"]),
        episode_count=int(data["episode_count"]),
        simulator=str(data["simulator"]),
        engine=str(data["engine"]),
        model_id=str(data["model_id"]),
        spec_path=str(data["spec_path"]),
        spec_sha256=str(data["spec_sha256"]),
        hand_fix_commit=str(data["hand_fix_commit"]),
        store_manifest=dict(data["store_manifest"]),
        content_digest=str(data["content_digest"]),
    )


def read_corpus_manifest(store_root: Path) -> MuJoCoCorpusManifest:
    """Load and digest-verify the WS-4 corpus manifest under ``store_root``."""
    path = Path(store_root) / CORPUS_MANIFEST_FILENAME
    verify_corpus_manifest_digest(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return _manifest_from_mapping(data)


def verify_corpus_manifest_digest(path: Path) -> str:
    """Recompute and return the corpus manifest digest; fail closed on mismatch."""
    if not path.is_file():
        raise ValueError(f"corpus manifest missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(data.get("content_digest", ""))
    computed = _sha256_json(_manifest_body(data))
    if recorded != computed:
        raise ValueError(
            "corpus manifest content_digest mismatch "
            f"(recorded={recorded}, computed={computed})"
        )
    return recorded


def resolve_repo_head_sha(repo_root: Path | None = None) -> str:
    """Return the current git HEAD SHA for provenance (empty when git is unavailable)."""
    root = repo_root or Path(__file__).resolve().parents[5]
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    sha = out.strip()
    if len(sha) != 40:
        return ""
    return sha


@dataclass(frozen=True)
class MuJoCoCorpusRunReceipt(SortedJsonWritableMixin):
    """Measured run receipt for a WS-4 corpus generation + family split."""

    schema: str
    store_root: str
    generator_seed: int
    episode_count: int
    split_seed: int
    wall_clock_s: float
    repo_commit_sha: str
    corpus_manifest_digest: str
    store_manifest_digest: str
    spec_sha256: str
    hand_fix_commit: str
    split_plan_digest: str
    split_counts: dict[str, int]
    held_out_strata: dict[str, tuple[str, ...]]
    split_ratios: dict[str, float]
    content_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "store_root": self.store_root,
            "generator_seed": self.generator_seed,
            "episode_count": self.episode_count,
            "split_seed": self.split_seed,
            "wall_clock_s": self.wall_clock_s,
            "repo_commit_sha": self.repo_commit_sha,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "store_manifest_digest": self.store_manifest_digest,
            "spec_sha256": self.spec_sha256,
            "hand_fix_commit": self.hand_fix_commit,
            "split_plan_digest": self.split_plan_digest,
            "split_counts": dict(self.split_counts),
            "held_out_strata": {
                key: list(values)
                for key, values in sorted(self.held_out_strata.items())
            },
            "split_ratios": dict(self.split_ratios),
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class MuJoCoCorpusGenerationRequest:
    """Inputs for a measured WS-4 corpus generation run."""

    store_root: Path
    seed: int
    episode_count: int
    split_seed: int
    spec: Mapping[str, Any] | None = None
    rollout: RolloutFn | None = None
    ratios: Mapping[str, float] | None = None
    held_out_strata: Mapping[str, Sequence[str]] | None = None
    receipt_path: Path | None = None
    repo_root: Path | None = None


@dataclass(frozen=True)
class MuJoCoCorpusTrainingBundle:
    """Episode store, family split, and manifest after DbC verification."""

    store: EpisodeStore
    split: FamilySplitPlan
    manifest: MuJoCoCorpusManifest
    receipt: MuJoCoCorpusRunReceipt | None = None


def load_corpus_run_receipt(path: Path) -> MuJoCoCorpusRunReceipt:
    data = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(data.pop("content_digest"))
    computed = _sha256_json(data)
    if recorded != computed:
        raise ValueError(
            "corpus run receipt content_digest mismatch "
            f"(recorded={recorded}, computed={computed})"
        )
    held = {
        key: tuple(str(v) for v in values)
        for key, values in dict(data.get("held_out_strata", {})).items()
    }
    ratios = {str(k): float(v) for k, v in dict(data.get("split_ratios", {})).items()}
    return MuJoCoCorpusRunReceipt(
        schema=str(data["schema"]),
        store_root=str(data["store_root"]),
        generator_seed=int(data["generator_seed"]),
        episode_count=int(data["episode_count"]),
        split_seed=int(data["split_seed"]),
        wall_clock_s=float(data["wall_clock_s"]),
        repo_commit_sha=str(data["repo_commit_sha"]),
        corpus_manifest_digest=str(data["corpus_manifest_digest"]),
        store_manifest_digest=str(data["store_manifest_digest"]),
        spec_sha256=str(data["spec_sha256"]),
        hand_fix_commit=str(data["hand_fix_commit"]),
        split_plan_digest=str(data["split_plan_digest"]),
        split_counts={str(k): int(v) for k, v in dict(data["split_counts"]).items()},
        held_out_strata=held,
        split_ratios=ratios,
        content_digest=recorded,
    )


def _load_episodes(store: EpisodeStore) -> list[EpisodeRecord]:
    return [store.read_episode(eid, lazy=True) for eid in store.iter_episode_ids()]


def _split_counts(plan: FamilySplitPlan) -> dict[str, int]:
    return {name: len(ids) for name, ids in plan.splits.items() if ids}


def build_corpus_run_receipt(
    *,
    store_root: Path,
    manifest: MuJoCoCorpusManifest,
    split: FamilySplitPlan,
    split_seed: int,
    wall_clock_s: float,
    repo_commit_sha: str,
    split_ratios: Mapping[str, float],
    held_out_strata: Mapping[str, Sequence[str]],
) -> MuJoCoCorpusRunReceipt:
    body = {
        "schema": MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA,
        "store_root": store_root.as_posix(),
        "generator_seed": manifest.generator_seed,
        "episode_count": manifest.episode_count,
        "split_seed": int(split_seed),
        "wall_clock_s": float(wall_clock_s),
        "repo_commit_sha": repo_commit_sha,
        "corpus_manifest_digest": manifest.content_digest,
        "store_manifest_digest": str(manifest.store_manifest["content_digest"]),
        "spec_sha256": manifest.spec_sha256,
        "hand_fix_commit": manifest.hand_fix_commit,
        "split_plan_digest": split.content_digest,
        "split_counts": _split_counts(split),
        "held_out_strata": {
            key: list(values) for key, values in sorted(held_out_strata.items())
        },
        "split_ratios": {k: float(v) for k, v in sorted(split_ratios.items())},
    }
    digest = _sha256_json(body)
    held = {key: tuple(values) for key, values in held_out_strata.items()}
    ratios_out = dict(split_ratios)
    return MuJoCoCorpusRunReceipt(
        schema=MUJOCO_CORPUS_RUN_RECEIPT_SCHEMA,
        store_root=store_root.as_posix(),
        generator_seed=manifest.generator_seed,
        episode_count=manifest.episode_count,
        split_seed=int(split_seed),
        wall_clock_s=float(wall_clock_s),
        repo_commit_sha=repo_commit_sha,
        corpus_manifest_digest=manifest.content_digest,
        store_manifest_digest=str(manifest.store_manifest["content_digest"]),
        spec_sha256=manifest.spec_sha256,
        hand_fix_commit=manifest.hand_fix_commit,
        split_plan_digest=split.content_digest,
        split_counts=_split_counts(split),
        held_out_strata=held,
        split_ratios=ratios_out,
        content_digest=digest,
    )


def run_mujoco_corpus_generation(
    request: MuJoCoCorpusGenerationRequest,
) -> tuple[MuJoCoCorpusManifest, FamilySplitPlan, MuJoCoCorpusRunReceipt]:
    if request.episode_count < 1:
        raise ValueError("episode_count must be positive")
    spec_doc = (
        request.spec if request.spec is not None else load_canonical_driver_spec()
    )
    split_ratios = dict(request.ratios or _DEFAULT_RATIOS)
    held = {
        key: tuple(values)
        for key, values in (request.held_out_strata or _DEFAULT_HELD_OUT).items()
    }
    store_root = Path(request.store_root)
    store_root.mkdir(parents=True, exist_ok=True)
    generator = MuJoCoEpisodeCorpusGenerator(rollout=request.rollout)
    started = time.perf_counter()
    manifest = generator.generate(
        store_root=store_root,
        spec=spec_doc,
        seed=request.seed,
        episode_count=request.episode_count,
    )
    wall_clock_s = time.perf_counter() - started
    store = EpisodeStore(store_root)
    episodes = _load_episodes(store)
    split = build_family_splits(
        episodes,
        ratios=split_ratios,
        seed=request.split_seed,
        held_out_strata=held,
    )
    receipt = build_corpus_run_receipt(
        store_root=store_root,
        manifest=manifest,
        split=split,
        split_seed=request.split_seed,
        wall_clock_s=wall_clock_s,
        repo_commit_sha=resolve_repo_head_sha(request.repo_root),
        split_ratios=split_ratios,
        held_out_strata=held,
    )
    out = request.receipt_path or (store_root / CORPUS_RUN_RECEIPT_FILENAME)
    write_sorted_json(out, receipt.as_dict())
    return manifest, split, receipt


def load_mujoco_corpus_for_training(
    store_root: Path,
    *,
    receipt_path: Path | None = None,
    expected_corpus_manifest_digest: str | None = None,
) -> MuJoCoCorpusTrainingBundle:
    root = Path(store_root)
    manifest_path = root / CORPUS_MANIFEST_FILENAME
    digest = verify_corpus_manifest_digest(manifest_path)
    if expected_corpus_manifest_digest is not None:
        if digest != expected_corpus_manifest_digest:
            raise ValueError(
                "corpus manifest digest does not match expected "
                f"({digest} != {expected_corpus_manifest_digest})"
            )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _manifest_from_mapping(data)
    if manifest.schema != MUJOCO_CORPUS_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported corpus manifest schema: {manifest.schema!r}")

    receipt: MuJoCoCorpusRunReceipt | None = None
    resolved_receipt = receipt_path
    if resolved_receipt is None:
        default_receipt = root / CORPUS_RUN_RECEIPT_FILENAME
        if default_receipt.is_file():
            resolved_receipt = default_receipt
    if resolved_receipt is not None:
        receipt = load_corpus_run_receipt(resolved_receipt)
        if receipt.corpus_manifest_digest != digest:
            raise ValueError(
                "receipt corpus_manifest_digest does not match on-disk manifest"
            )
        if receipt.store_manifest_digest != str(
            manifest.store_manifest["content_digest"]
        ):
            raise ValueError(
                "receipt store_manifest_digest does not match on-disk manifest"
            )
        receipt_root = Path(receipt.store_root)
        if not receipt_root.is_absolute():
            receipt_root = (_REPO_ROOT / receipt_root).resolve()
        if receipt_root != root.resolve():
            raise ValueError("receipt store_root does not match load path")

    store = EpisodeStore(root)
    episodes = _load_episodes(store)
    if not episodes:
        raise ValueError("corpus store is empty")
    if receipt is not None:
        split = build_family_splits(
            episodes,
            ratios=receipt.split_ratios,
            seed=receipt.split_seed,
            held_out_strata=receipt.held_out_strata,
        )
        if split.content_digest != receipt.split_plan_digest:
            raise ValueError("recomputed split digest does not match receipt")
    else:
        split = build_family_splits(
            episodes,
            ratios=_DEFAULT_RATIOS,
            seed=manifest.generator_seed,
            held_out_strata=_DEFAULT_HELD_OUT,
        )
    return MuJoCoCorpusTrainingBundle(
        store=store,
        split=split,
        manifest=manifest,
        receipt=receipt,
    )
