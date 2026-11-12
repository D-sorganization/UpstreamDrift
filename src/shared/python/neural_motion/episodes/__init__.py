"""Versioned episode storage, family splits and thin dataset views (NM-03 #10618).

Schema: ``neural-episode-store/1.0.0``. Raw episodes are stored once with
content hashes; task views and window caches derive from source + transform
version. Family-level splits prevent near-duplicate / alias leakage across
train/val/test. Compact-1.0 adapters preserve the 27-joint / 189-coefficient
contract without silent reinterpretation.
"""

from __future__ import annotations

from .adapters import CompactAdapter, CompactArrayBundle
from .cache import WindowCache
from .normalize import TrainOnlyNormalizer
from .record import EPISODE_STORE_SCHEMA, EpisodeRecord
from .splits import FamilySplitPlan, build_family_splits, detect_source_aliases
from .store import EpisodeManifest, EpisodeStore
from .mujoco_corpus import (
    CANONICAL_DRIVER_SPEC,
    HAND_FIX_COMMIT_SHA,
    MUJOCO_CORPUS_MANIFEST_SCHEMA,
    MuJoCoCorpusManifest,
    MuJoCoEpisodeCorpusGenerator,
    assert_spec_post_hand_fix,
    build_corpus_manifest,
    contract_rollout_arrays,
    load_canonical_driver_spec,
    raw_spec_sha256,
)
from .views import (
    FeasibilityView,
    InstantaneousDynamicsView,
    ObservationMaskView,
    SequenceMatchingView,
)

__all__ = [
    "EPISODE_STORE_SCHEMA",
    "CompactAdapter",
    "CompactArrayBundle",
    "EpisodeManifest",
    "EpisodeRecord",
    "EpisodeStore",
    "FamilySplitPlan",
    "FeasibilityView",
    "InstantaneousDynamicsView",
    "ObservationMaskView",
    "SequenceMatchingView",
    "TrainOnlyNormalizer",
    "WindowCache",
    "assert_spec_post_hand_fix",
    "build_corpus_manifest",
    "build_family_splits",
    "CANONICAL_DRIVER_SPEC",
    "contract_rollout_arrays",
    "detect_source_aliases",
    "HAND_FIX_COMMIT_SHA",
    "load_canonical_driver_spec",
    "MUJOCO_CORPUS_MANIFEST_SCHEMA",
    "MuJoCoCorpusManifest",
    "MuJoCoEpisodeCorpusGenerator",
    "raw_spec_sha256",
]
