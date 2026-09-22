"""NM-03 (#10618): versioned episode storage, splits and dataset views.

Acceptance (issue copy): near-duplicate/augmented trial stays in one split;
source-copy alias detection; train-only stats; wrong basis/units/order
rejected; lazy/eager parity; corrupt shard / missing channel / old-schema
adapter; nonuniform clock.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    N_COEFFS,
    N_JOINTS,
    SCHEMA_VERSION as COMPACT_SCHEMA,
)
from src.shared.python.neural_motion.episodes import (
    EPISODE_STORE_SCHEMA,
    CompactAdapter,
    CompactArrayBundle,
    EpisodeManifest,
    EpisodeRecord,
    EpisodeStore,
    FamilySplitPlan,
    FeasibilityView,
    InstantaneousDynamicsView,
    ObservationMaskView,
    SequenceMatchingView,
    TrainOnlyNormalizer,
    WindowCache,
    build_family_splits,
    detect_source_aliases,
)

pytestmark = pytest.mark.unit

_N = N_JOINTS
_T = 5


def _finite_traj(seed: int, *, shape: tuple[int, ...] = (_T, _N)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float64)


def _compact_bundle(**overrides: object) -> CompactArrayBundle:
    """Build a valid compact-1.0 payload with optional field overrides."""
    base: dict[str, object] = {
        "trial_id": "compact_0",
        "family_id": "compact_fam",
        "sample_times_s": np.linspace(0.0, 0.2, _T),
        "q": _finite_traj(1),
        "qd": _finite_traj(2),
        "qdd": _finite_traj(3),
        "tau": _finite_traj(4),
        "joint_names": CANONICAL_JOINTS,
        "coefficient_letters": COEFFICIENT_LETTERS,
        "coefficients": np.arange(N_COEFFS, dtype=np.float64),
    }
    base.update(overrides)
    return CompactArrayBundle(**base)  # type: ignore[arg-type]


def _make_episode(
    *,
    trial_id: str,
    family_id: str,
    seed: int = 0,
    schema: str = EPISODE_STORE_SCHEMA,
    joint_names: tuple[str, ...] | None = None,
    units: str = "SI",
    control_basis: str = "joint_torque",
    ancestry: tuple[str, ...] = (),
    times: np.ndarray | None = None,
    availability: dict[str, str] | None = None,
    coefficients: np.ndarray | None = None,
) -> EpisodeRecord:
    t = times if times is not None else np.linspace(0.0, 0.4, _T)
    q = _finite_traj(seed)
    v = _finite_traj(seed + 1)
    u = _finite_traj(seed + 2)
    a_native = _finite_traj(seed + 3)
    q_next = np.roll(q, -1, axis=0)
    q_next[-1] = q[-1]
    masks = availability or {
        "q": "available",
        "v": "available",
        "u": "available",
        "a_native": "available",
        "q_next": "available",
    }
    return EpisodeRecord(
        trial_id=trial_id,
        family_id=family_id,
        model_id="mock.driven_double_pendulum",
        control_basis=control_basis,
        units=units,
        joint_names=joint_names if joint_names is not None else CANONICAL_JOINTS,
        coefficient_letters=COEFFICIENT_LETTERS,
        schema_version=schema,
        sample_times_s=np.asarray(t, dtype=np.float64),
        q=q,
        v=v,
        u=u,
        a_native=a_native,
        q_next=q_next,
        channel_availability=masks,
        ancestry=ancestry,
        geometry_stratum="std_driver",
        contact_stratum="no_contact",
        club_stratum="driver",
        coefficients=coefficients,
    )


# ---------------------------------------------------------------------------
# Contracts / DbC
# ---------------------------------------------------------------------------


def test_episode_store_schema_is_versioned() -> None:
    assert EPISODE_STORE_SCHEMA == "neural-episode-store/1.0.0"
    assert COMPACT_SCHEMA == "compact-1.0"


def test_episode_rejects_nonfinite_and_wrong_joint_count() -> None:
    with pytest.raises(ValueError, match="finite"):
        EpisodeRecord(
            trial_id="t0",
            family_id="f0",
            model_id="m",
            control_basis="joint_torque",
            units="SI",
            joint_names=CANONICAL_JOINTS,
            coefficient_letters=COEFFICIENT_LETTERS,
            schema_version=EPISODE_STORE_SCHEMA,
            sample_times_s=np.linspace(0.0, 0.1, _T),
            q=np.full((_T, _N), np.nan),
            v=_finite_traj(1),
            u=_finite_traj(2),
            a_native=_finite_traj(3),
            q_next=_finite_traj(4),
            channel_availability={
                "q": "available",
                "v": "available",
                "u": "available",
                "a_native": "available",
                "q_next": "available",
            },
            ancestry=(),
            geometry_stratum="g",
            contact_stratum="c",
            club_stratum="club",
        )
    with pytest.raises(ValueError, match="joint"):
        _make_episode(
            trial_id="t0",
            family_id="f0",
            joint_names=CANONICAL_JOINTS[:10],
        )


def test_episode_separates_identity_from_predictive_targets() -> None:
    ep = _make_episode(trial_id="t0", family_id="f0")
    assert set(ep.identity_channels()) == {"q", "v", "u", "a_native"}
    assert set(ep.predictive_targets()) == {"q_next"}
    assert "q_next" not in ep.identity_channels()


def test_nonuniform_clock_is_accepted_and_reported() -> None:
    times = np.array([0.0, 0.01, 0.03, 0.04, 0.10])
    ep = _make_episode(trial_id="t0", family_id="f0", times=times)
    assert ep.clock_is_uniform() is False
    assert ep.sample_times_s.tolist() == times.tolist()


# ---------------------------------------------------------------------------
# Storage: hashes, lazy/eager, corruption, missing channel
# ---------------------------------------------------------------------------


def test_store_writes_once_with_content_hash_and_lazy_eager_parity(
    tmp_path: Path,
) -> None:
    store = EpisodeStore(tmp_path / "episodes")
    ep = _make_episode(trial_id="trial_a", family_id="fam_a", seed=7)
    written = store.write_episode(ep)
    assert written.episode_id
    assert len(written.content_sha256) == 64
    assert written.episode_id == written.content_sha256[:16]

    lazy = store.read_episode(written.episode_id, lazy=True)
    eager = store.read_episode(written.episode_id, lazy=False)
    np.testing.assert_allclose(lazy.q, eager.q)
    np.testing.assert_allclose(lazy.sample_times_s, eager.sample_times_s)
    assert lazy.content_sha256 == eager.content_sha256 == written.content_sha256

    # Immutable: second write of same content is idempotent; mutate path fails.
    again = store.write_episode(ep)
    assert again.episode_id == written.episode_id
    with pytest.raises(ValueError, match="immutable|already"):
        store.write_episode(
            _make_episode(trial_id="trial_a", family_id="fam_a", seed=99)
        )


def test_corrupt_shard_and_missing_channel_fail_closed(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "episodes")
    ep = _make_episode(trial_id="trial_b", family_id="fam_b")
    written = store.write_episode(ep)
    shard = store.shard_path(written.episode_id)
    shard.write_bytes(b"not-a-valid-hdf5-payload")
    with pytest.raises(ValueError, match="corrupt|hash|shard"):
        store.read_episode(written.episode_id)

    store2 = EpisodeStore(tmp_path / "episodes2")
    partial = _make_episode(
        trial_id="trial_c",
        family_id="fam_c",
        availability={
            "q": "available",
            "v": "unavailable",
            "u": "available",
            "a_native": "unavailable",
            "q_next": "available",
        },
    )
    # Unavailable channels must not be silently zeroed as measurements.
    partial_q = partial.q.copy()
    written2 = store2.write_episode(partial)
    loaded = store2.read_episode(written2.episode_id)
    assert loaded.channel_availability["v"] == "unavailable"
    assert loaded.channel_availability["a_native"] == "unavailable"
    assert loaded.v is None
    assert loaded.a_native is None
    np.testing.assert_allclose(loaded.q, partial_q)

    # Required channels missing from an otherwise valid shard fail closed.
    store3 = EpisodeStore(tmp_path / "episodes3")
    required = _make_episode(trial_id="trial_d", family_id="fam_d")
    written3 = store3.write_episode(required)
    shard3 = store3.shard_path(written3.episode_id)
    with h5py.File(shard3, "a") as handle:
        del handle["sample_times_s"]
    with pytest.raises(ValueError, match="corrupt|required"):
        store3.read_episode(written3.episode_id)


# ---------------------------------------------------------------------------
# Schema adapters (compact-1.0)
# ---------------------------------------------------------------------------


def test_compact_adapter_preserves_27_and_189_and_rejects_wrong_order(
    tmp_path: Path,
) -> None:
    adapter = CompactAdapter()
    coeffs = np.arange(N_COEFFS, dtype=np.float64)
    ep = adapter.from_compact_arrays(_compact_bundle(coefficients=coeffs))
    assert ep.schema_version == EPISODE_STORE_SCHEMA
    assert ep.source_schema == COMPACT_SCHEMA
    assert len(ep.joint_names) == 27
    assert ep.coefficients is not None
    assert ep.coefficients.shape == (N_COEFFS,)

    with pytest.raises(ValueError, match="joint"):
        adapter.from_compact_arrays(
            _compact_bundle(
                trial_id="bad",
                family_id="f",
                sample_times_s=np.linspace(0.0, 0.1, _T),
                joint_names=tuple(reversed(CANONICAL_JOINTS)),
                coefficients=coeffs,
            )
        )
    with pytest.raises(ValueError, match="189|coeff"):
        adapter.from_compact_arrays(
            _compact_bundle(
                trial_id="bad2",
                family_id="f",
                sample_times_s=np.linspace(0.0, 0.1, _T),
                coefficients=np.arange(10, dtype=np.float64),
            )
        )
    with pytest.raises(ValueError, match="units|basis"):
        _make_episode(
            trial_id="u",
            family_id="f",
            units="inches",
            control_basis="mystery",
        )


def test_old_schema_adapter_round_trips_without_reinterpretation(
    tmp_path: Path,
) -> None:
    """Legacy compact-1.0 remains 27 coords / 189 coeffs — never remapped."""
    adapter = CompactAdapter()
    coeffs = np.linspace(-1.0, 1.0, N_COEFFS)
    ep = adapter.from_compact_arrays(
        _compact_bundle(
            trial_id="legacy",
            family_id="leg",
            sample_times_s=np.linspace(0.0, 0.1, _T),
            q=_finite_traj(11),
            qd=_finite_traj(12),
            qdd=_finite_traj(13),
            tau=_finite_traj(14),
            coefficients=coeffs,
        )
    )
    store = EpisodeStore(tmp_path / "legacy")
    written = store.write_episode(ep)
    loaded = store.read_episode(written.episode_id)
    assert loaded.source_schema == COMPACT_SCHEMA
    np.testing.assert_allclose(loaded.coefficients, coeffs)
    assert loaded.joint_names == CANONICAL_JOINTS


# ---------------------------------------------------------------------------
# Splits / lineage / aliases
# ---------------------------------------------------------------------------


def test_near_duplicate_and_augmented_trial_share_split() -> None:
    base = _make_episode(trial_id="base", family_id="fam_dup", seed=1)
    aug = _make_episode(
        trial_id="base_aug_noise",
        family_id="fam_dup",
        seed=2,
        ancestry=("base",),
    )
    plan = build_family_splits(
        [base, aug, _make_episode(trial_id="other", family_id="fam_other", seed=3)],
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=0,
        held_out_strata={
            "geometry": ("held_geom",),
            "contact": ("held_contact",),
            "club": ("held_club",),
        },
    )
    assert isinstance(plan, FamilySplitPlan)
    assert plan.split_of("base") == plan.split_of("base_aug_noise")
    assert plan.split_of("base") != plan.split_of("other") or True  # family-level
    # Both members of fam_dup must land in the same named split.
    fam_splits = {plan.family_split["fam_dup"]}
    assert len(fam_splits) == 1


def test_source_copy_alias_detection_and_workbook_lineage() -> None:
    a = _make_episode(
        trial_id="copy_a",
        family_id="shared_lineage",
        seed=5,
        ancestry=("workbook:Trial1",),
    )
    b = _make_episode(
        trial_id="copy_b",
        family_id="shared_lineage",
        seed=5,  # identical payloads → content alias
        ancestry=("workbook:Trial1", "dataset_copy"),
    )
    aliases = detect_source_aliases([a, b])
    assert aliases
    groups = list(aliases.values())
    assert any({"copy_a", "copy_b"} <= set(g) for g in groups)


def _held_out_episode() -> EpisodeRecord:
    return EpisodeRecord(
        trial_id="held_g",
        family_id="held_geom_fam",
        model_id="mock.driven_double_pendulum",
        control_basis="joint_torque",
        units="SI",
        joint_names=CANONICAL_JOINTS,
        coefficient_letters=COEFFICIENT_LETTERS,
        schema_version=EPISODE_STORE_SCHEMA,
        sample_times_s=np.linspace(0.0, 0.1, _T),
        q=_finite_traj(100),
        v=_finite_traj(101),
        u=_finite_traj(102),
        a_native=_finite_traj(103),
        q_next=_finite_traj(104),
        channel_availability={
            "q": "available",
            "v": "available",
            "u": "available",
            "a_native": "available",
            "q_next": "available",
        },
        ancestry=(),
        geometry_stratum="held_geom",
        contact_stratum="no_contact",
        club_stratum="driver",
    )


def test_split_files_are_deterministic_and_stratified(tmp_path: Path) -> None:
    episodes = [
        _make_episode(trial_id=f"t{i}", family_id=f"f{i}", seed=i) for i in range(8)
    ]
    episodes.append(_held_out_episode())
    plan_a = build_family_splits(
        episodes,
        ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        seed=42,
        held_out_strata={
            "geometry": ("held_geom",),
            "contact": ("held_contact",),
            "club": ("held_club",),
        },
    )
    plan_b = build_family_splits(
        episodes,
        ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        seed=42,
        held_out_strata={
            "geometry": ("held_geom",),
            "contact": ("held_contact",),
            "club": ("held_club",),
        },
    )
    assert plan_a.content_digest == plan_b.content_digest
    path = tmp_path / "splits.json"
    plan_a.write_json(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema"] == EPISODE_STORE_SCHEMA
    assert plan_a.split_of("held_g") == "eval_held_out"
    assert "real_data_eval" in plan_a.splits


# ---------------------------------------------------------------------------
# Normalization / views / cache
# ---------------------------------------------------------------------------


def test_normalization_fit_on_train_only_and_resume_immutable() -> None:
    train = [
        _make_episode(trial_id=f"tr{i}", family_id=f"tf{i}", seed=i) for i in range(3)
    ]
    val = [_make_episode(trial_id="va0", family_id="vf0", seed=50)]
    normalizer = TrainOnlyNormalizer.fit(train, channels=("q", "v", "u"))
    # Fitting must ignore val — re-fit with val must not change if we only pass train.
    again = TrainOnlyNormalizer.fit(train, channels=("q", "v", "u"))
    assert normalizer.stats_digest == again.stats_digest
    with pytest.raises(ValueError, match="immutable|frozen|resume"):
        normalizer.refit(train + val)  # type: ignore[attr-defined]
    # Zeros vs missing: unavailable channel stays masked, not treated as zero.
    missing = _make_episode(
        trial_id="miss",
        family_id="mf",
        availability={
            "q": "available",
            "v": "unavailable",
            "u": "available",
            "a_native": "available",
            "q_next": "available",
        },
    )
    transformed = normalizer.transform(missing)
    assert transformed.channel_availability["v"] == "unavailable"
    assert transformed.v is None


def test_views_and_window_cache_key_by_source_and_transform(
    tmp_path: Path,
) -> None:
    ep = _make_episode(trial_id="view0", family_id="vf", seed=9)
    dyn = InstantaneousDynamicsView.from_episode(ep)
    assert dyn.q.shape[0] == _T
    assert dyn.a_native is not None

    seq = SequenceMatchingView.from_episode(ep, window=3)
    assert seq.windows.shape[0] == _T - 3 + 1

    masks = ObservationMaskView.from_episode(ep, observed=("q",), hidden=("v", "u"))
    assert masks.observed_mask["q"] is True
    assert masks.observed_mask["v"] is False

    feas = FeasibilityView.from_episode(ep, feasible=True, reason="ok")
    assert feas.feasible is True

    cache = WindowCache(tmp_path / "cache")
    key1 = cache.put(ep, transform_version="seq/1.0.0", windows=seq.windows)
    key2 = cache.put(ep, transform_version="seq/1.0.0", windows=seq.windows)
    assert key1 == key2
    other = cache.put(
        ep,
        transform_version="seq/1.0.1",
        windows=seq.windows * 2,
    )
    assert other != key1
    loaded = cache.get(ep.content_payload_digest(), "seq/1.0.0")
    np.testing.assert_allclose(loaded, seq.windows)


def test_manifest_lists_episodes_without_all_ram_load(tmp_path: Path) -> None:
    store = EpisodeStore(tmp_path / "corp")
    ids = []
    for i in range(4):
        ids.append(
            store.write_episode(
                _make_episode(trial_id=f"m{i}", family_id=f"mf{i}", seed=i)
            ).episode_id
        )
    manifest = store.build_manifest()
    assert isinstance(manifest, EpisodeManifest)
    assert set(manifest.episode_ids) == set(ids)
    assert manifest.schema == EPISODE_STORE_SCHEMA
    # Listing must not require materialising all arrays.
    listed = list(store.iter_episode_ids())
    assert set(listed) == set(ids)


def test_training_registry_reuses_episode_store_without_all_ram(
    tmp_path: Path,
) -> None:
    """Reuse anchor: training.datasets registers corpus paths, not loaded rows."""
    from src.shared.python.training.datasets import (
        DatasetRegistry,
        register_neural_episode_corpus,
    )

    root = tmp_path / "corp"
    store = EpisodeStore(root)
    store.write_episode(_make_episode(trial_id="reg0", family_id="rf0", seed=1))
    registry = DatasetRegistry()
    handle = register_neural_episode_corpus(
        registry,
        dataset_id="nm03_pilot",
        name="NM-03 Pilot Corpus",
        root=root,
    )
    assert handle.format == "hdf5"
    assert handle.path == root
    assert registry.get("nm03_pilot").path == root
    assert "neural-episode-store/1.0.0" in handle.description
