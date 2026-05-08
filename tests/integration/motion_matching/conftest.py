"""Fixtures for the motion-matching end-to-end smoke test.

These fixtures intentionally err on the side of being *small*: a 10-trial
synthetic sweep with 30 timesteps and 6 joints is large enough to exercise
every code path the pipeline touches but tiny enough that a 5-epoch overfit
finishes in seconds on CPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# Default location of the optional real C3D file. Resolved against the repo
# root via ``pytest.rootpath`` to keep the fixture usable from any CWD.
_C3D_RELATIVE = Path("Data") / "Mocap C3D Files" / "CaptureExport Tour average.c3d"

# Synthetic-dataset shape constants (kept here so the smoke test reads cleanly).
SMOKE_N_TRIALS = 10
SMOKE_N_JOINTS = 6
SMOKE_N_TIMESTEPS = 30
SMOKE_SEED = 0


@pytest.fixture(scope="module")
def synthetic_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny synthetic sweep dataset on disk."""
    make_synthetic_sweep = pytest.importorskip(
        "src.shared.python.motion_matching.dataset"
    ).make_synthetic_sweep
    out = tmp_path_factory.mktemp("smoke_sweep")
    return make_synthetic_sweep(
        out,
        n_trials=SMOKE_N_TRIALS,
        n_joints=SMOKE_N_JOINTS,
        n_timesteps=SMOKE_N_TIMESTEPS,
        seed=SMOKE_SEED,
    )


@pytest.fixture(scope="module")
def loaded_dataset(synthetic_dataset_dir: Path):
    """Load the on-disk synthetic dataset eagerly."""
    load_sweep_dataset = pytest.importorskip(
        "src.shared.python.motion_matching.dataset"
    ).load_sweep_dataset
    return load_sweep_dataset(synthetic_dataset_dir, lazy=False)


@pytest.fixture(scope="module")
def trained_surrogate(loaded_dataset):
    """Train a tiny surrogate that overfits the synthetic dataset (CPU, 5 epochs)."""
    pytest.importorskip("torch")
    train_mod = pytest.importorskip("src.shared.python.motion_matching.surrogate.train")
    cfg = train_mod.TrainConfig(
        n_epochs=5,
        batch_size=2,
        lr=3.0e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        seed=0,
        val_fraction=0.1,
        test_fraction=0.1,
        use_amp=False,
        device="cpu",
    )
    return train_mod.train_surrogate(loaded_dataset, cfg)


@pytest.fixture(scope="session")
def real_c3d_path(pytestconfig: pytest.Config) -> Path | None:
    """Return the real cluster-marker C3D path if it exists, else ``None``."""
    candidate = pytestconfig.rootpath / _C3D_RELATIVE
    return candidate if candidate.exists() else None


@pytest.fixture
def fake_provenance():
    """Return a ``SourceProvenance`` factory for synthesised targets."""
    club_target_mod = pytest.importorskip(
        "src.shared.python.motion_matching.club_target"
    )

    def _make(filename: str = "smoke.bin"):
        return club_target_mod.SourceProvenance(
            filename=filename,
            format="synthetic",
            subject_id="SMOKE",
            trial_id="0",
            sha256=hashlib.sha256(b"").hexdigest(),
        )

    return _make
