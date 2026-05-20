"""Tests for motion_matching/train_option3_example.py.

These tests cover the CLI entry point script for Option-3 inverse cVAE
training. The heavy training pipeline (``train_inverse_cvae``) is mocked
so the tests stay fast (<30s) and avoid torch / dataset dependencies.

Scope:
    - ``_resolve_device`` helper logic (auto / cpu / cuda branches).
    - ``main()`` argparse + control flow (success, missing dataset,
      precondition rejection via ValueError/FileNotFoundError, generic
      exception path).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loader: train_option3_example.py is a standalone script, not a
# package, so we load it via importlib.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "motion_matching" / "train_option3_example.py"


@pytest.fixture(scope="module")
def train_module() -> ModuleType:
    """Import the standalone training script as a module."""
    spec = importlib.util.spec_from_file_location(
        "motion_matching_train_option3_example", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _resolve_device tests
# ---------------------------------------------------------------------------


def test_resolve_device_explicit_cpu(train_module: ModuleType) -> None:
    assert train_module._resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda(train_module: ModuleType) -> None:
    assert train_module._resolve_device("cuda") == "cuda"


def test_resolve_device_auto_with_cuda(train_module: ModuleType) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert train_module._resolve_device("auto") == "cuda"


def test_resolve_device_auto_without_cuda(train_module: ModuleType) -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert train_module._resolve_device("auto") == "cpu"


def test_resolve_device_auto_no_torch(train_module: ModuleType) -> None:
    # Simulate ImportError for torch by injecting a finder that fails.
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )  # type: ignore[index]

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    # Also evict any cached torch so the local ``import torch`` re-imports.
    with (
        patch("builtins.__import__", side_effect=fake_import),
        patch.dict(sys.modules, {}, clear=False),
    ):
        sys.modules.pop("torch", None)
        assert train_module._resolve_device("auto") == "cpu"


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


def _make_argv(dataset: Path, **extra: str) -> list[str]:
    argv = ["train_option3_example.py", "--dataset", str(dataset)]
    for k, v in extra.items():
        argv.extend([f"--{k.replace('_', '-')}", str(v)])
    return argv


def _fake_training_result() -> SimpleNamespace:
    epoch_metrics = SimpleNamespace(epoch=3, train_loss=0.5, val_recon=0.4, val_kl=0.1)
    return SimpleNamespace(
        output_dir=Path("/tmp/out"),
        best_epoch=2,
        parameter_count=12345,
        history=[epoch_metrics],
    )


def test_main_missing_dataset_returns_1(
    train_module: ModuleType, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist"
    with patch.object(sys, "argv", _make_argv(missing)):
        assert train_module.main() == 1


def test_main_happy_path_returns_0(train_module: ModuleType, tmp_path: Path) -> None:
    dataset = tmp_path / "compact"
    dataset.mkdir()
    fake_train = MagicMock(return_value=_fake_training_result())
    with (
        patch.object(train_module, "train_inverse_cvae", fake_train),
        patch.object(sys, "argv", _make_argv(dataset, device="cpu", epochs="1")),
    ):
        rc = train_module.main()
    assert rc == 0
    fake_train.assert_called_once()
    kwargs = fake_train.call_args.kwargs
    assert kwargs["dataset_path"] == dataset
    assert kwargs["device"] == "cpu"
    assert kwargs["epochs"] == 1


def test_main_value_error_returns_1(train_module: ModuleType, tmp_path: Path) -> None:
    dataset = tmp_path / "compact"
    dataset.mkdir()
    with (
        patch.object(
            train_module,
            "train_inverse_cvae",
            side_effect=ValueError("bad schema"),
        ),
        patch.object(sys, "argv", _make_argv(dataset, device="cpu")),
    ):
        assert train_module.main() == 1


def test_main_file_not_found_returns_1(
    train_module: ModuleType, tmp_path: Path
) -> None:
    dataset = tmp_path / "compact"
    dataset.mkdir()
    with (
        patch.object(
            train_module,
            "train_inverse_cvae",
            side_effect=FileNotFoundError("missing parquet"),
        ),
        patch.object(sys, "argv", _make_argv(dataset, device="cpu")),
    ):
        assert train_module.main() == 1


def test_main_generic_exception_returns_1(
    train_module: ModuleType, tmp_path: Path
) -> None:
    dataset = tmp_path / "compact"
    dataset.mkdir()
    with (
        patch.object(
            train_module,
            "train_inverse_cvae",
            side_effect=RuntimeError("kaboom"),
        ),
        patch.object(sys, "argv", _make_argv(dataset, device="cpu")),
    ):
        assert train_module.main() == 1


def test_main_passes_cli_overrides(train_module: ModuleType, tmp_path: Path) -> None:
    """Ensure CLI flags are threaded through to ``train_inverse_cvae``."""
    dataset = tmp_path / "compact"
    dataset.mkdir()
    fake_train = MagicMock(return_value=_fake_training_result())
    argv = [
        "train_option3_example.py",
        "--dataset",
        str(dataset),
        "--epochs",
        "7",
        "--batch-size",
        "8",
        "--lr",
        "0.001",
        "--latent-dim",
        "16",
        "--kl-anneal-epochs",
        "2",
        "--max-beta",
        "0.25",
        "--device",
        "cpu",
        "--seed",
        "42",
    ]
    with (
        patch.object(train_module, "train_inverse_cvae", fake_train),
        patch.object(sys, "argv", argv),
    ):
        assert train_module.main() == 0
    kw = fake_train.call_args.kwargs
    assert kw["epochs"] == 7
    assert kw["batch_size"] == 8
    assert kw["lr"] == pytest.approx(0.001)
    assert kw["kl_anneal_epochs"] == 2
    assert kw["max_beta"] == pytest.approx(0.25)
    assert kw["device"] == "cpu"
    assert kw["seed"] == 42
    # cvae_config carries latent_dim override.
    assert kw["cvae_config"].latent_dim == 16


def test_main_device_auto_resolves(train_module: ModuleType, tmp_path: Path) -> None:
    dataset = tmp_path / "compact"
    dataset.mkdir()
    fake_train = MagicMock(return_value=_fake_training_result())
    with (
        patch.object(train_module, "train_inverse_cvae", fake_train),
        patch.object(train_module, "_resolve_device", return_value="cpu") as rd,
        patch.object(sys, "argv", _make_argv(dataset, device="auto")),
    ):
        assert train_module.main() == 0
    rd.assert_called_once_with("auto")
    assert fake_train.call_args.kwargs["device"] == "cpu"
