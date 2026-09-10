"""Resolve calibration only from the selected checkout or owning wheel."""

from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import sys

import pytest

from src.tools.capture_rig.reference_calibration import worker

pytestmark = pytest.mark.unit
PROVIDER = Path("shared/python/sidekick/lab/mocap/reference_placements.py")
WORKER = Path("src/tools/capture_rig/reference_calibration/worker.py")


@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("recorded", [True, False])
def test_installed_provider_requires_the_workers_distribution(
    tmp_path, monkeypatch, owned, recorded
):
    root = tmp_path / "site-packages"
    source = root / PROVIDER
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    monkeypatch.setattr(worker, "__file__", str(root / WORKER))
    monkeypatch.setattr(sys, "path", list(sys.path))
    distribution = SimpleNamespace(
        locate_file=lambda path: (root if owned else tmp_path / "other") / path,
        files=[PROVIDER, WORKER] if recorded else [],
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    if not owned or not recorded:
        with pytest.raises(ValueError, match="provider"):
            worker._configure_provider()
        return
    worker._configure_provider()
    assert sys.path[0] == str(root)
    assert str(root / "shared/python") in sys.path[:5]


def test_incomplete_checkout_never_falls_back_to_an_installed_provider(
    tmp_path, monkeypatch
):
    root = tmp_path / "checkout"
    (root / "vendor/ud-tools/src").mkdir(parents=True)
    monkeypatch.setattr(worker, "__file__", str(root / WORKER))
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ValueError, match="provider"):
        worker._configure_provider()
