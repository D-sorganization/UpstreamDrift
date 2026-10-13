"""Legacy Data Processing Imports Retain the Pinned Provider Contract."""

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def test_legacy_core_and_exceptions_are_pinned_provider_modules():
    from src.shared.python.sidekick_data_processing_bridge import load_provider_module

    core = load_provider_module("core")
    errors = load_provider_module("exceptions")
    assert (
        importlib.import_module("src.shared.python.sidekick.data_processing.core")
        is core
    )
    assert (
        importlib.import_module("src.shared.python.sidekick.data_processing.exceptions")
        is errors
    )
    assert (
        importlib.import_module("shared.python.sidekick.data_processing.core") is core
    )
    assert (
        importlib.import_module("shared.python.sidekick.data_processing.exceptions")
        is errors
    )
    assert importlib.import_module("sidekick.data_processing.core") is core
    assert importlib.import_module("sidekick.data_processing.exceptions") is errors
    assert core.FileIOError is errors.FileIOError
    helper_path = Path(inspect.getfile(core.validate_pandas_formula)).resolve()
    assert "vendor/ud-tools/src/shared/python/safe_pandas_eval.py" in str(helper_path)


def test_pinned_filter_and_rejected_query_preserve_data():
    from src.shared.python.sidekick.data_processing import (
        DataProcessorEngine,
        FilterError,
    )

    engine = DataProcessorEngine()
    engine.data = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert engine.filter_data("a", ">", 1.0).data["a"].tolist() == [2.0, 3.0]
    before = engine.data.copy()
    with pytest.raises(FilterError):
        engine.query("a > @unbound")
    pd.testing.assert_frame_equal(engine.data, before)


def test_provider_curve_fit_constant_target_uses_zero_r_squared():
    from src.shared.python.sidekick.data_processing import DataProcessorEngine
    from src.shared.python.sidekick_data_processing_bridge import load_provider_module

    core = load_provider_module("core")
    engine = DataProcessorEngine()
    engine.data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 5.0]})
    result = engine.fit_curve("x", "y", core.FitType.LINEAR)
    assert result.r_squared == 0.0


def test_legacy_public_exports_and_alias_warning():
    from src.shared.python.sidekick import data_processing as legacy
    from src.shared.python.sidekick_data_processing_bridge import load_provider_module

    core = load_provider_module("core")
    errors = load_provider_module("exceptions")
    for name in legacy.__all__:
        assert getattr(legacy, name) is not None
    assert legacy.DataProcessorEngine is core.DataProcessorEngine
    assert legacy.ProcessingResult is core.ProcessingResult
    assert legacy.FileIOError is errors.FileIOError
    repo_root = Path(__file__).resolve().parents[3]
    command = (
        "import warnings; warnings.simplefilter('always', DeprecationWarning); "
        "import runpy; "
        "runpy.run_path('src/shared/python/sidekick/data_processing/core.py'); "
        "runpy.run_path('src/shared/python/sidekick/data_processing/exceptions.py')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert (
        "Legacy sidekick.data_processing.core resolves to the pinned Tools provider"
        in completed.stderr
    )
    assert (
        "Legacy sidekick.data_processing.exceptions resolves to the pinned Tools provider"
        in completed.stderr
    )


def test_bridge_rejects_unknown_module_and_missing_vendor(monkeypatch, tmp_path):
    from src.shared.python import _seam_redirect
    from src.shared.python._seam_redirect import SeamResolutionError
    from src.shared.python.sidekick_data_processing_bridge import load_provider_module

    with pytest.raises(ValueError, match="unsupported data processing provider module"):
        load_provider_module("io")
    monkeypatch.setattr(_seam_redirect, "vendor_search_paths", lambda: (str(tmp_path),))
    monkeypatch.delitem(sys.modules, "_pinned_tools__sidekick", raising=False)
    with pytest.raises(SeamResolutionError, match="not in the pinned Tools tree"):
        load_provider_module("core")


def test_bridge_rejects_unexpected_provider_identity(monkeypatch):
    from types import SimpleNamespace

    from src.shared.python import sidekick_data_processing_bridge as bridge

    monkeypatch.setattr(
        bridge,
        "load_pinned_tools_package",
        lambda _: SimpleNamespace(__name__="untrusted"),
    )
    with pytest.raises(ValueError, match="unexpected pinned provider identity"):
        bridge.load_provider_module("core")
