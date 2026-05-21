"""Regression tests for issue #5738: Sidekick embed adapter path in models.yaml.

PR #5681 deleted src/shared/python/upstream_drift_tools/ui/tools_sidebar/_embed_adapter.py
and moved the sidekick adapter to src/tools/sidekick/_embed_adapter.py.  The
Sidekick tile in src/config/models.yaml was not updated, so SpecialAppHandler.launch()
fails with "script not found".

Contract these tests lock in:

1. Every ``special_app`` entry in models.yaml whose ``path`` ends with a known
   .py extension must resolve to a file that actually exists on disk.
2. The Sidekick tile in particular must resolve to an existing file (regression
   guard for #5738).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"


def _load_models() -> list[dict]:
    with MODELS_YAML.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("models", [])


def _special_app_script_entries() -> list[tuple[str, str]]:
    """Return (model_id, path) pairs for local special_app entries with .py paths.

    Entries with ``provider`` set are served from a sibling repository and are
    only resolvable at runtime when that repo is present; they are excluded from
    this static existence check.  Entries with a ``source_root`` that resolves
    to a path outside the current repo are similarly excluded.
    """
    result = []
    for model in _load_models():
        if model.get("type") != "special_app":
            continue
        # Skip entries served by an external provider (e.g. Tools repo)
        if model.get("provider"):
            continue
        # Skip entries with a source_root that is an external/sibling directory
        source_root = model.get("source_root", "")
        if source_root and not (REPO_ROOT / source_root).exists():
            continue
        path = model.get("path", "")
        if path.endswith(".py"):
            result.append((model["id"], path))
    return result


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


def test_models_yaml_exists() -> None:
    """Precondition: the config file must be present."""
    assert MODELS_YAML.exists(), f"models.yaml not found at {MODELS_YAML}"


def test_sidekick_entry_present() -> None:
    """models.yaml must contain a 'sidekick' entry (regression guard)."""
    ids = {m["id"] for m in _load_models()}
    assert "sidekick" in ids, "Sidekick tile missing from models.yaml"


def test_sidekick_path_exists_on_disk() -> None:
    """Bug #5738: the sidekick path in models.yaml must resolve to a real file.

    This is the primary regression guard.  After the fix models.yaml should
    point to src/tools/sidekick/_embed_adapter.py (which exists), not to
    the deleted src/shared/python/upstream_drift_tools/ui/tools_sidebar/_embed_adapter.py.
    """
    models = _load_models()
    sidekick = next((m for m in models if m["id"] == "sidekick"), None)
    assert sidekick is not None, "Sidekick tile missing from models.yaml"

    declared_path: str = sidekick.get("path", "")
    assert declared_path, "Sidekick entry has no 'path' key"

    resolved = REPO_ROOT / declared_path
    assert resolved.exists(), (
        f"Sidekick path declared in models.yaml does not exist on disk.\n"
        f"  Declared : {declared_path}\n"
        f"  Resolved : {resolved}\n"
        f"  Hint     : update models.yaml to point to src/tools/sidekick/_embed_adapter.py"
    )


@pytest.mark.parametrize("model_id,rel_path", _special_app_script_entries())
def test_special_app_script_paths_exist(model_id: str, rel_path: str) -> None:
    """Every special_app .py path referenced in models.yaml must exist.

    DbC precondition enforced by SpecialAppHandler.launch(): if the script is
    missing it logs a warning and returns False, silently breaking the tile.
    This test surfaces the breakage at CI time instead.
    """
    resolved = REPO_ROOT / rel_path
    assert resolved.exists(), (
        f"special_app '{model_id}' references a missing script.\n"
        f"  Declared : {rel_path}\n"
        f"  Resolved : {resolved}"
    )
