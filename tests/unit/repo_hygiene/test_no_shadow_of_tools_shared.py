"""Test for repo hygiene: no shadows of tools shared."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent.parent.parent
VENDOR_SRC = ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
LOCAL_SRC = ROOT / "src" / "shared" / "python"
SHADOW_CONFIG = ROOT / "scripts" / "config" / "shadow_modules.yaml"


def test_no_shadow_of_tools_shared() -> None:
    if not VENDOR_SRC.exists():
        pytest.skip("Vendor submodule not checked out")

    # Load allow-list
    allowed_shadows = set()
    if SHADOW_CONFIG.exists():
        with open(SHADOW_CONFIG, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            allowed_shadows = set(config.get("allowed_shadows", []))

    vendor_modules = [
        d.name
        for d in VENDOR_SRC.iterdir()
        if d.is_dir() and not d.name.startswith("__")
    ]

    violations = []
    for mod in vendor_modules:
        if mod in allowed_shadows:
            continue
        local_mod = LOCAL_SRC / mod
        if local_mod.exists():
            violations.append(mod)

    if violations:
        pytest.fail(
            f"Shadow copies of vendor tools found: {violations}. "
            f"Please remove them or add them to {SHADOW_CONFIG} with a tracking issue."
        )
