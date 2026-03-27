"""Unit tests for the dependency direction checker script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "check_dependency_direction.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_dependency_direction", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_exception_index_flags_invalid_and_expired_entries():
    module = _load_module()

    config = {
        "exceptions": [
            {
                "path": "shared/python/engine_core/engine_loaders.py",
                "owner": "@platform-architecture",
                "reason": "shim",
                "expires_on": "2027-01-31",
            },
            {
                "path": "shared/python/bad.py",
                "owner": "",
                "reason": "missing owner",
            },
            {
                "path": "shared/python/expired.py",
                "owner": "@platform-architecture",
                "reason": "legacy",
                "expires_on": "2020-01-01",
            },
        ]
    }

    active, invalid = module.build_exception_index(config)

    assert "shared/python/engine_core/engine_loaders.py" in active
    assert any("missing required fields" in msg for msg in invalid)
    assert any("Expired exception" in msg for msg in invalid)


def test_check_rules_respects_rules_path_and_exceptions(tmp_path):
    module = _load_module()

    project_root = tmp_path
    src_root = project_root / "src"
    target_file = src_root / "shared" / "python" / "example.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("import src.engines.loader\n", encoding="utf-8")

    rules_dir = project_root / "scripts" / "config"
    rules_dir.mkdir(parents=True)
    rules_path = Path("scripts/config/test_rules.json")
    (project_root / rules_path).write_text(
        """
{
  "rules": [
    {
      "source_dir": "shared/python",
      "forbidden_prefixes": ["src.engines"],
      "description": "shared -> engines"
    }
  ],
  "exceptions": []
}
""".strip(),
        encoding="utf-8",
    )

    violations = module.check_rules(src_root, rules_path)
    assert violations

    (project_root / rules_path).write_text(
        """
{
  "rules": [
    {
      "source_dir": "shared/python",
      "forbidden_prefixes": ["src.engines"],
      "description": "shared -> engines"
    }
  ],
  "exceptions": [
    {
      "path": "shared/python/example.py",
      "owner": "@platform-architecture",
      "reason": "temporary shim",
      "expires_on": "2027-01-31"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    violations = module.check_rules(src_root, rules_path)
    assert violations == []
