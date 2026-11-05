#!/usr/bin/env python3
"""Check config/secret separation and migration upgrade/rollback guarantees (#14274)."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("check_config_secret_separation")

REPO_ROOT = Path(__file__).resolve().parents[2]

SECRET_KEYWORDS: tuple[str, ...] = (
    "secret",
    "password",
    "token",
    "credential",
    "api_key",
)

ALLOWED_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "",
        "change_me_in_production",
        "changeme",
        "change_me",
        "generate_a_random_string_here",
        "generate_another_random_string_here",
        "unsafe-no-secret-key-set-authentication-will-fail",
        "admin",
        "password",
    }
)


def check_ignore_rules(repo_root: Path) -> list[str]:
    """Verify that .gitignore and .dockerignore exclude .env and .env.*."""
    errors: list[str] = []
    for ignore_name in (".gitignore", ".dockerignore"):
        ignore_path = repo_root / ignore_name
        if not ignore_path.is_file():
            errors.append(f"Missing ignore file: {ignore_name}")
            continue
        text = ignore_path.read_text(encoding="utf-8")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if ".env" not in lines:
            errors.append(f"{ignore_name} missing '.env' exclusion")
        if ".env.*" not in lines and "*.env" not in lines:
            errors.append(f"{ignore_name} missing '.env.*' wildcard exclusion")
    return errors


def check_env_example_secrets(repo_root: Path) -> list[str]:
    """Verify .env.example contains only non-sensitive placeholder credentials."""
    env_example = repo_root / ".env.example"
    if not env_example.is_file():
        return ["Missing .env.example template"]
    errors: list[str] = []
    for line in env_example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip("'\"")
        key_lower = key.lower()
        if any(kw in key_lower for kw in SECRET_KEYWORDS):
            val_norm = val.lower()
            if val_norm not in ALLOWED_PLACEHOLDERS and not val_norm.startswith(
                "change_me"
            ):
                errors.append(
                    f".env.example contains suspicious non-placeholder value for secret key '{key}'"
                )
    return errors


def check_migration_rollback_guarantees(repo_root: Path) -> list[str]:
    """Verify all Alembic database migrations define reversible upgrade and downgrade paths."""
    mig_dir = repo_root / "src" / "api" / "migrations" / "versions"
    if not mig_dir.is_dir():
        return []
    errors: list[str] = []
    for f in sorted(mig_dir.glob("*.py")):
        if f.name.startswith("__"):
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (SyntaxError, OSError) as exc:
            errors.append(f"Could not parse migration {f.name}: {exc}")
            continue

        fn_names: set[str] = set()
        has_nontrivial_downgrade = False
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                fn_names.add(node.name)
                if node.name == "downgrade":
                    # Check body is not just pass or docstring
                    statements = [
                        stmt
                        for stmt in node.body
                        if not (
                            isinstance(stmt, ast.Expr)
                            and isinstance(stmt.value, ast.Constant)
                        )
                        and not isinstance(stmt, ast.Pass)
                    ]
                    if statements:
                        has_nontrivial_downgrade = True

        if "upgrade" not in fn_names:
            errors.append(f"Migration {f.name} missing upgrade() function")
        if "downgrade" not in fn_names:
            errors.append(
                f"Migration {f.name} missing downgrade() function (violates rollback guarantee)"
            )
        elif not has_nontrivial_downgrade:
            errors.append(
                f"Migration {f.name} has empty downgrade() function (cannot be safely rolled back)"
            )
    return errors


def check_typed_settings_secrets(repo_root: Path) -> list[str]:
    """Verify that settings classes do not define hardcoded default secrets."""
    settings_dir = repo_root / "src" / "shared" / "python" / "config"
    if not settings_dir.is_dir():
        return []
    errors: list[str] = []
    for py_file in sorted(settings_dir.glob("*.py")):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except (SyntaxError, OSError) as exc:
            errors.append(f"Could not parse {py_file.name}: {exc}")
            continue

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.AnnAssign) or not isinstance(
                    item.target, ast.Name
                ):
                    continue
                var_name = item.target.id
                var_lower = var_name.lower()
                if not any(kw in var_lower for kw in SECRET_KEYWORDS):
                    continue
                # If value is a string literal, flag it as hardcoded secret
                val = item.value
                if (
                    isinstance(val, ast.Constant)
                    and isinstance(val.value, str)
                    and val.value
                ):
                    errors.append(
                        f"Hardcoded secret default in {py_file.name}:{node.name}.{var_name}"
                    )
                elif (
                    isinstance(val, ast.Call)
                    and isinstance(val.func, ast.Name)
                    and val.func.id == "Field"
                ):
                    for kw in val.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, str) and kw.value.value:
                                errors.append(
                                    f"Hardcoded secret Field default in {py_file.name}:{node.name}.{var_name}"
                                )
    return errors


def run_config_secret_checks(repo_root: Path | None = None) -> dict[str, Any]:
    """Execute all config/secret separation and rollback checks."""
    root = (repo_root or REPO_ROOT).resolve()
    all_errors: list[str] = []

    all_errors.extend(check_ignore_rules(root))
    all_errors.extend(check_env_example_secrets(root))
    all_errors.extend(check_migration_rollback_guarantees(root))
    all_errors.extend(check_typed_settings_secrets(root))

    status = "passed" if not all_errors else "failed"
    return {
        "status": status,
        "summary": {"error_count": len(all_errors), "repo_root": str(root)},
        "errors": all_errors,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for config/secret separation and upgrade/rollback check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_config_secret_checks(args.repo_root)

    if args.json:
        print(json.dumps(report, indent=2))
    elif report["status"] == "passed":
        logger.info("Config/secret separation and rollback guarantees PASSED")
    else:
        logger.error(
            "Config/secret separation FAILED with %d errors:",
            report["summary"]["error_count"],
        )
        for err in report["errors"]:
            logger.error("  - %s", err)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
