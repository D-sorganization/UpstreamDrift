#!/usr/bin/env python3
"""
Wave 2 Manifest Validator

Scans src/engines/* and src/shared/python/* directories and validates
that all modules are properly documented in MANIFEST.md.

Usage:
    python3 scripts/wave2_manifest_validator.py [--update]
    python3 scripts/wave2_manifest_validator.py --check-only
"""

import argparse
import json
import re
import sys
from pathlib import Path


class ManifestValidator:
    """Validate and update module manifest."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.src_path = repo_root / "src"
        self.manifest_path = repo_root / "MANIFEST.md"
        self.modules_on_disk: dict[str, list[str]] = {}
        self.modules_in_manifest: dict[str, list[str]] = {}

    def scan_engines(self) -> dict[str, list[str]]:
        """Scan src/engines/* for Python modules."""
        engines_path = self.src_path / "engines"
        if not engines_path.exists():
            print(f"Warning: {engines_path} not found")
            return {}

        modules = {}
        for engine_dir in engines_path.iterdir():
            if not engine_dir.is_dir() or engine_dir.name.startswith("__"):
                continue

            engine_modules = []
            for py_file in engine_dir.glob("**/*.py"):
                if py_file.name.startswith("__"):
                    continue
                rel_path = py_file.relative_to(engine_dir)
                engine_modules.append(str(rel_path))

            if engine_modules:
                modules[f"engines/{engine_dir.name}"] = sorted(engine_modules)

        return modules

    def scan_shared(self) -> dict[str, list[str]]:
        """Scan src/shared/python/* for modules."""
        shared_path = self.src_path / "shared" / "python"
        if not shared_path.exists():
            print(f"Warning: {shared_path} not found")
            return {}

        modules = {}
        for item in shared_path.iterdir():
            if item.name.startswith("__") or item.name.startswith("."):
                continue

            if item.is_file() and item.suffix == ".py":
                modules[f"shared/python/{item.name}"] = [item.name]
            elif item.is_dir():
                sub_modules = []
                for py_file in item.glob("**/*.py"):
                    if py_file.name.startswith("__"):
                        continue
                    rel_path = py_file.relative_to(item)
                    sub_modules.append(str(rel_path))

                if sub_modules:
                    modules[f"shared/python/{item.name}"] = sorted(sub_modules)

        return modules

    def scan_all_modules(self) -> dict[str, list[str]]:
        """Scan all module directories."""
        all_modules = {}
        all_modules.update(self.scan_engines())
        all_modules.update(self.scan_shared())
        self.modules_on_disk = all_modules
        return all_modules

    def parse_manifest(self) -> dict[str, list[str]]:
        """Parse MANIFEST.md to extract documented modules."""
        if not self.manifest_path.exists():
            print(f"Warning: {self.manifest_path} not found")
            return {}

        content = self.manifest_path.read_text()
        modules = {}

        # Look for sections like "## shared/python/contracts"
        section_pattern = r"^## ([a-z0-9/_-]+)"
        current_section = None

        for line in content.split("\n"):
            section_match = re.match(section_pattern, line, re.IGNORECASE)
            if section_match:
                current_section = section_match.group(1)
                modules[current_section] = []
            elif current_section and line.strip().startswith("- "):
                # Extract module name from bullet point
                module_name = line.strip()[2:].strip()
                # Remove markdown links and extra formatting
                module_name = re.sub(r"\[.*?\]\(.*?\)", "", module_name)
                module_name = module_name.split("(")[0].strip()
                if module_name:
                    modules[current_section].append(module_name)

        self.modules_in_manifest = modules
        return modules

    def validate(self) -> tuple[bool, list[str], list[str]]:
        """Validate manifest against actual modules.

        Returns:
            (is_valid, missing_from_manifest, orphaned_in_manifest)
        """
        missing = []
        orphaned = []

        # Check for modules on disk not in manifest
        for module_path in self.modules_on_disk.keys():
            if module_path not in self.modules_in_manifest:
                missing.append(module_path)

        # Check for modules in manifest not on disk
        for module_path in self.modules_in_manifest.keys():
            if module_path not in self.modules_on_disk:
                orphaned.append(module_path)

        is_valid = len(missing) == 0 and len(orphaned) == 0
        return is_valid, missing, orphaned

    def generate_report(self) -> str:
        """Generate validation report."""
        self.scan_all_modules()
        self.parse_manifest()
        is_valid, missing, orphaned = self.validate()

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("WAVE 2 MANIFEST VALIDATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Repository: {self.repo_root.name}")
        report_lines.append(f"Scan time: {Path(__file__).stat().st_mtime}")
        report_lines.append("")

        # Summary
        total_on_disk = sum(len(v) for v in self.modules_on_disk.values())
        total_in_manifest = sum(len(v) for v in self.modules_in_manifest.values())
        report_lines.append("SUMMARY")
        report_lines.append(f"  Modules on disk: {len(self.modules_on_disk)}")
        report_lines.append(f"  Total on-disk files: {total_on_disk}")
        report_lines.append(f"  Modules in manifest: {len(self.modules_in_manifest)}")
        report_lines.append(f"  Total manifest entries: {total_in_manifest}")
        report_lines.append(f"  Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
        report_lines.append("")

        # Modules on disk
        report_lines.append("MODULES ON DISK")
        for module, files in sorted(self.modules_on_disk.items()):
            status = "✓" if module in self.modules_in_manifest else "✗ MISSING"
            report_lines.append(f"  {status} {module} ({len(files)} files)")

        report_lines.append("")

        # Missing from manifest
        if missing:
            report_lines.append("MISSING FROM MANIFEST (Action required)")
            for module in sorted(missing):
                report_lines.append(f"  - {module}")
            report_lines.append("")

        # Orphaned in manifest
        if orphaned:
            report_lines.append("ORPHANED IN MANIFEST (Stale entries)")
            for module in sorted(orphaned):
                report_lines.append(f"  - {module}")
            report_lines.append("")

        report_lines.append("=" * 70)

        return "\n".join(report_lines)

    def generate_manifest_sections(self) -> str:
        """Generate MANIFEST.md sections from scan."""
        self.scan_all_modules()

        sections = []
        sections.append("# UpstreamDrift Module Manifest")
        sections.append("")
        sections.append("**Auto-generated from source scan. Last updated: see git log.**")
        sections.append("")
        sections.append("## Engine Wrappers")
        sections.append("")

        # Engines
        for category in ["engines"]:
            category_modules = {
                k: v for k, v in self.modules_on_disk.items() if k.startswith(category)
            }
            for module, files in sorted(category_modules.items()):
                sections.append(f"### {module}")
                sections.append("")
                for file in sorted(files):
                    sections.append(f"- {file}")
                sections.append("")

        sections.append("## Shared Utilities")
        sections.append("")

        # Shared utilities
        for category in ["shared"]:
            category_modules = {
                k: v for k, v in self.modules_on_disk.items() if k.startswith(category)
            }
            for module, files in sorted(category_modules.items()):
                sections.append(f"### {module}")
                sections.append("")
                for file in sorted(files):
                    sections.append(f"- {file}")
                sections.append("")

        return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and update UpstreamDrift module manifest"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update MANIFEST.md (requires --force for safety)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force manifest update without confirmation"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check, don't update",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    validator = ManifestValidator(repo_root)

    # Generate and print report
    report = validator.generate_report()
    print(report)

    # Validate
    validator.scan_all_modules()
    validator.parse_manifest()
    is_valid, missing, orphaned = validator.validate()

    # JSON output if requested
    if args.json:
        result = {
            "valid": is_valid,
            "modules_on_disk": validator.modules_on_disk,
            "modules_in_manifest": validator.modules_in_manifest,
            "missing": missing,
            "orphaned": orphaned,
        }
        print(json.dumps(result, indent=2))

    # Update manifest if requested
    if args.update and not is_valid:
        if not args.force:
            print("\nUse --force to apply update")
            sys.exit(1)

        manifest_content = validator.generate_manifest_sections()
        validator.manifest_path.write_text(manifest_content)
        print(f"\nManifest updated: {validator.manifest_path}")

    # Exit code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
