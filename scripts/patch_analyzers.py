"""Patch perturbation analyzers to inherit shared base helpers."""

from __future__ import annotations

from pathlib import Path

ENGINE_NAMES: tuple[str, ...] = ("drake", "mujoco", "myosuite", "opensim", "pinocchio")
BASE_IMPORT = """from src.shared.python.perturbation.analyzer_base import (
    ComparisonReport,
    MANDATORY_METRICS,
    PerturbationAnalyzerBase,
)
"""


def resolve_repo_root() -> Path:
    """Return the repository root for this maintenance script."""
    return Path(__file__).resolve().parent.parent


def iter_analyzer_files(repo_root: Path) -> list[Path]:
    """Return maintained engine analyzer files that can be patched."""
    engines_dir = repo_root / "src/engines/physics_engines"
    return [
        engines_dir / engine / "python/perturbation/analyzer.py"
        for engine in ENGINE_NAMES
        if (engines_dir / engine / "python/perturbation/analyzer.py").exists()
    ]


def patch_imports(content: str) -> str:
    """Insert the shared analyzer-base import block when it is missing."""
    if "PerturbationAnalyzerBase" in content:
        return content
    statistics_import = "from src.shared.python.perturbation.statistics import ("
    if statistics_import in content:
        return content.replace(
            statistics_import, BASE_IMPORT + "\n" + statistics_import
        )
    return content.replace("import numpy as np", "import numpy as np\n" + BASE_IMPORT)


def patch_class_inheritance(content: str, engine: str) -> str:
    """Make the engine analyzer inherit from the shared base class."""
    class_names = (
        f"{engine.capitalize()}PerturbationAnalyzer",
        engine.replace("myosuite", "MyoSuite")
        .replace("mujoco", "MuJoCo")
        .replace("opensim", "OpenSim")
        .replace("drake", "Drake")
        .replace("pinocchio", "Pinocchio")
        + "PerturbationAnalyzer",
    )
    for class_name in class_names:
        plain = f"class {class_name}:"
        inherited = f"class {class_name}(PerturbationAnalyzerBase):"
        if plain in content:
            return content.replace(plain, inherited)
    return content


def remove_redundant_comparison_report(content: str) -> str:
    """Drop the local ComparisonReport dataclass when present."""
    marker = "@dataclass\nclass ComparisonReport:"
    if marker not in content:
        return content
    prefix, suffix = content.split(marker, maxsplit=1)
    next_block = suffix.find("# ---")
    if next_block == -1:
        return content
    return prefix + suffix[next_block:]


def remove_redundant_metrics_tuple(content: str) -> str:
    """Drop the local MANDATORY_METRICS constant when present."""
    marker = "MANDATORY_METRICS: tuple[str, ...] = ("
    if marker not in content:
        return content
    prefix, suffix = content.split(marker, maxsplit=1)
    end_idx = suffix.find(")")
    if end_idx == -1:
        return content
    return prefix + suffix[end_idx + 1 :]


def remove_redundant_methods(content: str) -> str:
    """Remove methods that now live on the shared base class."""
    methods_to_remove = (
        "    def perturb_torque(",
        "    def run_batch(",
        "    def compare_profiles(",
    )
    patched = content
    for method_prefix in methods_to_remove:
        start_idx = patched.find(method_prefix)
        if start_idx == -1:
            continue
        next_method = patched.find("\n    def ", start_idx + 10)
        next_block = patched.find("\n# ---", start_idx + 10)
        next_idx = min(
            [index for index in (next_method, next_block) if index != -1],
            default=-1,
        )
        if next_idx != -1:
            patched = patched[:start_idx] + patched[next_idx + 1 :]
    return patched


def patch_analyzer(content: str, engine: str) -> str:
    """Apply the inheritance and de-duplication patch sequence."""
    patched = patch_imports(content)
    patched = patch_class_inheritance(patched, engine)
    patched = remove_redundant_comparison_report(patched)
    patched = remove_redundant_metrics_tuple(patched)
    return remove_redundant_methods(patched)


def main() -> None:
    """Patch all supported perturbation analyzer files in place."""
    repo_root = resolve_repo_root()
    for analyzer_file in iter_analyzer_files(repo_root):
        content = analyzer_file.read_text(encoding="utf-8")
        engine = analyzer_file.parts[-4]
        patched = patch_analyzer(content, engine)
        if patched != content:
            analyzer_file.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
