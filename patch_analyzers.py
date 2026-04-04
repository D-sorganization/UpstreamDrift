from pathlib import Path

repo_root = Path(r"C:\Users\diete\Repositories\UpstreamDrift")
engines_dir = repo_root / "src/engines/physics_engines"

engine_names = ["drake", "mujoco", "myosuite", "opensim", "pinocchio"]

for engine in engine_names:
    analyzer_file = engines_dir / engine / "python/perturbation/analyzer.py"
    if not analyzer_file.exists():
        continue

    content = analyzer_file.read_text(encoding="utf-8")

    # Check if already patched
    if "PerturbationAnalyzerBase" in content:
        continue

    # 1. Add the import
    import_statement = "from src.shared.python.perturbation.analyzer_base import (\n    ComparisonReport,\n    MANDATORY_METRICS,\n    PerturbationAnalyzerBase,\n)\n"

    # We will search for metric imports and stick it there
    if "from src.shared.python.perturbation.statistics import (" in content:
        content = content.replace(
            "from src.shared.python.perturbation.statistics import (",
            import_statement
            + "\nfrom src.shared.python.perturbation.statistics import (",
        )
    else:
        # Just put it under generic imports
        content = content.replace(
            "import numpy as np", "import numpy as np\n" + import_statement
        )

    # 2. Inherit from PerturbationAnalyzerBase
    if "class " + engine.capitalize() + "PerturbationAnalyzer:" in content:
        content = content.replace(
            "class " + engine.capitalize() + "PerturbationAnalyzer:",
            "class "
            + engine.capitalize()
            + "PerturbationAnalyzer(PerturbationAnalyzerBase):",
        )
    elif (
        "class "
        + engine.replace("myosuite", "MyoSuite")
        .replace("mujoco", "MuJoCo")
        .replace("opensim", "OpenSim")
        .replace("drake", "Drake")
        .replace("pinocchio", "Pinocchio")
        + "PerturbationAnalyzer:"
        in content
    ):
        content = content.replace(
            f"class {engine.replace('myosuite', 'MyoSuite').replace('mujoco', 'MuJoCo').replace('opensim', 'OpenSim').replace('drake', 'Drake').replace('pinocchio', 'Pinocchio')}PerturbationAnalyzer:",
            f"class {engine.replace('myosuite', 'MyoSuite').replace('mujoco', 'MuJoCo').replace('opensim', 'OpenSim').replace('drake', 'Drake').replace('pinocchio', 'Pinocchio')}PerturbationAnalyzer(PerturbationAnalyzerBase):",
        )

    # 3. Remove Redundant ComparisonReport
    # Split by "@dataclass\nclass ComparisonReport:"
    if "@dataclass\nclass ComparisonReport:" in content:
        parts = content.split("@dataclass\nclass ComparisonReport:")
        # We need to remove until the next '# ---'
        part2 = parts[1]
        next_block = part2.find("# ---")
        if next_block != -1:
            content = parts[0] + part2[next_block:]

    # 4. Remove Redundant MANDATORY_METRICS tuple
    if "MANDATORY_METRICS: tuple[str, ...] =" in content:
        parts = content.split("MANDATORY_METRICS: tuple[str, ...] = (")
        if len(parts) > 1:
            part2 = parts[1]
            end_idx = part2.find(")")
            if end_idx != -1:
                content = parts[0] + part2[end_idx + 1 :]

    # 5. Remove 'def perturb_torque' and 'def run_batch' and 'def compare_profiles'
    # and all their bodies. They span usually correctly formatted.

    methods_to_remove = [
        "    def perturb_torque(",
        "    def run_batch(",
        "    def compare_profiles(",
    ]

    for meth in methods_to_remove:
        start_idx = content.find(meth)
        if start_idx != -1:
            # find next method "    def _"
            next_idx_1 = content.find("\n    def ", start_idx + 10)
            next_idx_2 = content.find("\n# ---", start_idx + 10)
            next_idx = -1
            if next_idx_1 != -1 and next_idx_2 != -1:
                next_idx = min(next_idx_1, next_idx_2)
            elif next_idx_1 != -1:
                next_idx = next_idx_1
            elif next_idx_2 != -1:
                next_idx = next_idx_2

            if next_idx != -1:
                content = content[:start_idx] + content[next_idx + 1 :]

    analyzer_file.write_text(content, encoding="utf-8")
