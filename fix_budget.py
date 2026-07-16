from pathlib import Path
import re

filepath = "src/engines/Simscape_Multibody_Models/3D_Golf_Model/MachineLearning/validate_club_calibration.py"
content = Path(filepath).read_text()

# Replaces the `validate` signature, leaving `**kwargs`
content = re.sub(
    r"def validate\([\s\S]*?\) -> dict\[str, Any\]:",
    """def validate(
    measured_target_csv: Path,
    calibrated_target_csv: Path,
    sim_csv: Path,
    output_dir: Path,
    run_label: str = "club_calibration",
    transform_json: Path | None = None,
    impact_time: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    impact_window_s = kwargs.get("impact_window_s", 0.02)
    write_plots = kwargs.get("write_plots", True)
    min_finite_samples = kwargs.get("min_finite_samples", 3)
    poor_impact_threshold = kwargs.get("poor_impact_threshold", 0.05)
    anisotropy_threshold = kwargs.get("anisotropy_threshold", 10.0)
    extreme_scale_min = kwargs.get("extreme_scale_min", 0.2)
    extreme_scale_max = kwargs.get("extreme_scale_max", 5.0)""",
    content
)


# Fix the main function call using regex correctly
new_main = """    validate(
        measured_target_csv=args.measured_target_csv,
        calibrated_target_csv=args.calibrated_target_csv,
        sim_csv=args.sim_csv,
        output_dir=args.output_dir,
        run_label=args.run_label,
        transform_json=args.transform_json,
        impact_time=args.impact_time,
        impact_window_s=args.impact_window_s,
        write_plots=not args.no_plots,
        min_finite_samples=args.min_finite_samples,
        poor_impact_threshold=args.poor_impact_threshold,
        anisotropy_threshold=args.anisotropy_threshold,
        extreme_scale_min=args.extreme_scale_min,
        extreme_scale_max=args.extreme_scale_max,
    )"""

content = re.sub(
    r"    validate\(\n\s*measured_target_csv=args\.measured_target_csv,[\s\S]*?extreme_scale_max=args\.extreme_scale_max,\n    \)",
    new_main,
    content
)

Path(filepath).write_text(content)
