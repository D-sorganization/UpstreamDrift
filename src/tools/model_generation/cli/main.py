"""
Command-line interface for model_generation package.

Provides CLI access to URDF generation, conversion, editing, and library features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate URDF from parameters or preset."""
    from model_generation.builders.parametric_builder import ParametricBuilder

    builder = ParametricBuilder(robot_name=args.name)

    # Apply parameters
    if args.height:
        builder.set_height(args.height)
    if args.mass:
        builder.set_mass(args.mass)
    if args.proportions:
        # Parse proportions as JSON
        try:
            proportions = json.loads(args.proportions)
            builder.set_proportions(**proportions)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid proportions JSON: {e}")
            return 1

    # Add humanoid segments
    if args.humanoid:
        builder.add_humanoid_segments()

    # Build
    result = builder.build()

    if not result.success:
        logger.error("Build failed:")
        for error in result.errors:
            logger.error(f"  - {error}")
        return 1

    # Output
    urdf_string = result.to_urdf()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(urdf_string)
        logger.info(f"Wrote URDF to {output_path}")
    else:
        print(urdf_string)

    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert between model formats."""
    source_path = Path(args.input)
    if not source_path.exists():
        logger.error(f"Input file not found: {source_path}")
        return 1

    output_path = Path(args.output) if args.output else None
    suffix = source_path.suffix.lower()

    # Determine conversion type
    if args.from_format == "auto":
        if suffix in (".slx", ".mdl"):
            args.from_format = "simscape"
        elif suffix == ".xml" and args.to_format == "urdf":
            args.from_format = "mjcf"
        elif suffix == ".urdf":
            args.from_format = "urdf"

    try:
        if args.from_format == "simscape":
            from model_generation.converters.simscape import (
                ConversionConfig,
                SimscapeToURDFConverter,
            )

            config = ConversionConfig(robot_name=args.name)
            converter = SimscapeToURDFConverter(config)
            result = converter.convert(source_path, output_path)

            if not result.success:
                logger.error("Conversion failed:")
                for error in result.errors:
                    logger.error(f"  - {error}")
                return 1

            for warning in result.warnings:
                logger.warning(warning)

            if not output_path:
                print(result.urdf_string)

            logger.info(
                f"Converted {len(result.links)} links, {len(result.joints)} joints"
            )

        elif args.from_format == "mjcf" and args.to_format == "urdf":
            from model_generation.converters.mjcf_converter import MJCFConverter

            converter = MJCFConverter()
            urdf_string = converter.mjcf_to_urdf(source_path, output_path)

            if not output_path:
                print(urdf_string)

        elif args.from_format == "urdf" and args.to_format == "mjcf":
            from model_generation.converters.mjcf_converter import MJCFConverter

            converter = MJCFConverter()
            mjcf_string = converter.urdf_to_mjcf(source_path, output_path)

            if not output_path:
                print(mjcf_string)

        else:
            logger.error(
                f"Unsupported conversion: {args.from_format} -> {args.to_format}"
            )
            return 1

    except Exception as e:
        logger.error(f"Conversion error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a URDF file."""
    from model_generation.editor.text_editor import URDFTextEditor, ValidationSeverity

    source_path = Path(args.input)
    if not source_path.exists():
        logger.error(f"File not found: {source_path}")
        return 1

    editor = URDFTextEditor()
    editor.load_file(source_path)

    messages = editor.validate()

    # Filter by severity
    if not args.show_info:
        messages = [m for m in messages if m.severity != ValidationSeverity.INFO]
    if args.errors_only:
        messages = [m for m in messages if m.severity == ValidationSeverity.ERROR]

    # Output
    if args.json:
        output = {
            "file": str(source_path),
            "valid": not any(m.severity == ValidationSeverity.ERROR for m in messages),
            "messages": [
                {
                    "severity": m.severity.value,
                    "line": m.line,
                    "column": m.column,
                    "message": m.message,
                    "element": m.element,
                }
                for m in messages
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if messages:
            for msg in messages:
                print(str(msg))
        else:
            print(f"OK: {source_path}")

    # Return code
    has_errors = any(m.severity == ValidationSeverity.ERROR for m in messages)
    return 1 if has_errors else 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Show differences between URDF files."""
    from model_generation.editor.text_editor import URDFTextEditor

    file_a = Path(args.file_a)
    file_b = Path(args.file_b)

    if not file_a.exists():
        logger.error(f"File not found: {file_a}")
        return 1
    if not file_b.exists():
        logger.error(f"File not found: {file_b}")
        return 1

    editor = URDFTextEditor()
    content_a = file_a.read_text()
    content_b = file_b.read_text()

    editor.load_string(content_a)
    diff_result = editor.get_diff_with_string(content_b)

    if args.json:
        output = {
            "file_a": str(file_a),
            "file_b": str(file_b),
            "has_changes": diff_result.has_changes,
            "additions": diff_result.additions,
            "deletions": diff_result.deletions,
            "hunks": len(diff_result.hunks),
        }
        print(json.dumps(output, indent=2))
    elif args.side_by_side:
        side_by_side = editor.get_side_by_side_diff(content_a, content_b)
        for left, right, change_type in side_by_side:
            if change_type == "equal":
                print(f"  {left or '':<40} | {right or ''}")
            elif change_type == "delete":
                print(f"- {left or '':<40} |")
            elif change_type == "insert":
                print(f"  {'':<40} | + {right or ''}")
            elif change_type == "replace":
                print(f"! {left or '':<40} | ! {right or ''}")
    else:
        print(diff_result.unified_diff)

    return 0 if not diff_result.has_changes or not args.fail_on_diff else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a URDF model."""
    from model_generation.converters.urdf_parser import URDFParser

    source_path = Path(args.input)
    if not source_path.exists():
        logger.error(f"File not found: {source_path}")
        return 1

    parser = URDFParser()
    model = parser.parse(source_path)

    # Calculate statistics
    total_mass = sum(link.inertia.mass for link in model.links)
    joint_types = {}
    for j in model.joints:
        jt = j.joint_type.value
        joint_types[jt] = joint_types.get(jt, 0) + 1

    root = model.get_root_link()

    if args.json:
        output = {
            "name": model.name,
            "source": str(source_path),
            "links": len(model.links),
            "joints": len(model.joints),
            "materials": len(model.materials),
            "total_mass": total_mass,
            "root_link": root.name if root else None,
            "joint_types": joint_types,
            "link_names": [link.name for link in model.links],
            "joint_names": [j.name for j in model.joints],
        }
        if model.warnings:
            output["warnings"] = model.warnings
        print(json.dumps(output, indent=2))
    else:
        print(f"Model: {model.name}")
        print(f"Source: {source_path}")
        print(f"Links: {len(model.links)}")
        print(f"Joints: {len(model.joints)}")
        print(f"Materials: {len(model.materials)}")
        print(f"Total Mass: {total_mass:.3f} kg")
        print(f"Root Link: {root.name if root else 'N/A'}")
        print(f"Joint Types: {joint_types}")

        if args.verbose:
            print("\nLinks:")
            for link in model.links:
                print(f"  - {link.name} (mass: {link.inertia.mass:.3f} kg)")
            print("\nJoints:")
            for joint in model.joints:
                print(
                    f"  - {joint.name}: {joint.parent} -> {joint.child} ({joint.joint_type.value})"
                )

        if model.warnings:
            print("\nWarnings:")
            for w in model.warnings:
                print(f"  - {w}")

    return 0


def cmd_library_list(args: argparse.Namespace) -> int:
    """List models in the library."""
    from model_generation.library import ModelLibrary

    library = ModelLibrary()

    # Apply filters
    models = library.list_models(
        category=args.category,
        source=args.source,
        search=args.search,
    )

    if args.json:
        output = {
            "count": len(models),
            "models": [
                {
                    "id": m.model_id,
                    "name": m.name,
                    "category": m.category.value,
                    "source": m.source.value if m.source else None,
                    "tags": m.tags,
                }
                for m in models
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        if models:
            print(f"Found {len(models)} models:\n")
            for model in models:
                source = f"[{model.source.value}]" if model.source else ""
                print(f"  {model.model_id:<30} {model.category.value:<12} {source}")
                if args.verbose:
                    print(f"    Path: {model.urdf_path}")
                    if model.tags:
                        print(f"    Tags: {', '.join(model.tags)}")
        else:
            print("No models found")

    return 0


def cmd_library_add(args: argparse.Namespace) -> int:
    """Add a model to the library."""
    from model_generation.library import ModelCategory, ModelLibrary

    library = ModelLibrary()
    source_path = Path(args.input)

    if not source_path.exists():
        logger.error(f"File not found: {source_path}")
        return 1

    # Parse category
    category = None
    if args.category:
        try:
            category = ModelCategory(args.category)
        except ValueError:
            logger.error(f"Invalid category: {args.category}")
            return 1

    # Parse tags
    tags = args.tags.split(",") if args.tags else []

    entry = library.add_local_model(
        urdf_path=source_path,
        name=args.name,
        category=category,
        tags=tags,
    )

    if entry:
        logger.info(f"Added model: {entry.model_id}")
        return 0
    else:
        logger.error("Failed to add model")
        return 1


def cmd_library_download(args: argparse.Namespace) -> int:
    """Download a model from repository."""
    from model_generation.library import ModelLibrary

    library = ModelLibrary()
    model = library.load_model(args.model_id, force_download=args.force)

    if model:
        logger.info(f"Downloaded model: {args.model_id}")
        if args.output:
            urdf_string = model.to_urdf()
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(urdf_string)
            logger.info(f"Wrote to {output_path}")
        return 0
    else:
        logger.error(f"Failed to download model: {args.model_id}")
        return 1


def cmd_edit_compose(args: argparse.Namespace) -> int:
    """Compose a model from multiple sources."""
    from model_generation.editor import FrankensteinEditor

    editor = FrankensteinEditor()

    # Load source models
    for source_spec in args.sources:
        parts = source_spec.split(":", 1)
        if len(parts) == 2:
            model_id, path = parts
        else:
            path = parts[0]
            model_id = Path(path).stem

        try:
            editor.load_model(model_id, path, read_only=True)
            logger.info(f"Loaded source: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return 1

    # Create target model
    editor.create_model("output", args.name or "composed_robot")

    # Process operations
    for op in args.operations:
        parts = op.split(":")
        if len(parts) < 2:
            logger.error(f"Invalid operation: {op}")
            continue

        op_type = parts[0]

        if op_type == "copy":
            # copy:source_model:link_name
            if len(parts) >= 3:
                source_model, link_name = parts[1], parts[2]
                if editor.copy_subtree(source_model, link_name):
                    logger.info(f"Copied subtree: {source_model}/{link_name}")
                else:
                    logger.warning(f"Failed to copy: {source_model}/{link_name}")

        elif op_type == "paste":
            # paste:attach_to[:prefix]
            attach_to = parts[1]
            prefix = parts[2] if len(parts) > 2 else ""
            created = editor.paste("output", attach_to=attach_to, prefix=prefix)
            if created:
                logger.info(f"Pasted {len(created)} links to {attach_to}")

        elif op_type == "delete":
            # delete:link_name
            link_name = parts[1]
            if editor.delete_subtree("output", link_name):
                logger.info(f"Deleted subtree: {link_name}")

    # Export
    output_path = Path(args.output)
    editor.export_model("output", output_path)
    logger.info(f"Wrote composed model to {output_path}")

    return 0


def cmd_inertia(args: argparse.Namespace) -> int:
    """Calculate inertia for a shape."""
    from model_generation.core.types import Inertia

    mass = args.mass

    if args.shape == "box":
        if len(args.dimensions) != 3:
            logger.error("Box requires 3 dimensions: x y z")
            return 1
        inertia = Inertia.from_box(mass, *args.dimensions)

    elif args.shape == "cylinder":
        if len(args.dimensions) != 2:
            logger.error("Cylinder requires 2 dimensions: radius length")
            return 1
        inertia = Inertia.from_cylinder(mass, args.dimensions[0], args.dimensions[1])

    elif args.shape == "sphere":
        if len(args.dimensions) != 1:
            logger.error("Sphere requires 1 dimension: radius")
            return 1
        inertia = Inertia.from_sphere(mass, args.dimensions[0])

    elif args.shape == "capsule":
        if len(args.dimensions) != 2:
            logger.error("Capsule requires 2 dimensions: radius length")
            return 1
        inertia = Inertia.from_capsule(mass, args.dimensions[0], args.dimensions[1])

    else:
        logger.error(f"Unknown shape: {args.shape}")
        return 1

    if args.json:
        output = {
            "shape": args.shape,
            "mass": mass,
            "dimensions": args.dimensions,
            "inertia": {
                "ixx": inertia.ixx,
                "iyy": inertia.iyy,
                "izz": inertia.izz,
                "ixy": inertia.ixy,
                "ixz": inertia.ixz,
                "iyz": inertia.iyz,
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Shape: {args.shape}")
        print(f"Mass: {mass} kg")
        print(f"Dimensions: {args.dimensions}")
        print("\nInertia tensor:")
        print(f"  ixx: {inertia.ixx:.6g}")
        print(f"  iyy: {inertia.iyy:.6g}")
        print(f"  izz: {inertia.izz:.6g}")
        print(f"  ixy: {inertia.ixy:.6g}")
        print(f"  ixz: {inertia.ixz:.6g}")
        print(f"  iyz: {inertia.iyz:.6g}")

        print("\nURDF element:")
        print(f"  {inertia.to_urdf_string()}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="model-gen",
        description="URDF Model Generation and Manipulation Tools",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output"
    )
    parser.add_argument("--version", action="version", version="model-gen 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser(
        "generate", aliases=["gen"], help="Generate URDF from parameters"
    )
    gen_parser.add_argument("name", help="Robot name")
    gen_parser.add_argument("-o", "--output", help="Output file path")
    gen_parser.add_argument("--height", type=float, help="Model height in meters")
    gen_parser.add_argument("--mass", type=float, help="Total mass in kg")
    gen_parser.add_argument("--proportions", help="Proportions as JSON")
    gen_parser.add_argument(
        "--humanoid", action="store_true", help="Generate humanoid model"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # Convert command
    conv_parser = subparsers.add_parser(
        "convert", aliases=["conv"], help="Convert between model formats"
    )
    conv_parser.add_argument("input", help="Input file path")
    conv_parser.add_argument("-o", "--output", help="Output file path")
    conv_parser.add_argument(
        "-f",
        "--from-format",
        default="auto",
        choices=["auto", "simscape", "urdf", "mjcf"],
        help="Input format",
    )
    conv_parser.add_argument(
        "-t",
        "--to-format",
        default="urdf",
        choices=["urdf", "mjcf"],
        help="Output format",
    )
    conv_parser.add_argument("-n", "--name", help="Override robot name")
    conv_parser.set_defaults(func=cmd_convert)

    # Validate command
    val_parser = subparsers.add_parser(
        "validate", aliases=["val"], help="Validate URDF file"
    )
    val_parser.add_argument("input", help="URDF file to validate")
    val_parser.add_argument("--json", action="store_true", help="Output as JSON")
    val_parser.add_argument(
        "--errors-only", action="store_true", help="Show only errors"
    )
    val_parser.add_argument(
        "--show-info", action="store_true", help="Show info-level messages"
    )
    val_parser.set_defaults(func=cmd_validate)

    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two URDF files")
    diff_parser.add_argument("file_a", help="First file")
    diff_parser.add_argument("file_b", help="Second file")
    diff_parser.add_argument("--json", action="store_true", help="Output as JSON")
    diff_parser.add_argument(
        "-s", "--side-by-side", action="store_true", help="Side-by-side view"
    )
    diff_parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with error if files differ",
    )
    diff_parser.set_defaults(func=cmd_diff)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("input", help="URDF file")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    info_parser.set_defaults(func=cmd_info)

    # Library commands
    lib_parser = subparsers.add_parser(
        "library", aliases=["lib"], help="Model library operations"
    )
    lib_subparsers = lib_parser.add_subparsers(dest="lib_command")

    # library list
    lib_list = lib_subparsers.add_parser("list", help="List models")
    lib_list.add_argument("-c", "--category", help="Filter by category")
    lib_list.add_argument("-s", "--source", help="Filter by source")
    lib_list.add_argument("--search", help="Search by name")
    lib_list.add_argument("--json", action="store_true", help="Output as JSON")
    lib_list.set_defaults(func=cmd_library_list)

    # library add
    lib_add = lib_subparsers.add_parser("add", help="Add model to library")
    lib_add.add_argument("input", help="URDF file to add")
    lib_add.add_argument("-n", "--name", help="Model name")
    lib_add.add_argument("-c", "--category", help="Category")
    lib_add.add_argument("--tags", help="Comma-separated tags")
    lib_add.set_defaults(func=cmd_library_add)

    # library download
    lib_download = lib_subparsers.add_parser(
        "download", aliases=["dl"], help="Download model from repository"
    )
    lib_download.add_argument("model_id", help="Model ID")
    lib_download.add_argument("-o", "--output", help="Output file path")
    lib_download.add_argument(
        "-f", "--force", action="store_true", help="Force re-download"
    )
    lib_download.set_defaults(func=cmd_library_download)

    # Compose command
    compose_parser = subparsers.add_parser(
        "compose", help="Compose model from multiple sources"
    )
    compose_parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        required=True,
        help="Source models (id:path or path)",
    )
    compose_parser.add_argument("-o", "--output", required=True, help="Output file")
    compose_parser.add_argument("-n", "--name", help="Robot name")
    compose_parser.add_argument(
        "--operations",
        nargs="+",
        default=[],
        help="Operations (copy:model:link, paste:parent, delete:link)",
    )
    compose_parser.set_defaults(func=cmd_edit_compose)

    # Inertia calculator
    inertia_parser = subparsers.add_parser(
        "inertia", help="Calculate inertia for primitive shapes"
    )
    inertia_parser.add_argument(
        "shape",
        choices=["box", "cylinder", "sphere", "capsule"],
        help="Shape type",
    )
    inertia_parser.add_argument("mass", type=float, help="Mass in kg")
    inertia_parser.add_argument(
        "dimensions",
        type=float,
        nargs="+",
        help="Dimensions (shape-dependent)",
    )
    inertia_parser.add_argument("--json", action="store_true", help="Output as JSON")
    inertia_parser.set_defaults(func=cmd_inertia)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(
        args.verbose if hasattr(args, "verbose") else False,
        args.quiet if hasattr(args, "quiet") else False,
    )

    if not args.command:
        parser.print_help()
        return 0

    # Handle library subcommands
    if args.command in ("library", "lib"):
        if not hasattr(args, "lib_command") or not args.lib_command:
            parser.parse_args([args.command, "-h"])
            return 0

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
