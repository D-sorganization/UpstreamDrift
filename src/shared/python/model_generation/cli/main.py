# mypy: ignore-errors
# ruff: noqa: E501
# TRACKED_TASK: see #2310 — architecture debt extraction schedule

"""
Command-line interface for model_generation package.

Provides CLI access to URDF generation, conversion, editing, and library features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TypeVar, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)
_EnumT = TypeVar("_EnumT", bound=Enum)


def _parse_enum_arg(
    enum_type: type[_EnumT], value: str | None, label: str
) -> _EnumT | None:
    """Parse an optional CLI enum value with a consistent error message."""
    if value is None:
        return None
    try:
        return enum_type(value)
    except ValueError:
        logger.error(f"Invalid {label}: {value}")
        return None


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level."""
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate URDF from parameters or preset."""
    from shared.python.model_generation.builders.parametric_builder import (
        ParametricBuilder,
    )

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
        logger.info(urdf_string)

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
            from shared.python.model_generation.converters.simscape import (
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
                logger.info(result.urdf_string)

            logger.info(
                f"Converted {len(result.links)} links, {len(result.joints)} joints"
            )

        elif args.from_format == "mjcf" and args.to_format == "urdf":
            from shared.python.model_generation.converters.mjcf_converter import (
                MJCFConverter,
            )

            converter = MJCFConverter()
            urdf_string = converter.mjcf_to_urdf(source_path, output_path)

            if not output_path:
                logger.info(urdf_string)

        elif args.from_format == "urdf" and args.to_format == "mjcf":
            from shared.python.model_generation.converters.mjcf_converter import (
                MJCFConverter,
            )

            converter = MJCFConverter()
            mjcf_string = converter.urdf_to_mjcf(source_path, output_path)

            if not output_path:
                logger.info(mjcf_string)

        else:
            logger.error(
                f"Unsupported conversion: {args.from_format} -> {args.to_format}"
            )
            return 1

    except ImportError as e:
        logger.error(f"Conversion error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a URDF file."""
    from shared.python.model_generation.editor.text_editor import (
        URDFTextEditor,
        ValidationSeverity,
    )

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
        logger.info(json.dumps(output, indent=2))
    else:
        if messages:
            for msg in messages:
                logger.info(str(msg))
        else:
            logger.info(f"OK: {source_path}")

    # Return code
    has_errors = any(m.severity == ValidationSeverity.ERROR for m in messages)
    return 1 if has_errors else 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Show differences between URDF files."""
    from shared.python.model_generation.editor.text_editor import URDFTextEditor

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
        logger.info(json.dumps(output, indent=2))
    elif args.side_by_side:
        side_by_side = editor.get_side_by_side_diff(content_a, content_b)
        for left, right, change_type in side_by_side:
            if change_type == "equal":
                logger.info(f"  {left or '':<40} | {right or ''}")
            elif change_type == "delete":
                logger.info(f"- {left or '':<40} |")
            elif change_type == "insert":
                logger.info(f"  {'':<40} | + {right or ''}")
            elif change_type == "replace":
                logger.info(f"! {left or '':<40} | ! {right or ''}")
    else:
        logger.info(diff_result.unified_diff)

    return 0 if not diff_result.has_changes or not args.fail_on_diff else 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a URDF model."""
    from shared.python.model_generation.converters.urdf_parser import URDFParser

    source_path = Path(args.input)
    if not source_path.exists():
        logger.error(f"File not found: {source_path}")
        return 1

    parser = URDFParser()
    model = parser.parse(source_path)

    # Calculate statistics
    total_mass = sum(link.inertia.mass for link in model.links)
    joint_types: dict[str, int] = {}
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
        logger.info(json.dumps(output, indent=2))
    else:
        logger.info(f"Model: {model.name}")
        logger.info(f"Source: {source_path}")
        logger.info(f"Links: {len(model.links)}")
        logger.info(f"Joints: {len(model.joints)}")
        logger.info(f"Materials: {len(model.materials)}")
        logger.info(f"Total Mass: {total_mass:.3f} kg")
        logger.info(f"Root Link: {root.name if root else 'N/A'}")
        logger.info(f"Joint Types: {joint_types}")

        if args.verbose:
            logger.info("\nLinks:")
            for link in model.links:
                logger.info(f"  - {link.name} (mass: {link.inertia.mass:.3f} kg)")
            logger.info("\nJoints:")
            for joint in model.joints:
                logger.info(
                    f"  - {joint.name}: {joint.parent} -> {joint.child} ({joint.joint_type.value})"
                )

        if model.warnings:
            logger.warning("\nWarnings:")
            for w in model.warnings:
                logger.info(f"  - {w}")

    return 0


def cmd_library_list(args: argparse.Namespace) -> int:
    """List models in the library."""
    from shared.python.model_generation.library import (
        ModelCategory,
        ModelLibrary,
        RepositorySource,
    )

    library = ModelLibrary()
    category = _parse_enum_arg(ModelCategory, args.category, "category")
    if args.category and category is None:
        return 1
    source = _parse_enum_arg(RepositorySource, args.source, "source")
    if args.source and source is None:
        return 1

    # Apply filters
    models = library.list_models(
        category=category,
        source=source,
        search=args.search,
    )

    if args.json:
        output = {
            "count": len(models),
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "category": m.category.value,
                    "source": m.source.value if m.source else None,
                    "tags": m.tags,
                }
                for m in models
            ],
        }
        logger.info(json.dumps(output, indent=2))
    else:
        if models:
            logger.info(f"Found {len(models)} models:\n")
            for model in models:
                source = f"[{model.source.value}]" if model.source else ""
                logger.info(f"  {model.id:<30} {model.category.value:<12} {source}")
                if args.verbose:
                    logger.info(f"    Path: {model.urdf_path}")
                    if model.tags:
                        logger.info(f"    Tags: {', '.join(model.tags)}")
        else:
            logger.info("No models found")

    return 0


def cmd_library_add(args: argparse.Namespace) -> int:
    """Add a model to the library."""
    from shared.python.model_generation.library import ModelCategory, ModelLibrary

    library = ModelLibrary()
    source_path = Path(args.input)

    if not source_path.exists():
        logger.error(f"File not found: {source_path}")
        return 1

    # Parse category
    category = ModelCategory.OTHER
    if args.category:
        parsed_category = _parse_enum_arg(ModelCategory, args.category, "category")
        if parsed_category is None:
            return 1
        category = parsed_category

    # Parse tags
    tags = (
        [tag.strip() for tag in args.tags.split(",") if tag.strip()]
        if args.tags
        else []
    )

    entry = library.add_local_model(
        urdf_path=source_path,
        name=args.name,
        category=category,
        tags=tags,
    )

    if entry:
        logger.info(f"Added model: {entry.id}")
        return 0
    logger.error("Failed to add model")
    return 1


def cmd_library_download(args: argparse.Namespace) -> int:
    """Download a model from repository."""
    from shared.python.model_generation.library import ModelLibrary

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
    logger.error(f"Failed to download model: {args.model_id}")
    return 1


def cmd_library_import_github(args: argparse.Namespace) -> int:
    """Import models from GitHub."""
    from shared.python.model_generation.library import GitHubImporter

    importer = GitHubImporter()

    # Handle popular libraries
    if args.popular:
        urls = [
            f"https://github.com/{repo}" for repo in GitHubImporter.POPULAR_REPOSITORIES
        ]
        results = importer.import_from_urls(urls)

    elif args.url:
        results = importer.import_from_urls([args.url])

    elif args.query:
        results = importer.import_from_search(
            query=args.query,
            min_stars=args.min_stars,
            max_results=args.limit,
            dry_run=args.dry_run,
        )
    else:
        logger.error("Must specify --query, --url, or --popular")
        return 1

    # Print results
    if args.json:
        output = [
            {
                "url": r.source_url,
                "status": r.status,
                "model_id": r.model_id,
                "error": r.error,
                "name": r.name,
            }
            for r in results
        ]
        logger.info(json.dumps(output, indent=2))
    else:
        for r in results:
            status_icon = (
                "✓" if r.status == "success" else "✗" if r.status == "failed" else "?"
            )
            logger.info(f"{status_icon} {r.name or r.source_url}: {r.status}")
            if r.description:
                logger.info(f"  {r.description}")
            if r.error:
                logger.error(f"  Error: {r.error}")

    return 0


def cmd_edit_compose(args: argparse.Namespace) -> int:
    """Compose a model from multiple sources."""
    from shared.python.model_generation.editor import FrankensteinEditor

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
        except (OSError, ValueError, KeyError) as e:
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
    from shared.python.model_generation.core.types import Inertia

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
        logger.info(json.dumps(output, indent=2))
    else:
        logger.info(f"Shape: {args.shape}")
        logger.info(f"Mass: {mass} kg")
        logger.info(f"Dimensions: {args.dimensions}")
        logger.info("\nInertia tensor:")
        logger.info(f"  ixx: {inertia.ixx:.6g}")
        logger.info(f"  iyy: {inertia.iyy:.6g}")
        logger.info(f"  izz: {inertia.izz:.6g}")
        logger.info(f"  ixy: {inertia.ixy:.6g}")
        logger.info(f"  ixz: {inertia.ixz:.6g}")
        logger.info(f"  iyz: {inertia.iyz:.6g}")

        logger.info("\nURDF element:")
        logger.info(f"  {inertia.to_urdf_string()}")

    return 0


def _add_core_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register core CLI subcommands (generate, convert, validate, diff, info)."""
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

    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("input", help="URDF file")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")
    info_parser.set_defaults(func=cmd_info)


def _add_library_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register library management subcommands."""
    lib_parser = subparsers.add_parser(
        "library", aliases=["lib"], help="Model library operations"
    )
    lib_subparsers = lib_parser.add_subparsers(dest="lib_command")

    lib_list = lib_subparsers.add_parser("list", help="List models")
    lib_list.add_argument("-c", "--category", help="Filter by category")
    lib_list.add_argument("-s", "--source", help="Filter by source")
    lib_list.add_argument("--search", help="Search by name")
    lib_list.add_argument("--json", action="store_true", help="Output as JSON")
    lib_list.set_defaults(func=cmd_library_list)

    lib_add = lib_subparsers.add_parser("add", help="Add model to library")
    lib_add.add_argument("input", help="URDF file to add")
    lib_add.add_argument("-n", "--name", help="Model name")
    lib_add.add_argument("-c", "--category", help="Category")
    lib_add.add_argument("--tags", help="Comma-separated tags")
    lib_add.set_defaults(func=cmd_library_add)

    lib_download = lib_subparsers.add_parser(
        "download", aliases=["dl"], help="Download model from repository"
    )
    lib_download.add_argument("model_id", help="Model ID")
    lib_download.add_argument("-o", "--output", help="Output file path")
    lib_download.add_argument(
        "-f", "--force", action="store_true", help="Force re-download"
    )
    lib_download.set_defaults(func=cmd_library_download)

    lib_import = lib_subparsers.add_parser(
        "import-github", aliases=["igh"], help="Import from GitHub"
    )
    lib_import.add_argument("--query", help="Search query")
    lib_import.add_argument("--url", help="GitHub URL")
    lib_import.add_argument(
        "--popular", action="store_true", help="Import from popular libraries"
    )
    lib_import.add_argument("--min-stars", type=int, default=10, help="Minimum stars")
    lib_import.add_argument("--limit", type=int, default=10, help="Max results")
    lib_import.add_argument(
        "--dry-run", action="store_true", help="Search without importing"
    )
    lib_import.add_argument("--json", action="store_true", help="Output as JSON")
    lib_import.set_defaults(func=cmd_library_import_github)


def _add_utility_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register utility subcommands (compose, inertia)."""
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
    _add_core_subparsers(subparsers)
    _add_library_subparsers(subparsers)
    _add_utility_subparsers(subparsers)

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
    if args.command in ("library", "lib") and (
        not hasattr(args, "lib_command") or not args.lib_command
    ):
        parser.parse_args([args.command, "-h"])
        return 0

    if hasattr(args, "func"):
        handler = cast(Callable[[argparse.Namespace], int], args.func)
        return handler(args)
    parser.print_help()
    return 0


class _CallableMainModule(types.ModuleType):
    def __call__(self, argv: list[str] | None = None) -> int:
        return main(argv)


sys.modules[__name__].__class__ = _CallableMainModule


if __name__ == "__main__":
    sys.exit(main())
