"""Entry point for ``python -m sidekick`` and the ``sidekick`` console script.

Sub-commands
------------
gui          Launch the standalone Sidekick window.
run          Run a named calculator headlessly and emit JSON results.
list         List available headless calculators.

Design notes
------------
- All GUI imports are deferred inside the ``gui`` branch so ``sidekick run``
  and ``sidekick --help`` work without a display or PyQt6 installed.
- ``main()`` is the sole public symbol; the module is not importable as a
  library (it is a CLI shell only).
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sidekick",
        description="Sidekick — UpstreamDrift engineering assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  sidekick gui\n"
            "  sidekick run --calculator wgs_reactor --inputs inputs.json\n"
            "  sidekick list\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="sidekick 0.1.0",
    )
    parser.add_argument(
        "--skip-onboarding",
        action="store_true",
        default=False,
        help="Skip the first-run onboarding dialog (useful for CI smoke tests).",
    )

    sub = parser.add_subparsers(
        dest="command", title="sub-commands", metavar="<command>"
    )

    # gui -----------------------------------------------------------------
    sub.add_parser(
        "gui",
        help="Launch the standalone Sidekick window",
        description="Open the standalone Sidekick desktop application.",
    )

    # run -----------------------------------------------------------------
    run_p = sub.add_parser(
        "run",
        help="Run a calculator headlessly and emit JSON",
        description="Execute a named calculator with JSON inputs and write results.",
    )
    run_p.add_argument(
        "--calculator",
        required=True,
        metavar="NAME",
        help="Calculator name (see 'sidekick list').",
    )
    run_p.add_argument(
        "--inputs",
        required=True,
        metavar="FILE",
        help="Path to a JSON file containing calculator inputs.",
    )
    run_p.add_argument(
        "--output",
        default="-",
        metavar="FILE",
        help="Output file path; '-' writes to stdout (default: -).",
    )

    # list ----------------------------------------------------------------
    sub.add_parser(
        "list",
        help="List available headless calculators",
    )

    return parser


def main() -> None:
    """CLI entry point: ``sidekick [--help] [--version] <command> ...``

    Precondition: none (arguments come from sys.argv).
    Postcondition: exits via SystemExit; never returns normally.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        from sidekick.standalone.runner import run_calculator

        code = run_calculator(args.calculator, args.inputs, args.output)
        sys.exit(code)

    elif args.command == "list":
        from sidekick.standalone.runner import list_calculators

        for name in list_calculators():
            print(name)
        sys.exit(0)

    elif args.command == "gui":
        try:
            from sidekick.standalone.window import launch  # type: ignore[import]

            sys.exit(launch(skip_onboarding=args.skip_onboarding))
        except ImportError as exc:
            print(
                f"GUI unavailable: {exc}\n"
                "Install PyQt6 to use the Sidekick window: pip install sidekick[gui]",
                file=sys.stderr,
            )
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
