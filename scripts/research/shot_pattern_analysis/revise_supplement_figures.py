"""Stage or finalize a reviewed graphics-only child; never run Monte Carlo/bootstrap.

First create a pin against the explicit parent commit and approved execution JSON.
Stage in a new directory, inspect all revised graphics, then separately finalize
with the hash of an independently written visual-and-contract review. The command
never writes an accepted review or updates the original pipeline receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .supplement_parent_verifier import verify_parent_in_frozen_source
from .supplement_publication import (
    CANDIDATE_RECEIPT,
    create_density_revision,
    finalize_density_revision,
    load_verified_supplement_publication,
    sha256,
)
from .supplement_revision_adapter import render_selected_records
from .supplement_revision_pin import create_parent_pin, load_parent_pin


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    pin = sub.add_parser("pin")
    for flag in ("source-root", "python", "execution", "output"):
        pin.add_argument("--" + flag, type=Path, required=True)
    for flag in ("source-commit", "execution-sha256"):
        pin.add_argument("--" + flag, required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("parent", type=Path)
    stage.add_argument("child", type=Path)
    stage.add_argument("--pin", type=Path, required=True)
    stage.add_argument("--pin-sha256", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("child", type=Path)
    finalize.add_argument("--review", type=Path, required=True)
    finalize.add_argument("--review-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("child", type=Path)
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "pin":
        result = create_parent_pin(
            args.source_root,
            args.python,
            args.source_commit,
            args.execution,
            args.execution_sha256,
            args.output,
        )
    elif args.command == "stage":
        binding = load_parent_pin(args.pin, args.pin_sha256, base=args.child)

        def verifier(parent):
            return verify_parent_in_frozen_source(parent, binding, base=args.child)

        loaded = verifier(args.parent)

        def render(records, child):
            return render_selected_records(loaded, records, child)

        result = create_density_revision(
            args.parent,
            args.child,
            parent_verifier=verifier,
            render_density=render,
            verifier_binding=binding,
        )
    elif args.command == "finalize":
        candidate = json.loads((args.child / CANDIDATE_RECEIPT).read_text())
        binding = candidate["parent_verifier"]
        result = finalize_density_revision(
            args.child,
            review_path=args.review,
            review_sha256=args.review_sha256,
            parent_verifier=lambda parent: verify_parent_in_frozen_source(
                parent, binding, base=args.child
            ),
        )
    else:
        publication = load_verified_supplement_publication(args.child)
        print(json.dumps(publication["publication_identity"], sort_keys=True))
        return
    print(json.dumps({"path": str(result), "sha256": sha256(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
