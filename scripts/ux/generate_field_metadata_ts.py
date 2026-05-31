"""Generate ``ui/src/ux/fieldMetadata.ts`` from the YAML registry.

The YAML at ``configs/ux/field_metadata.yaml`` is the single source of
truth for UX field metadata (DRY).  The React side must not hand-copy
that content; instead this script derives a typed TypeScript module from
the *validated* registry so Python and TS can never drift.

A Vitest round-trip test re-parses the same YAML and asserts the
generated module matches, which fails CI if someone edits the YAML and
forgets to regenerate.

Run::

    python3 scripts/ux/generate_field_metadata_ts.py            # write
    python3 scripts/ux/generate_field_metadata_ts.py --check    # verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.shared.python.ux.field_metadata import load_registry  # noqa: E402

_YAML_PATH = _REPO_ROOT / "configs" / "ux" / "field_metadata.yaml"
_TS_PATH = _REPO_ROOT / "ui" / "src" / "ux" / "fieldMetadata.ts"

_HEADER = """\
// AUTO-GENERATED from configs/ux/field_metadata.yaml — DO NOT EDIT BY HAND.
// Regenerate with: python3 scripts/ux/generate_field_metadata_ts.py
// The YAML is the single source of truth (epic #5968, DRY).

export interface FieldMetadata {
  id: string;
  label: string;
  shortHelp: string;
  longHelp: string;
  units: string | null;
  validRange: [number, number] | string[] | null;
  default: unknown;
  defaultSource: string;
  consumers: string[];
  producers: string[];
  example: string;
}
"""


def _to_camel(payload: dict) -> dict:
    """Map the snake_case YAML keys to the camelCase TS shape."""
    return {
        "id": payload["id"],
        "label": payload["label"],
        "shortHelp": payload["short_help"],
        "longHelp": payload["long_help"],
        "units": payload["units"],
        "validRange": payload["valid_range"],
        "default": payload["default"],
        "defaultSource": payload["default_source"],
        "consumers": payload["consumers"],
        "producers": payload["producers"],
        "example": payload["example"],
    }


def render() -> str:
    """Return the TypeScript source derived from the validated YAML."""
    registry = load_registry(_YAML_PATH)
    records = [_to_camel(fm.to_dict()) for fm in registry.iter_fields()]
    body = json.dumps(records, indent=2, ensure_ascii=False)
    return (
        f"{_HEADER}\n"
        f"export const FIELD_METADATA: FieldMetadata[] = {body};\n\n"
        "export const FIELD_METADATA_BY_ID: Record<string, FieldMetadata> =\n"
        "  Object.fromEntries(FIELD_METADATA.map((f) => [f.id, f]));\n\n"
        "export function getFieldMetadata(id: string): FieldMetadata {\n"
        "  const fm = FIELD_METADATA_BY_ID[id];\n"
        "  if (fm === undefined) {\n"
        "    throw new Error(`unknown field id: ${id}`);\n"
        "  }\n"
        "  return fm;\n"
        "}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the generated file is up to date (exit 1 if stale)",
    )
    args = parser.parse_args(argv)
    rendered = render()
    if args.check:
        current = _TS_PATH.read_text(encoding="utf-8") if _TS_PATH.exists() else ""
        if current != rendered:
            sys.stderr.write(
                f"{_TS_PATH} is stale — run scripts/ux/generate_field_metadata_ts.py\n"
            )
            return 1
        return 0
    _TS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TS_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
