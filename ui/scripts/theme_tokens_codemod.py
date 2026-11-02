#!/usr/bin/env python3
"""Codemod to migrate raw Tailwind gray-*/blue-* classes to semantic theme tokens.

Mapped pairs (issue #14252):
- bg-gray-950 / bg-gray-900 -> bg-canvas
- bg-gray-800 -> bg-surface
- bg-gray-700 -> bg-surface-raised
- border-gray-800 / border-gray-700 -> border-token-border
- text-gray-400 -> text-token-muted
- text-gray-500 -> text-token-subtle
- text-gray-100 / text-gray-200 / text-gray-300 -> text-token-text
- bg-blue-600 -> bg-primary
- hover:bg-blue-700 -> hover:bg-primary-hover
"""

from __future__ import annotations

import argparse
import collections.abc
import re
import sys
from pathlib import Path

TOKEN_REPLACEMENTS: list[tuple[str, str]] = [
    # Canvas
    ("bg-gray-950", "bg-canvas"),
    ("bg-gray-900", "bg-canvas"),
    # Surface & Surface Raised
    ("bg-gray-800", "bg-surface"),
    ("bg-gray-700", "bg-surface-raised"),
    ("hover:bg-gray-600", "hover:bg-surface"),
    # Borders
    ("border-gray-800", "border-token-border"),
    ("border-gray-700", "border-token-border"),
    ("border-gray-600", "border-token-border"),
    # Text
    ("text-gray-400", "text-token-muted"),
    ("text-gray-500", "text-token-subtle"),
    ("text-gray-300", "text-token-text"),
    ("text-gray-200", "text-token-text"),
    ("text-gray-100", "text-token-text"),
    # Accent / Primary
    ("bg-blue-600", "bg-primary"),
    ("hover:bg-blue-700", "hover:bg-primary-hover"),
    ("hover:bg-blue-500", "hover:bg-primary-hover"),
    ("focus:ring-blue-500", "focus:ring-primary"),
]

# Build regex pattern for token replacement preserving variants (e.g. hover:, focus:, etc.)
VARIANT_PREFIX = r"(?<=[\s\"'`;{(])((?:[a-zA-Z0-9_\-]+:)*)"
VARIANT_SUFFIX = r"(?=[\s\"'`;})])"


def apply_token_replacements(content: str) -> tuple[str, dict[str, int]]:
    """Apply token mappings to content and return (transformed_content, stats)."""
    stats: dict[str, int] = {}
    modified = content

    for raw, token in TOKEN_REPLACEMENTS:
        # Check if raw has a variant prefix (like hover:bg-blue-700)
        if ":" in raw:
            prefix, base_raw = raw.split(":", 1)
            token_prefix, base_token = (
                token.split(":", 1) if ":" in token else ("", token)
            )
            pattern = re.compile(
                r"(?<=[\s\"'`;{(])((?:[a-zA-Z0-9_\-]+:)*)"
                + re.escape(prefix)
                + r":"
                + re.escape(base_raw)
                + VARIANT_SUFFIX
            )

            def make_repl_prefix(
                tp: str, bt: str
            ) -> collections.abc.Callable[[re.Match[str]], str]:
                def repl(m: re.Match[str]) -> str:
                    full_prefix = m.group(1) + (tp + ":" if tp else "")
                    return full_prefix + bt

                return repl

            repl_func = make_repl_prefix(token_prefix, base_token)
        else:
            pattern = re.compile(VARIANT_PREFIX + re.escape(raw) + VARIANT_SUFFIX)

            def make_repl_plain(
                tok: str,
            ) -> collections.abc.Callable[[re.Match[str]], str]:
                def repl(m: re.Match[str]) -> str:
                    return m.group(1) + tok

                return repl

            repl_func = make_repl_plain(token)

        count = len(pattern.findall(modified))
        if count > 0:
            modified = pattern.sub(repl_func, modified)
            stats[f"{raw} -> {token}"] = count

    return modified, stats


def count_palette_classes(content: str) -> dict[str, int]:
    """Count raw gray-* and blue-* classes in source content."""
    raw_gray_pattern = re.compile(
        r"\b(?:[a-zA-Z0-9_\-]+:)*(?:bg|text|border|ring|divide)-gray-\d+(?:/\d+)?\b"
    )
    raw_blue_pattern = re.compile(
        r"\b(?:[a-zA-Z0-9_\-]+:)*(?:bg|text|border|ring|divide)-blue-\d+(?:/\d+)?\b"
    )

    return {
        "gray": len(raw_gray_pattern.findall(content)),
        "blue": len(raw_blue_pattern.findall(content)),
    }


def process_file(path: Path, write: bool = False) -> tuple[bool, dict[str, int]]:
    """Process a single file, optionally writing changes."""
    content = path.read_text(encoding="utf-8")
    new_content, stats = apply_token_replacements(content)
    changed = new_content != content
    if changed and write:
        path.write_text(new_content, encoding="utf-8")
    return changed, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Codemod raw Tailwind classes to semantic tokens."
    )
    parser.add_argument("paths", nargs="*", help="Files or directories to process")
    parser.add_argument(
        "--write", "-w", action="store_true", help="Write changes to files"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print raw class usage statistics across files",
    )
    args = parser.parse_args()

    target_paths = [Path(p) for p in args.paths] if args.paths else [Path("ui/src")]

    all_files: list[Path] = []
    for p in target_paths:
        if p.is_file():
            all_files.append(p)
        elif p.is_dir():
            for ext in ("*.tsx", "*.ts", "*.jsx", "*.js"):
                all_files.extend(
                    f
                    for f in p.rglob(ext)
                    if not any(
                        part in f.parts
                        for part in (
                            "test",
                            "tests",
                            "__tests__",
                            "node_modules",
                            "dist",
                        )
                    )
                    and not f.name.endswith(
                        (".test.tsx", ".test.ts", ".spec.tsx", ".spec.ts")
                    )
                )

    if args.stats:
        total_gray = 0
        total_blue = 0
        for f in all_files:
            c = count_palette_classes(f.read_text(encoding="utf-8"))
            total_gray += c["gray"]
            total_blue += c["blue"]
        sys.stdout.write(f"Total non-test source files scanned: {len(all_files)}\n")
        sys.stdout.write(f"Raw gray-* classes: {total_gray}\n")
        sys.stdout.write(f"Raw blue-* classes: {total_blue}\n")
        sys.stdout.write(f"Total raw palette classes: {total_gray + total_blue}\n")
        return 0

    total_changed = 0
    total_stats: dict[str, int] = {}
    for f in all_files:
        changed, stats = process_file(f, write=args.write)
        if changed:
            total_changed += 1
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v

    action = "Updated" if args.write else "Would update"
    sys.stdout.write(f"{action} {total_changed} of {len(all_files)} files.\n")
    for k, v in sorted(total_stats.items()):
        sys.stdout.write(f"  {k}: {v}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
