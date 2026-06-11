#!/usr/bin/env python3
"""LOD (Law of Demeter) lint check for production Python code.

Flags attribute-access chains deeper than 2 levels (i.e. `a.b.c.d` and longer)
that are likely to be genuine LOD violations.

A chain is considered "deep" when more than 2 ``ast.Attribute`` nodes are
stacked on top of an underlying ``ast.Name`` receiver (so ``a.b.c`` is fine
but ``a.b.c.d`` is not). This matches the standard LOD interpretation used
in the project's CODING_STANDARDS.md.

Allow-list rationale (these are library API patterns, not LOD violations):
- Qt signal/slot wiring: ``self.btn.clicked.connect(...)`` — fixed call shape
  imposed by PyQt's signal API. Refactoring would just hide the wiring.
- numpy / scipy / pandas method chains where the second-to-last segment is a
  method that returns a new array (``.copy()``, ``.tolist()``, ``.reshape()``,
  ``.astype()``, ``.flatten()``, ``.T``, ``.add_subplot()``, ``.fig.clear()``).
  These are method chains on library return values, not object-graph
  navigation.
- Logging: ``self.logger.info.something`` style does not actually appear; the
  logger calls in the codebase are well-behaved.

The script exits with code 0 if no violations are found, code 1 otherwise.
When a baseline is supplied, checked-in violations are tolerated and only
new path/chain occurrences fail the check.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import sys
from pathlib import Path

DEFAULT_ROOT = "src"
DEFAULT_BASELINE = "scripts/ci/lod_baseline.txt"

# Final attribute / method names that indicate a library API chain rather
# than navigation through application state.
LIBRARY_API_LEAVES: frozenset[str] = frozenset(
    {
        # Qt signal/slot wiring
        "connect",
        "disconnect",
        "emit",
        # numpy / pandas tail operations
        "tolist",
        "copy",
        "reshape",
        "astype",
        "flatten",
        "ravel",
        "squeeze",
        "transpose",
        "T",
        "item",
        "all",
        "any",
        "sum",
        "mean",
        "max",
        "min",
        "argmax",
        "argmin",
        # matplotlib/figure helpers commonly chained on canvas.fig
        "add_subplot",
        "clear",
        "tight_layout",
        "savefig",
        "draw",
        "draw_idle",
    }
)

# Intermediate segments that indicate this chain reaches into a library
# object (so the *next* call is library API, not object navigation).
LIBRARY_API_INTERMEDIATES: frozenset[str] = frozenset(
    {
        # PyQt signals
        "clicked",
        "toggled",
        "currentTextChanged",
        "currentIndexChanged",
        "valueChanged",
        "textChanged",
        "timeout",
        "returnPressed",
        "stateChanged",
        "triggered",
        "activated",
        "released",
        "pressed",
        "editingFinished",
        # Custom Qt-style signals defined in the codebase
        "gravity_changed",
        "pose_loaded",
        "interpolation_requested",
        # Matplotlib canvas
        "fig",
        "figure",
        "axes",
        "canvas",
    }
)


def _attr_chain(node: ast.AST) -> list[str] | None:
    """Return the dotted-attribute chain for an Attribute/Name expression.

    Returns ``None`` if the chain bottoms out in something other than a Name
    (e.g. a Call, Subscript, or literal) — those cases are filtered separately.
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        parts.reverse()
        return parts
    return None


def _is_library_chain(chain: list[str]) -> bool:
    """Heuristically classify a chain as library-API rather than navigation."""
    if not chain:
        return True
    if chain[-1] in LIBRARY_API_LEAVES:
        return True
    if any(seg in LIBRARY_API_INTERMEDIATES for seg in chain):
        return True
    # Static enum / module access — e.g. QtCore.Qt.Orientation.Horizontal,
    # pin.GeometryType.VISUAL, np.linalg.norm. These are namespaced constants
    # or qualified function references, not object navigation.
    return chain[0] in {
        "QtCore",
        "QtGui",
        "QtWidgets",
        "Qt",
        "matplotlib",
        "pin",
        "pinocchio",
        "np",
        "numpy",
        "pd",
        "pandas",
        "scipy",
    }


class LODVisitor(ast.NodeVisitor):
    """Collect deep attribute chains rooted at a Name node."""

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        # Only consider the *outermost* Attribute of a chain. We detect that by
        # the parent-tracking established in ``check_file``.
        parent = getattr(node, "_parent", None)
        if isinstance(parent, ast.Attribute):
            self.generic_visit(node)
            return

        chain = _attr_chain(node)
        if chain is None:
            self.generic_visit(node)
            return

        # Chain length: the number of attribute hops past the receiver.
        # `a.b` -> chain == ["a","b"] -> 1 hop. `a.b.c.d` -> 3 hops.
        hops = len(chain) - 1
        if hops <= 2:
            self.generic_visit(node)
            return

        if _is_library_chain(chain):
            self.generic_visit(node)
            return

        self.violations.append((node.lineno, ".".join(chain)))
        self.generic_visit(node)


def _annotate_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]


def check_file(path: Path) -> list[tuple[int, str]]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    _annotate_parents(tree)
    visitor = LODVisitor()
    visitor.visit(tree)
    return visitor.violations


def iter_python_files(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*.py")
        if "tests" not in p.parts and "examples" not in p.parts
    ]


Violation = tuple[str, int, str]
BaselineKey = tuple[str, str]


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root.parent).as_posix()
    except ValueError:
        return path.as_posix()


def collect_violations(root: Path) -> tuple[list[Path], list[Violation]]:
    files = iter_python_files(root)
    violations: list[Violation] = []
    for path in sorted(files):
        display_path = _display_path(path, root)
        for lineno, chain in check_file(path):
            violations.append((display_path, lineno, chain))
    return files, violations


def _violation_counts(violations: list[Violation]) -> Counter[BaselineKey]:
    return Counter((path, chain) for path, _lineno, chain in violations)


def load_baseline(path: Path) -> Counter[BaselineKey]:
    counts: Counter[BaselineKey] = Counter()
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}:{lineno}: expected path<TAB>chain<TAB>count")
        rel_path, chain, count_text = parts
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: count must be an integer") from exc
        if count <= 0:
            raise ValueError(f"{path}:{lineno}: count must be positive")
        counts[(rel_path, chain)] += count
    return counts


def write_baseline(path: Path, violations: list[Violation]) -> None:
    counts = _violation_counts(violations)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Law-of-Demeter baseline for issue #7308.",
        "# Owner: @core",
        "# Reduction plan: refactor existing entries opportunistically; this gate",
        "# blocks any path/chain count growth across production src/ Python files.",
        "# Format: <relative path><TAB><attribute chain><TAB><allowed count>",
    ]
    for (rel_path, chain), count in sorted(counts.items()):
        lines.append(f"{rel_path}\t{chain}\t{count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _baseline_excess(
    violations: list[Violation],
    baseline: Counter[BaselineKey],
) -> list[Violation]:
    current = _violation_counts(violations)
    remaining = Counter(
        {
            key: count - baseline.get(key, 0)
            for key, count in current.items()
            if count > baseline.get(key, 0)
        }
    )
    excess: list[Violation] = []
    for violation in violations:
        key = (violation[0], violation[2])
        if remaining[key] <= 0:
            continue
        excess.append(violation)
        remaining[key] -= 1
    return excess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Path to a checked-in path/chain count baseline for no-growth mode.",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write the current findings as a path/chain count baseline and exit.",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Print violations but exit 0 (warning mode).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"check_lod: root path does not exist: {root}", file=sys.stderr)
        return 2

    files, violations = collect_violations(root)

    if args.write_baseline is not None:
        write_baseline(args.write_baseline, violations)
        print(
            "check_lod: wrote baseline "
            f"{args.write_baseline} ({len(_violation_counts(violations))} entries, "
            f"{len(violations)} violation occurrences)"
        )
        return 0

    if args.baseline is not None:
        try:
            baseline = load_baseline(args.baseline)
        except (OSError, ValueError) as exc:
            print(f"check_lod: invalid baseline: {exc}", file=sys.stderr)
            return 2
        excess = _baseline_excess(violations, baseline)
        for rel_path, lineno, chain in excess:
            print(f"{rel_path}:{lineno}: new LOD chain >2 deep: {chain}")
        if not excess:
            reductions = sum((baseline - _violation_counts(violations)).values())
            print(
                f"check_lod: clean no-growth scan ({len(files)} files scanned under "
                f"{root}; {len(violations)} baseline violation occurrences; "
                f"{reductions} reductions)"
            )
            return 0
        print(
            f"check_lod: found {len(excess)} new LOD violation occurrence(s)",
            file=sys.stderr,
        )
        return 0 if args.advisory else 1

    for rel_path, lineno, chain in violations:
        print(f"{rel_path}:{lineno}: LOD chain >2 deep: {chain}")

    if not violations:
        print(f"check_lod: clean ({len(files)} files scanned under {root})")
        return 0

    print(f"check_lod: found {len(violations)} LOD violation(s)", file=sys.stderr)
    return 0 if args.advisory else 1


if __name__ == "__main__":
    sys.exit(main())
