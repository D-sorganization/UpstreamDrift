"""ADR-0046 Stage 2 (G2) wave-1 preconditions for the six lowest-risk modules.

Wave 1 of ADR-0046 Stage 2 re-points `dispersion`, `multivariate`, `trends`,
`comparison`, `schema`, and `treatment` at the canonical launch-monitor layer
vendored from Tools (`vendor/ud-tools/src/shared/python/launch_monitor/`,
Tools#4899-#4900, pinned by UD#9400) and retires UpstreamDrift's private
copies. Two things must hold before a single module may be deleted, and this
file measures both.

**Precondition 1 - the twins are identical (passing).** Stage 2 is a pure
import re-point: "behaviour cannot change" is only true while the vendored
module and the UpstreamDrift module are the same program. This file proves it
structurally, comparing parsed syntax trees rather than bytes so that the
port's added docstrings, its added ``__all__``, and its 88-column rewraps do
not mask a real edit. Nothing else on either side is allowed to drift: if this
fails, the re-point is no longer a no-op and the divergence must be measured
(ADR-0046 G0) before anything moves.

**Precondition 2 - the canonical name is reachable (now met).** Until
2026-09-02 ``shared.python.launch_monitor`` did *not* resolve to the vendored
package here. It resolved to UpstreamDrift's own
``src/shared/python/launch_monitor``, because both were regular packages on
``shared.python.__path__`` and the UpstreamDrift entry preceded the vendor
entry, so the import rewrite ADR-0048's port order prescribes
(``src.<ud path>.X`` -> ``shared.python.launch_monitor.X``) was
self-referential: it resolved back to the module it was supposed to replace,
and would have failed outright once that module was deleted. ADR-0048's
"Stage 2 Blocker (G2)" Option 1 cleared that by moving UpstreamDrift's
transitional copy out of the ``shared.python`` namespace to
``src/tools/launch_monitor_model/``, beside the workbench that consumes it.

The second test below now pins the *resolved* state rather than the blocker:
``shared.python.launch_monitor`` must resolve inside ``vendor/ud-tools``. It
is the provenance probe every wave of Stage 2 depends on — if a UD package
ever re-enters that namespace it goes red before a single retirement can be
mistaken for a no-op. See ADR-0048, "Stage 2 Blocker (G2)".
"""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UD_PACKAGE = _REPO_ROOT / "src" / "tools" / "launch_monitor_model"
_VENDORED_PACKAGE = (
    _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python" / "launch_monitor"
)

# ADR-0046 Stage 2 wave 1: the six modules ADR-0048 orders first (P1-P6),
# every one of them a tier-0 leaf with no intra-package dependency.
WAVE_1_MODULES = (
    "comparison",
    "dispersion",
    "multivariate",
    "schema",
    "treatment",
    "trends",
)

# ADR-0048 P3 renamed this symbol in the canonical layer, deliberately with no
# back-compat alias, because `rate_of_closure` exports the same name for a
# different estimand. The rename is the one intended difference between the
# two `trends` modules; it is normalised here so that every *other* difference
# still fails the comparison.
_TRENDS_RENAME = ("TemporalTrendResult", "TrendResult")

_MISSING_VENDOR_HINT = (
    f"The vendored Tools tree is missing at {_VENDORED_PACKAGE}. Run "
    "`git submodule update --init vendor/ud-tools` to materialise it. In CI "
    "this is a hard failure: a parity gate that silently skips reports green "
    "while the two copies drift, which is the exact failure ADR-0046 exists "
    "to prevent."
)


def _require_vendored_package() -> Path:
    """Return the vendored canonical package, or fail closed in CI."""
    if _VENDORED_PACKAGE.is_dir():
        return _VENDORED_PACKAGE
    if os.environ.get("CI"):
        raise AssertionError(_MISSING_VENDOR_HINT)
    pytest.skip(_MISSING_VENDOR_HINT)


def _normalised_tree(source: str) -> str:
    """Return a dumped AST with docstrings and ``__all__`` removed.

    The port added a module docstring to every module, a class/function
    docstring to some, and an ``__all__`` to all six. None of that is
    executable behaviour, and keeping it in the comparison would make the gate
    fail on documentation edits while still passing on a real logic change.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        )
    ]
    return ast.dump(tree)


@pytest.mark.parametrize("module_name", WAVE_1_MODULES)
def test_wave_1_module_is_identical_to_its_canonical_twin(module_name: str) -> None:
    """The UD copy and the vendored canonical copy are the same program.

    This is what makes ADR-0046 Stage 2 wave 1 a pure import re-point rather
    than a behaviour change. It must keep holding until the UD copy is
    retired: a fix landed on one side only is precisely the silent divergence
    ADR-0046 was accepted to end.
    """
    vendored_package = _require_vendored_package()

    ud_source = (_UD_PACKAGE / f"{module_name}.py").read_text(encoding="utf-8")
    vendored_source = (vendored_package / f"{module_name}.py").read_text(
        encoding="utf-8"
    )
    if module_name == "trends":
        canonical_name, ud_name = _TRENDS_RENAME
        vendored_source = vendored_source.replace(canonical_name, ud_name)

    assert _normalised_tree(ud_source) == _normalised_tree(vendored_source), (
        f"launch_monitor/{module_name}.py has diverged from its canonical twin "
        f"at {vendored_package.relative_to(_REPO_ROOT)}/{module_name}.py. "
        "ADR-0046 Stage 2 re-points consumers at the canonical copy on the "
        "premise that the two are the same program, so this divergence must "
        "be measured with an ADR-0046 G0 gate and resolved in Tools before "
        "wave 1 can proceed. Do not silence this by editing either copy to "
        "match; land the fix in Tools and bump the pin."
    )


def test_canonical_import_path_resolves_into_the_vendored_tools_layer() -> None:
    """``shared.python.launch_monitor`` must be the vendored canonical package.

    This is the provenance probe for every wave of ADR-0046 Stage 2. The
    re-point ADR-0048's port order prescribes is only a re-point if the
    canonical name resolves somewhere other than UpstreamDrift's own tree;
    while both packages sat on ``shared.python.__path__`` it was a
    self-referential alias, because the first entry that carries an
    ``__init__.py`` wins outright and ``src/shared/python`` precedes
    ``vendor/ud-tools/src/shared/python``.

    ADR-0048 "Stage 2 Blocker (G2)" Option 1 resolved that by moving the
    UpstreamDrift copy to ``src/tools/launch_monitor_model/``. This test keeps
    it resolved: if any UpstreamDrift package named ``launch_monitor`` ever
    reappears under ``src/shared/python/``, the retirements that follow would
    silently re-point at UpstreamDrift code while claiming to consume Tools,
    and that regression fails here first.
    """
    vendored_package = _require_vendored_package()

    spec = importlib.util.find_spec("shared.python.launch_monitor")
    assert spec is not None and spec.origin is not None, (
        "shared.python.launch_monitor did not resolve at all. Stage 2 "
        "consumers import through this name, so it must resolve to the "
        "vendored canonical package. Check that vendor/ud-tools is "
        "materialised."
    )

    resolved = Path(spec.origin).resolve()
    assert resolved == (vendored_package / "__init__.py").resolve(), (
        f"shared.python.launch_monitor resolves to {resolved}, not to the "
        f"vendored canonical package at "
        f"{(vendored_package / '__init__.py')}. A UpstreamDrift package has "
        "re-entered the shared.python namespace and is shadowing the "
        "canonical layer again — see ADR-0048, 'Stage 2 Blocker (G2)', and "
        "scripts/config/shadow_modules.yaml."
    )
