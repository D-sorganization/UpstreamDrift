"""Theme tokens codemod tool (issue #14252).

Exposes the theme tokens codemod under src/tools for repository-wide tool discovery.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "ui" / "scripts" / "theme_tokens_codemod.py"

_spec = importlib.util.spec_from_file_location("theme_tokens_codemod", _SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TOKEN_REPLACEMENTS = _mod.TOKEN_REPLACEMENTS
apply_token_replacements = _mod.apply_token_replacements
count_palette_classes = _mod.count_palette_classes
main = _mod.main
process_file = _mod.process_file

if __name__ == "__main__":
    sys.exit(main())
