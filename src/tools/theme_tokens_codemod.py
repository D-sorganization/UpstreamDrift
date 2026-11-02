"""Theme tokens codemod tool (issue #14252).

Exposes the theme tokens codemod under src/tools for repository-wide tool discovery.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_SCRIPTS = _REPO_ROOT / "ui" / "scripts"
if str(_UI_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_UI_SCRIPTS))

from theme_tokens_codemod import (  # type: ignore[import-not-found]
    TOKEN_REPLACEMENTS as TOKEN_REPLACEMENTS,
    apply_token_replacements as apply_token_replacements,
    count_palette_classes as count_palette_classes,
    main as main,
    process_file as process_file,
)

if __name__ == "__main__":
    sys.exit(main())
