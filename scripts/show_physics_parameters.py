"""CLI utility to display physics parameters."""

import sys
from pathlib import Path

# Add project root to path first (script is in scripts/ directory)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from src.shared.python.path_utils import get_src_root  # noqa: E402

# Add shared python to path for physics_parameters import
root = get_src_root()
sys.path.insert(0, str(root / "shared" / "python"))

from physics_parameters import get_registry  # noqa: E402


def main() -> None:
    """Display physics parameters."""
    registry = get_registry()
    print(registry.get_summary())


if __name__ == "__main__":
    main()
