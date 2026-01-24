"""CLI utility to display physics parameters."""

import sys
from pathlib import Path
from src.shared.python.path_utils import get_repo_root, get_src_root


# Add project root to path (script is in scripts/ directory)
root = get_src_root()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "shared" / "python"))

from physics_parameters import get_registry  # noqa: E402


def main() -> None:
    """Display physics parameters."""
    registry = get_registry()
    print(registry.get_summary())


if __name__ == "__main__":
    main()
