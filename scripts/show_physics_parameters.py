"""CLI utility to display physics parameters."""

import sys
from pathlib import Path

# Add project root to path (script is in scripts/ directory)
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "shared" / "python"))

from physics_parameters import get_registry  # noqa: E402


def main() -> None:
    """Display physics parameters."""
    registry = get_registry()
    print(registry.get_summary())


if __name__ == "__main__":
    main()
