"""CLI utility to display physics parameters."""

import sys
from pathlib import Path

# Add shared to path (script is in scripts/ directory)
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

from physics_parameters import get_registry


def main() -> None:
    """Display physics parameters."""
    registry = get_registry()
    print(registry.get_summary())


if __name__ == "__main__":
    main()
