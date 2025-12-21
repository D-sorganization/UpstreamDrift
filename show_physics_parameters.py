"""CLI utility to display physics parameters."""

import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent / "shared" / "python"))

from physics_parameters import get_registry


def main():
    """Display physics parameters."""
    registry = get_registry()
    print(registry.get_summary())


if __name__ == "__main__":
    main()
