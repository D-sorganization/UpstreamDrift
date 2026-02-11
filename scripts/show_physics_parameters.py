"""CLI utility to display physics parameters."""

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

from src.shared.python.physics.physics_parameters import get_registry  # noqa: E402


def main() -> None:
    """Display physics parameters."""
    registry = get_registry()
    print(registry.get_summary())


if __name__ == "__main__":
    main()
