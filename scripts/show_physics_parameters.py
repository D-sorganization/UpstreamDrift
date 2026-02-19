"""CLI utility to display physics parameters."""

import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# Ensure the project root is on sys.path so ``src`` is importable.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.shared.python.physics.physics_parameters import get_registry  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    """Display physics parameters."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    registry = get_registry()
    logger.info(registry.get_summary())


if __name__ == "__main__":
    main()
