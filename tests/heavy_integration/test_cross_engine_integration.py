import pytest
from src.shared.python.logging_pkg.logging_config import get_logger


logger = get_logger(__name__)


pytestmark = pytest.mark.live_simulation
