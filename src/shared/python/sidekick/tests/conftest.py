"""Pytest configuration for upstream_drift_tools tests.

Uses shared path setup from utils.path_helpers.
"""

import sys

print("CONFTEST SYS.PATH:", sys.path)
from utils.path_helpers import ensure_utils_in_path

# Ensure utils is available for test imports
ensure_utils_in_path()
