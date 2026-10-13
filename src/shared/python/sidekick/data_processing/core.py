"""Compatibility Alias for the Pinned Tools Data Processing Core."""

import sys
import warnings

from src.shared.python.sidekick_data_processing_bridge import load_provider_module

warnings.warn(
    "Legacy sidekick.data_processing.core resolves to the pinned Tools provider.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = load_provider_module("core")
