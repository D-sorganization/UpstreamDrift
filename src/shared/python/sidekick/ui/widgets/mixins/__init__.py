"""Widget mixin classes package.

Exports mixin classes that augment base Qt widgets with additional operations
such as data-processor pipeline integration.
"""

from .data_processor_ops import DataProcessorOpsMixin

__all__ = [
    "DataProcessorOpsMixin",
]
