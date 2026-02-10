"""Backward compatibility shim - module moved to data_io.dataset_generator."""

import sys as _sys

from .data_io import dataset_generator as _real_module  # noqa: E402
from .data_io.dataset_generator import (  # noqa: F401
    ControlProfile,
    DatasetGenerator,
    GeneratorConfig,
    ParameterRange,
    SimulationSample,
    TrainingDataset,
    logger,
)

_sys.modules[__name__] = _real_module
