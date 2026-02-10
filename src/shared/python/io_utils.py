"""Backward compatibility shim - module moved to data_io.io_utils."""

import sys as _sys

from .data_io import io_utils as _real_module  # noqa: E402
from .data_io.io_utils import (  # noqa: F401
    FileNotFoundIOError,
    FileParseError,
    IOUtilsError,
    ensure_directory,
    file_exists,
    get_file_size,
    load_json,
    load_yaml,
    read_text,
    save_json,
    save_yaml,
    write_text,
)

_sys.modules[__name__] = _real_module
