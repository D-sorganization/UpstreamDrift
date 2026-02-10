"""Backward compatibility shim - module moved to data_io.path_utils."""

import sys as _sys

from .data_io import path_utils as _real_module  # noqa: E402
from .data_io.path_utils import (  # noqa: F401
    ensure_directory,
    find_file_in_parents,
    get_data_dir,
    get_docs_dir,
    get_drake_python_root,
    get_engines_dir,
    get_mujoco_python_root,
    get_output_dir,
    get_pinocchio_python_root,
    get_relative_path,
    get_repo_root,
    get_shared_dir,
    get_shared_python_root,
    get_simscape_model_path,
    get_src_root,
    get_tests_root,
    logger,
    setup_import_paths,
)

_sys.modules[__name__] = _real_module
