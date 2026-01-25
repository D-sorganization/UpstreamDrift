"""Centralized I/O utilities for the Golf Modeling Suite.

This module consolidates common file I/O patterns across the codebase,
addressing DRY violations identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.io_utils import (
        load_json,
        save_json,
        load_yaml,
        save_yaml,
        ensure_directory,
    )

    # Load/save JSON
    data = load_json("config.json")
    save_json("output.json", data)

    # Load/save YAML
    config = load_yaml("settings.yaml")
    save_yaml("output.yaml", config)

    # Ensure directory exists
    ensure_directory("output/results")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine_availability import YAML_AVAILABLE
from .error_utils import (
    FileNotFoundIOError,
    FileParseError,
)
from .error_utils import IOError as IOUtilsError

if YAML_AVAILABLE:
    import yaml


# Re-export for backwards compatibility
__all__ = [
    "IOUtilsError",
    "FileNotFoundIOError",
    "FileParseError",
    "ensure_directory",
    "load_json",
    "save_json",
    "load_yaml",
    "save_yaml",
    "read_text",
    "write_text",
    "file_exists",
    "get_file_size",
]


def ensure_directory(path: Path | str, parents: bool = True) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.
        parents: If True, create parent directories as needed.

    Returns:
        The Path object for the directory.

    Example:
        output_dir = ensure_directory("output/results")
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=parents, exist_ok=True)
    return dir_path


def load_json(
    path: Path | str,
    *,
    encoding: str = "utf-8",
    default: Any = None,
    strict: bool = True,
) -> Any:
    """Load data from a JSON file.

    Args:
        path: Path to the JSON file.
        encoding: File encoding.
        default: Default value if file doesn't exist and strict is False.
        strict: If True, raise error when file doesn't exist.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundIOError: If file doesn't exist and strict is True.
        FileParseError: If JSON parsing fails.

    Example:
        config = load_json("config.json")
        settings = load_json("settings.json", default={})
    """
    file_path = Path(path)

    if not file_path.exists():
        if strict:
            raise FileNotFoundIOError(file_path, "read")
        return default

    try:
        with file_path.open("r", encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FileParseError(file_path, "JSON", str(e)) from e


def save_json(
    path: Path | str,
    data: Any,
    *,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    create_parents: bool = True,
    sort_keys: bool = False,
) -> Path:
    """Save data to a JSON file.

    Args:
        path: Path to save the JSON file.
        data: Data to serialize to JSON.
        encoding: File encoding.
        indent: Indentation level for pretty printing.
        ensure_ascii: If True, escape non-ASCII characters.
        create_parents: If True, create parent directories as needed.
        sort_keys: If True, sort dictionary keys.

    Returns:
        The Path object for the saved file.

    Example:
        save_json("output.json", {"key": "value"})
        save_json("data.json", results, indent=4, sort_keys=True)
    """
    file_path = Path(path)

    if create_parents and file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding=encoding) as f:
        json.dump(
            data, f, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys
        )

    return file_path


def load_yaml(
    path: Path | str,
    *,
    encoding: str = "utf-8",
    default: Any = None,
    strict: bool = True,
    loader: Any = None,
) -> Any:
    """Load data from a YAML file.

    Args:
        path: Path to the YAML file.
        encoding: File encoding.
        default: Default value if file doesn't exist and strict is False.
        strict: If True, raise error when file doesn't exist.
        loader: YAML loader to use (default: SafeLoader).

    Returns:
        Parsed YAML data.

    Raises:
        FileNotFoundIOError: If file doesn't exist and strict is True.
        FileParseError: If YAML parsing fails.
        ImportError: If PyYAML is not installed.

    Example:
        config = load_yaml("config.yaml")
        settings = load_yaml("settings.yaml", default={})
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for YAML operations. "
            "Install it with: pip install pyyaml"
        )

    file_path = Path(path)

    if not file_path.exists():
        if strict:
            raise FileNotFoundIOError(file_path, "read")
        return default

    yaml_loader = loader or yaml.SafeLoader

    try:
        with file_path.open("r", encoding=encoding) as f:
            return yaml.load(f, Loader=yaml_loader)
    except yaml.YAMLError as e:
        raise FileParseError(file_path, "YAML", str(e)) from e


def save_yaml(
    path: Path | str,
    data: Any,
    *,
    encoding: str = "utf-8",
    default_flow_style: bool = False,
    allow_unicode: bool = True,
    create_parents: bool = True,
    sort_keys: bool = False,
    dumper: Any = None,
) -> Path:
    """Save data to a YAML file.

    Args:
        path: Path to save the YAML file.
        data: Data to serialize to YAML.
        encoding: File encoding.
        default_flow_style: If True, use flow style for collections.
        allow_unicode: If True, allow unicode characters.
        create_parents: If True, create parent directories as needed.
        sort_keys: If True, sort dictionary keys.
        dumper: YAML dumper to use (default: SafeDumper).

    Returns:
        The Path object for the saved file.

    Raises:
        ImportError: If PyYAML is not installed.

    Example:
        save_yaml("output.yaml", {"key": "value"})
    """
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for YAML operations. "
            "Install it with: pip install pyyaml"
        )

    file_path = Path(path)

    if create_parents and file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_dumper = dumper or yaml.SafeDumper

    with file_path.open("w", encoding=encoding) as f:
        yaml.dump(
            data,
            f,
            Dumper=yaml_dumper,
            default_flow_style=default_flow_style,
            allow_unicode=allow_unicode,
            sort_keys=sort_keys,
        )

    return file_path


def read_text(
    path: Path | str,
    *,
    encoding: str = "utf-8",
    default: str | None = None,
    strict: bool = True,
) -> str:
    """Read text content from a file.

    Args:
        path: Path to the text file.
        encoding: File encoding.
        default: Default value if file doesn't exist and strict is False.
        strict: If True, raise error when file doesn't exist.

    Returns:
        File content as string.

    Example:
        content = read_text("README.md")
    """
    file_path = Path(path)

    if not file_path.exists():
        if strict:
            raise FileNotFoundIOError(file_path, "read")
        return default or ""

    return file_path.read_text(encoding=encoding)


def write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    create_parents: bool = True,
) -> Path:
    """Write text content to a file.

    Args:
        path: Path to save the text file.
        content: Text content to write.
        encoding: File encoding.
        create_parents: If True, create parent directories as needed.

    Returns:
        The Path object for the saved file.

    Example:
        write_text("output.txt", "Hello, World!")
    """
    file_path = Path(path)

    if create_parents and file_path.parent:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(content, encoding=encoding)
    return file_path


def file_exists(path: Path | str) -> bool:
    """Check if a file exists.

    Args:
        path: Path to check.

    Returns:
        True if file exists, False otherwise.
    """
    return Path(path).exists()


def get_file_size(path: Path | str) -> int:
    """Get the size of a file in bytes.

    Args:
        path: Path to the file.

    Returns:
        File size in bytes.

    Raises:
        FileNotFoundIOError: If file doesn't exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundIOError(file_path, "stat")
    return file_path.stat().st_size
