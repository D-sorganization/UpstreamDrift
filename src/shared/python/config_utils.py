"""Configuration utilities for eliminating configuration loading duplication.

This module provides reusable configuration loading and validation patterns.

Usage:
    from src.shared.python.config_utils import (
        load_json_config,
        load_yaml_config,
        save_json_config,
        ConfigLoader,
    )

    # Load configuration
    config = load_json_config("config.json", default={})

    # Save configuration
    save_json_config("config.json", config)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from src.shared.python.error_decorators import log_errors
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@log_errors("Failed to load JSON configuration", reraise=False, default_return={})
def load_json_config(
    path: str | Path,
    default: dict[str, Any] | None = None,
    create_if_missing: bool = False,
) -> dict[str, Any]:
    """Load JSON configuration file with error handling.

    Args:
        path: Path to JSON file
        default: Default configuration if file doesn't exist or fails to load
        create_if_missing: Create file with default config if missing

    Returns:
        Configuration dictionary

    Example:
        config = load_json_config("settings.json", default={"theme": "dark"})
    """
    path_obj = Path(path)

    if not path_obj.exists():
        if default is not None:
            if create_if_missing:
                save_json_config(path, default)
            return default.copy()
        return {}

    with open(path_obj, encoding="utf-8") as f:
        config: dict[str, Any] = json.load(f)

    logger.debug(f"Loaded configuration from {path}")
    return config


@log_errors("Failed to save JSON configuration", reraise=False)
def save_json_config(
    path: str | Path,
    config: dict[str, Any],
    indent: int = 2,
    create_dirs: bool = True,
) -> bool:
    """Save configuration to JSON file with error handling.

    Args:
        path: Path to JSON file
        config: Configuration dictionary
        indent: JSON indentation level
        create_dirs: Create parent directories if they don't exist

    Returns:
        True if successful, False otherwise

    Example:
        save_json_config("settings.json", {"theme": "dark"})
    """
    path_obj = Path(path)

    if create_dirs:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=indent)

    logger.debug(f"Saved configuration to {path}")
    return True


@log_errors("Failed to load YAML configuration", reraise=False, default_return={})
def load_yaml_config(
    path: str | Path,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML configuration file with error handling.

    Args:
        path: Path to YAML file
        default: Default configuration if file doesn't exist or fails to load

    Returns:
        Configuration dictionary

    Example:
        config = load_yaml_config("settings.yaml", default={"theme": "dark"})
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, cannot load YAML config")
        return default.copy() if default else {}

    path_obj = Path(path)

    if not path_obj.exists():
        if default is not None:
            return default.copy()
        return {}

    with open(path_obj, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Loaded YAML configuration from {path}")
    return config or {}


@log_errors("Failed to save YAML configuration", reraise=False)
def save_yaml_config(
    path: str | Path,
    config: dict[str, Any],
    create_dirs: bool = True,
) -> bool:
    """Save configuration to YAML file with error handling.

    Args:
        path: Path to YAML file
        config: Configuration dictionary
        create_dirs: Create parent directories if they don't exist

    Returns:
        True if successful, False otherwise

    Example:
        save_yaml_config("settings.yaml", {"theme": "dark"})
    """
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML not installed, cannot save YAML config")
        return False

    path_obj = Path(path)

    if create_dirs:
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    logger.debug(f"Saved YAML configuration to {path}")
    return True


class ConfigLoader:
    """Configuration loader with caching and validation.

    Example:
        loader = ConfigLoader("config.json")
        config = loader.load(default={"theme": "dark"})
        loader.save(config)
    """

    def __init__(self, path: str | Path, format: str = "json"):
        """Initialize configuration loader.

        Args:
            path: Path to configuration file
            format: Configuration format ("json" or "yaml")
        """
        self.path = Path(path)
        self.format = format.lower()
        self._cache: dict[str, Any] | None = None

    def load(
        self,
        default: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Load configuration.

        Args:
            default: Default configuration
            use_cache: Use cached configuration if available

        Returns:
            Configuration dictionary
        """
        if use_cache and self._cache is not None:
            return self._cache.copy()

        if self.format == "json":
            config = load_json_config(self.path, default)
        elif self.format == "yaml":
            config = load_yaml_config(self.path, default)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        self._cache = config
        return config.copy()

    def save(self, config: dict[str, Any]) -> bool:
        """Save configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if successful, False otherwise
        """
        if self.format == "json":
            success = save_json_config(self.path, config)
        elif self.format == "yaml":
            success = save_yaml_config(self.path, config)
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        if success:
            self._cache = config.copy()

        return success

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., "ui.theme")
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            theme = loader.get("ui.theme", "dark")
        """
        config = self.load()

        # Support dot notation
        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Returns:
            True if successful, False otherwise

        Example:
            loader.set("ui.theme", "light")
        """
        config = self.load()

        # Support dot notation
        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

        return self.save(config)

    def clear_cache(self) -> None:
        """Clear cached configuration."""
        self._cache = None


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Later configurations override earlier ones.

    Args:
        *configs: Configuration dictionaries to merge

    Returns:
        Merged configuration

    Example:
        default_config = {"theme": "dark", "size": 12}
        user_config = {"theme": "light"}
        config = merge_configs(default_config, user_config)
        # Result: {"theme": "light", "size": 12}
    """
    result: dict[str, Any] = {}

    for config in configs:
        _deep_merge(result, config)

    return result


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Deep merge source into target dictionary.

    Args:
        target: Target dictionary (modified in place)
        source: Source dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def validate_config(
    config: dict[str, Any],
    required_keys: list[str],
    optional_keys: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate configuration has required keys.

    Args:
        config: Configuration to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (for documentation)

    Returns:
        Tuple of (is_valid, missing_keys)

    Example:
        valid, missing = validate_config(
            config,
            required_keys=["engine", "model_path"],
            optional_keys=["timestep", "gravity"]
        )
        if not valid:
            print(f"Missing keys: {missing}")
    """
    missing_keys = [key for key in required_keys if key not in config]
    is_valid = len(missing_keys) == 0

    if not is_valid:
        logger.warning(f"Configuration validation failed. Missing keys: {missing_keys}")

    return is_valid, missing_keys
