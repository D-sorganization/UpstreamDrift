"""Resolve Legacy Data Processing Modules to the Pinned Tools Provider."""

from importlib import import_module
from types import ModuleType

from src.shared.python._seam_redirect import load_pinned_tools_package

__all__ = ["load_provider_module"]


def load_provider_module(name: str) -> ModuleType:
    """Return a canonical module without grafting Tools onto a local package.

    Only the retired core and exception modules are exposed. Missing pinned
    dependencies fail through the existing seam loader with its setup hint.
    """
    if name not in {"core", "exceptions"}:
        raise ValueError(f"unsupported data processing provider module: {name}")
    provider = load_pinned_tools_package("sidekick")
    if provider.__name__ != "_pinned_tools__sidekick":
        raise ValueError("unexpected pinned provider identity")
    if name == "core":
        return import_module("_pinned_tools__sidekick.data_processing.core")
    return import_module("_pinned_tools__sidekick.data_processing.exceptions")
