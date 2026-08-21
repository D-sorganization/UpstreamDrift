"""Importable shim adapters for embeddable tools.

Some embeddable-tool adapters live at file paths that cannot be spelled
as dotted import paths (e.g. a path segment starting with a digit).
Modules in this package load those adapters from their file paths and
trigger their registration, so the launcher bootstrap can reference an
importable module for every tool (issue #8856).
"""
