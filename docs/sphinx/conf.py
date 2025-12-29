# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
project = "Golf Modeling Suite"
copyright = "2025, Golf Modeling Suite Team"
author = "Golf Modeling Suite Team"
release = "1.0.0"

# -- General configuration ---------------------------------------------------
sys.path.insert(0, os.path.abspath("../../"))  # Point to root

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "myst_parser",  # For Markdown support
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Autodoc configuration ---------------------------------------------------
autodoc_mock_imports = [
    "mujoco",
    "pydrake",
    "pinocchio",
    "opensim",
    "PyQt6",
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
]
