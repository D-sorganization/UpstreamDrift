"""Sphinx configuration for DTACK API documentation."""


# Add python directory to path

project = "DTACK Golf Biomechanics Platform"
copyright = "2024, Dieter Olson"
author = "Dieter Olson"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pinocchio": ("https://stack-of-tasks.github.io/pinocchio/", None),
}
