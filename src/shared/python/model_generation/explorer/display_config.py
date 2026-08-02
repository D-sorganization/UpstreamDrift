"""
Display configuration constants for the Model Explorer.

Separated from the GUI module so they can be tested without PyQt6.
"""

# Display option definitions: (key, label, default_checked)
# Order matters - this is the order checkboxes appear in the UI.
DISPLAY_OPTIONS: list[tuple[str, str, bool]] = [
    ("segments", "Segments", True),
    ("joints", "Joints", True),
    ("collisions", "Collisions", True),
    ("inertias", "Inertias", True),
    ("frames", "Frames", False),
]
