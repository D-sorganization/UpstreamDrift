#!/usr/bin/env python3
"""Test suite for Golf Modeling Suite UX improvements."""

import unittest
from unittest.mock import Mock, patch  # noqa: F401

from src.shared.python.engine_core.engine_availability import PYQT6_AVAILABLE

if PYQT6_AVAILABLE:
    from PyQt6.QtCore import Qt  # noqa: F401
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget  # noqa: F401


if __name__ == "__main__":
    unittest.main()
