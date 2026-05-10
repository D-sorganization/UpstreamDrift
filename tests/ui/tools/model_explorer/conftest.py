"""Local fixtures for the Model Explorer embed-adapter tests.

The registry is process-wide; we only clear it for the cases that
explicitly assert on its state, since the import-side-effect test must
observe a *fresh* registration.
"""

from __future__ import annotations

import os

# Force the offscreen Qt platform before any PyQt6 import so headless
# CI runners do not fail to instantiate widgets.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
