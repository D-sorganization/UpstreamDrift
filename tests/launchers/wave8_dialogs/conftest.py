"""Wave 8 dialog/layout test fixtures.

Ensures Qt runs in offscreen mode for headless safety.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
