"""Drake GUI UI Setup Mixin.

``DrakeSimApp`` historically imported ``UISetupMixin`` from this module. The
implementation is unified with ``DrakeUIMixin`` to avoid parallel copies that
trip the DRY duplication ratchet.
"""

from __future__ import annotations

from .drake_ui_mixin import (
    HAS_MATPLOTLIB,
    HAS_QT,
)
from .drake_ui_mixin import (
    DrakeUIMixin as UISetupMixin,
)

__all__ = ["HAS_MATPLOTLIB", "HAS_QT", "UISetupMixin"]
