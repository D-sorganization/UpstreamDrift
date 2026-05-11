"""MyoSuite motion-matching provider.

First-pass implementation satisfying the canonical discovery interface.
The actual optimizer over muscle activations is deferred to a Phase 2 
surrogate model. See AUDIT.md.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.shared.python.motion_matching.club_ball_target import ClubBallTarget
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    register_provider,
)

if TYPE_CHECKING:
    from src.shared.python.motion_matching.fit_result import CanonicalFitResult

logger = logging.getLogger(__name__)

__all__ = ["MyoSuiteFitSwingProvider"]


class MyoSuiteFitSwingProvider:
    engine_name: str = "myosuite"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget | ClubBallTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        """Not yet implemented for MyoSuite."""
        raise NotImplementedError(
            "MyoSuite motion-matching over muscle activations is deferred to Phase 2. "
            "See src/engines/physics_engines/myosuite/python/motion_matching/AUDIT.md."
        )

    def supports_body_target(self) -> bool:
        """MyoSuite models have full musculature and can target body markers."""
        return True

    def supports_ball_target(self) -> bool:
        """Ball impact constraints not yet implemented for MyoSuite."""
        return False

    def engine_version(self) -> str:
        try:
            import myosuite
            return str(getattr(myosuite, "__version__", "unknown"))
        except ImportError:
            return "unknown"

register_provider(MyoSuiteFitSwingProvider())
