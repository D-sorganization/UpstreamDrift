"""Putting Green GUI package.

Calculation reference: the putting-stroke kinematics and kinetics this tool
presents are derived in ``docs/physics/PUTTING_KINEMATICS_KINETICS_REVIEW.md``,
which is also catalogued under "Calculation and Derivation References" in
``docs/index.md``. Change the model here only alongside that sheet (#8850).
"""

from src.shared.python.launcher_embed import register_embeddable_tool
from ._embed_adapter import PuttingGreenGuiAdapter

# Register immediately when the package is imported
register_embeddable_tool(PuttingGreenGuiAdapter())

__all__ = ["PuttingGreenGuiAdapter"]
