"""Internal type aliases for the anthropometrics package.

These names exist purely so the public-facing modules
(:mod:`segment_properties`, :mod:`_subject_anthropometrics`,
:mod:`contracts`) read uniformly. They are not part of the
package's public API.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# A floating-point ndarray of arbitrary shape.
FloatArray: TypeAlias = NDArray[np.floating]

# A canonical body-segment identifier (e.g. "head", "torso",
# "upper_arm_left"). Kept as a plain ``str`` alias for now; a
# stricter ``Literal`` set may be introduced in a later issue.
BodyPartId: TypeAlias = str

# Tolerance for inertia-tensor symmetry / triangle-inequality checks.
INERTIA_NUMERIC_TOL: float = 1e-9


class Sex(str, Enum):
    """Subject sex.

    ``StrEnum`` is unavailable on Python 3.10, so we subclass
    ``(str, Enum)`` to retain string-equality semantics.
    """

    MALE = "M"
    FEMALE = "F"
    UNSPECIFIED = "unspecified"
