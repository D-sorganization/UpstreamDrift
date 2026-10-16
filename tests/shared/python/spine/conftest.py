"""Spine contract tests run against SciPy < 1.15 read-only semantics.

The spine dataclasses freeze their arrays; SciPy 1.13/1.14 ``Rotation``
constructors reject read-only buffers. Emulating that here keeps a current
local SciPy from hiding the defect the cross-engine lane (SciPy 1.14) sees.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.helpers.scipy_read_only import strict_rotation_constructors


@pytest.fixture(autouse=True, scope="module")
def _scipy_read_only_rotation() -> Iterator[None]:
    # Module scope so module-scoped fixtures run under the emulation too.
    with pytest.MonkeyPatch.context() as patch:
        strict_rotation_constructors(patch)
        yield
