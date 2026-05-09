"""Concrete :class:`~anthropometrics.contracts.EngineAdapter` implementations.

Five adapters ship out of the box, one per supported physics
engine. Each implements :class:`anthropometrics.EngineAdapter`
and round-trips a :class:`SubjectAnthropometrics` losslessly
(``rtol=1e-9, atol=1e-12``) through the engine's native on-disk
format:

* :class:`DrakeAdapter`     — URDF
* :class:`PinocchioAdapter` — URDF
* :class:`MyoSuiteAdapter`  — URDF + MJCF
* :class:`OpenSimAdapter`   — ``.osim`` XML
* :class:`SimscapeAdapter`  — ``.mat`` (scipy.io.savemat)

Adapters are registered in :data:`ADAPTER_REGISTRY` keyed by
``engine_name`` so callers can resolve one dynamically::

    from anthropometrics.engine_adapters import ADAPTER_REGISTRY
    adapter = ADAPTER_REGISTRY["drake"]
    adapter.export(subject, Path("subject.urdf"))
"""

from __future__ import annotations

from ..contracts import EngineAdapter
from .drake import DrakeAdapter
from .myosuite import MyoSuiteAdapter
from .opensim import OpenSimAdapter
from .pinocchio import PinocchioAdapter
from .simscape import SimscapeAdapter

ADAPTER_REGISTRY: dict[str, EngineAdapter] = {
    DrakeAdapter.engine_name: DrakeAdapter(),
    PinocchioAdapter.engine_name: PinocchioAdapter(),
    MyoSuiteAdapter.engine_name: MyoSuiteAdapter(),
    OpenSimAdapter.engine_name: OpenSimAdapter(),
    SimscapeAdapter.engine_name: SimscapeAdapter(),
}

__all__ = [
    "ADAPTER_REGISTRY",
    "DrakeAdapter",
    "MyoSuiteAdapter",
    "OpenSimAdapter",
    "PinocchioAdapter",
    "SimscapeAdapter",
]
