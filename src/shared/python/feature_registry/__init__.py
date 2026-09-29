"""Runtime feature availability registry.

Public API:

* :class:`Feature` and :data:`FEATURES` — canonical feature definitions.
* :class:`FeatureReport` — per-feature status output.
* :class:`CapabilityRegistry` and :func:`get_registry` — the registry
  itself.
* :func:`refresh` — re-run probes after an install.

Typical use::

    from src.shared.python.feature_registry import get_registry

    reg = get_registry()
    if not reg.is_available("drake"):
        report = reg.check("drake")
        print("Drake is missing; install with:", report.install_command)
"""

from __future__ import annotations

from src.shared.python.feature_registry.features import (
    FEATURES,
    Feature,
    all_features,
    features_for_stage,
    get_feature,
)
from src.shared.python.feature_registry.installer import (
    InstallResult,
    install_feature,
)
from src.shared.python.feature_registry.probes import PROBES, ProbeOutcome
from src.shared.python.feature_registry.registry import (
    CapabilityRegistry,
    FeatureReport,
    get_registry,
    refresh,
)

__all__ = [
    "FEATURES",
    "Feature",
    "FeatureReport",
    "CapabilityRegistry",
    "PROBES",
    "ProbeOutcome",
    "InstallResult",
    "all_features",
    "features_for_stage",
    "get_feature",
    "get_registry",
    "install_feature",
    "refresh",
]
