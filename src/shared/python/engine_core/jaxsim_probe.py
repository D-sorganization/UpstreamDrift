"""JaxSim engine readiness probe."""

from __future__ import annotations

import importlib
from pathlib import Path

from .engine_probes import EngineProbe, EngineProbeResult, ProbeStatus, _resolve_engines_root


class JaxSimProbe(EngineProbe):
    """Probe for JaxSim physics engine readiness."""

    def __init__(self, suite_root: Path) -> None:
        """Initialize JaxSim probe."""
        if suite_root is None:
            raise ValueError("suite_root must be provided")
        super().__init__("JaxSim", suite_root)

    def probe(self) -> EngineProbeResult:
        """Check JaxSim package/API availability and local adapter assets."""
        try:
            jaxsim = importlib.import_module("jaxsim")
            importlib.import_module("jaxsim.api")
        except ImportError:
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.NOT_INSTALLED,
                version=None,
                missing_dependencies=["jaxsim"],
                diagnostic_message=(
                    "JaxSim Python package not installed. Install the runtime "
                    "before selecting the JaxSim engine."
                ),
            )
        except Exception as exc:  # noqa: BLE001 - import-time runtime failures
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_BINARY,
                version=None,
                missing_dependencies=["jaxsim.api"],
                diagnostic_message=f"JaxSim runtime import failed: {exc}",
            )

        version = getattr(jaxsim, "__version__", "unknown")
        engine_dir = _resolve_engines_root(self.suite_root) / "physics_engines" / "jaxsim"
        if not engine_dir.exists():
            return EngineProbeResult(
                engine_name=self.engine_name,
                status=ProbeStatus.MISSING_ASSETS,
                version=version,
                missing_dependencies=["engine directory"],
                diagnostic_message=f"JaxSim {version} installed but adapter is missing.",
            )

        return EngineProbeResult(
            engine_name=self.engine_name,
            status=ProbeStatus.AVAILABLE,
            version=version,
            missing_dependencies=[],
            diagnostic_message=f"JaxSim {version} ready",
            details={"engine_dir": str(engine_dir)},
        )
