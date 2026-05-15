"""AIP JSON-RPC 2.0 server package.

Provides a JSON-RPC 2.0 interface for external tool integration
with UpstreamDrift simulation and analysis capabilities.
"""

from .dispatcher import dispatch
from .methods import MethodRegistry, create_registry

__all__: list[str] = [
    "MethodRegistry",
    "create_registry",
    "dispatch",
]
