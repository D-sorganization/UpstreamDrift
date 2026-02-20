"""Engine interfaces, registry, manager, and loaders.

Source of Truth: UpstreamDrift (this repository)
Cross-repo install: pip install upstream-drift-shared

Sub-protocols (Interface Segregation):
    Loadable, Steppable, Queryable, DynamicsComputable,
    CounterfactualComputable, Recordable
"""

__all__: list[str] = [
    "CounterfactualComputable",
    "DynamicsComputable",
    "EngineLifecycle",
    "EnginePluginMetadata",
    "Loadable",
    "PluginRegistry",
    "Queryable",
    "Recordable",
    "Steppable",
]
