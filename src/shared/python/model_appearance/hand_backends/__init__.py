"""Swappable hand visual backends for grip comparison (#15122)."""

__all__ = ["build_hand_visual", "registered_backend_ids"]


def __getattr__(name: str):
    if name in __all__:
        from src.shared.python.model_appearance.hand_backends import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
