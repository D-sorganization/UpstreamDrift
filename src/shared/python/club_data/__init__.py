"""Club-data authority with lazy GUI/Excel loading for headless consumers."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PlayerClub": "player_clubs",
    "PlayerBag": "player_clubs",
    "CaptureClubSnapshot": "player_clubs",
    "ClubIdentity": "catalog",
    "ClubRecord": "catalog",
    "PropertyClaim": "catalog",
    "SpecificationSource": "catalog",
    "ClubDataLoader": "loader",
    "ClubSpecification": "loader",
    "ProPlayerData": "loader",
    "SwingMetrics": "loader",
    "load_club_data": "loader",
    "load_pro_player_data": "loader",
    "ClubDataDisplayWidget": "display",
    "ClubTargetOverlay": "display",
    "ClubTargetManager": "targets",
    "TargetTrajectory": "targets",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Retain the public facade without importing optional UI dependencies eagerly."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
