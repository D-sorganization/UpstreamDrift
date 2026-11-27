"""Swappable grip force and contact parameter providers (HRG-GB1, #15123)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from src.shared.python.contracts import require
from src.shared.python.grip_contact.grip_force_limit import UNLIMITED, GripForceLimit
from src.shared.python.grip_spec.defaults import (
    DEFAULT_CONTACT_PROVIDER_NAME,
    DEFAULT_FORCE_PROVIDER_NAME,
)

if TYPE_CHECKING:
    from src.shared.python.grip_spec.types import GripSpec


@runtime_checkable
class GripForceProvider(Protocol):
    """Resolves per-hand normal-force limits (never hard-coded on the spec)."""

    def lead_force_limit(self) -> GripForceLimit: ...

    def trail_force_limit(self) -> GripForceLimit: ...


@runtime_checkable
class ContactParameterProvider(Protocol):
    """Resolves contact law inputs for a grip build or physics attach."""

    provider_name: str

    def preload_penetration_m(self) -> float: ...

    def document(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class _UnspecifiedForceProvider:
    def lead_force_limit(self) -> GripForceLimit:
        return UNLIMITED

    def trail_force_limit(self) -> GripForceLimit:
        return UNLIMITED


@dataclass(frozen=True)
class _UnspecifiedContactProvider:
    provider_name: str = DEFAULT_CONTACT_PROVIDER_NAME

    def preload_penetration_m(self) -> float:
        from src.shared.python.grip_contact.pad_layout import PadLayout

        return PadLayout().preload_penetration_m

    def document(self) -> dict[str, object]:
        return {"provider_name": self.provider_name, "source": "pad_layout"}


_FORCE_REGISTRY: dict[str, GripForceProvider] = {
    DEFAULT_FORCE_PROVIDER_NAME: _UnspecifiedForceProvider(),
}
_CONTACT_REGISTRY: dict[str, ContactParameterProvider] = {
    DEFAULT_CONTACT_PROVIDER_NAME: cast(
        ContactParameterProvider, _UnspecifiedContactProvider()
    ),
}


def register_grip_force_provider(name: str, provider: GripForceProvider) -> None:
    require(bool(isinstance(name, str) and name.strip()), "provider name required")
    if not isinstance(provider, GripForceProvider):
        raise TypeError("provider must implement GripForceProvider")
    _FORCE_REGISTRY[name] = provider


def register_contact_parameter_provider(
    name: str, provider: ContactParameterProvider
) -> None:
    require(bool(isinstance(name, str) and name.strip()), "provider name required")
    if not isinstance(provider, ContactParameterProvider):
        raise TypeError("provider must implement ContactParameterProvider")
    _CONTACT_REGISTRY[name] = provider


def resolve_grip_force_limits(spec: GripSpec) -> dict[str, GripForceLimit]:
    name = spec.force_provider_name
    if name not in _FORCE_REGISTRY:
        require(False, f"unknown force provider {name!r}")
    provider = _FORCE_REGISTRY[name]
    return {
        "lead": provider.lead_force_limit(),
        "trail": provider.trail_force_limit(),
    }


def resolve_contact_parameters(spec: GripSpec) -> ContactParameterProvider:
    name = spec.contact_provider_name
    if name not in _CONTACT_REGISTRY:
        require(False, f"unknown contact provider {name!r}")
    return _CONTACT_REGISTRY[name]


__all__ = [
    "ContactParameterProvider",
    "GripForceProvider",
    "register_contact_parameter_provider",
    "register_grip_force_provider",
    "resolve_contact_parameters",
    "resolve_grip_force_limits",
]
