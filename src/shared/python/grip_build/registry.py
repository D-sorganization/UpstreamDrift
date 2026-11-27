"""Builder and contact-model registries (HRG-GB2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from src.shared.python.grip_build.contact_protocol import (
    ContactDistanceModel,
    DefaultCapsuleContactModel,
)
from src.shared.python.grip_build.receipt import FullGripBuildResult

DEFAULT_BUILDER_NAME = "flat_hand_curl_v1"

_builders: dict[str, GripBuilder] = {}
_contact_models: dict[str, ContactDistanceModel] = {
    DefaultCapsuleContactModel.name: DefaultCapsuleContactModel()
}


@runtime_checkable
class GripBuilder(Protocol):
    """Procedural grip builder contract."""

    name: str

    def build_lead_fingers(self, *args, **kwargs):
        """Build lead-hand finger curl stages."""

    def build_grip(self, *args, **kwargs) -> FullGripBuildResult:
        """Build the full GB-3 grip (optional on older builders)."""


def register_grip_builder(name: str, builder: GripBuilder) -> None:
    if not name:
        raise ValueError("builder name must be non-empty")
    if name in _builders:
        raise ValueError(f"grip builder already registered: {name!r}")
    if not isinstance(builder, GripBuilder):
        raise TypeError("builder must implement GripBuilder")
    _builders[name] = builder


def get_grip_builder(name: str = DEFAULT_BUILDER_NAME) -> GripBuilder:
    if name not in _builders:
        raise KeyError(f"unknown grip builder {name!r}")
    return _builders[name]


def register_contact_distance_model(name: str, model: ContactDistanceModel) -> None:
    if not name:
        raise ValueError("model name must be non-empty")
    if name in _contact_models:
        raise ValueError(f"contact model already registered: {name!r}")
    if not isinstance(model, ContactDistanceModel):
        raise TypeError("model must implement ContactDistanceModel")
    _contact_models[name] = model


def get_contact_distance_model(
    name: str = DefaultCapsuleContactModel.name,
) -> ContactDistanceModel:
    if name not in _contact_models:
        raise KeyError(f"unknown contact distance model {name!r}")
    return _contact_models[name]


def builder_registry() -> Mapping[str, GripBuilder]:
    return dict(_builders)


def contact_model_registry() -> Mapping[str, ContactDistanceModel]:
    return dict(_contact_models)
