"""Registry API for grip hand visual backends (ADR-0057 A3)."""

from __future__ import annotations

import pytest

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.hand_backends import (
    HandVisualBackend,
    HandVisualResult,
    build_hand_visual,
    get_hand_visual_backend,
    list_hand_visual_backends,
    register_hand_visual_backend,
)
from src.shared.python.model_appearance.hand_backends import registry

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _empty_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_BUILTIN_BACKEND_MODULES", ())


def _record(name: str, *, default_enabled: bool = True) -> HandVisualBackend:
    def builder(grip_spec, side, joint_angles=None, *, grip_radius_m, skin_rgb=None):
        return (name, side, grip_radius_m, skin_rgb)

    return HandVisualBackend(
        name=name, builder=builder, description=name, default_enabled=default_enabled
    )


def test_register_get_and_list() -> None:
    register_hand_visual_backend(_record("beta"))
    register_hand_visual_backend(_record("alpha"))
    assert list_hand_visual_backends() == ("alpha", "beta")
    assert get_hand_visual_backend("alpha").name == "alpha"


def test_duplicate_name_is_rejected() -> None:
    register_hand_visual_backend(_record("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        register_hand_visual_backend(_record("alpha"))


@pytest.mark.parametrize(
    "name", ["", "Alpha", "alpha-hand", "1hand", "alpha hand", "_x"]
)
def test_bad_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        register_hand_visual_backend(_record(name))


def test_unknown_name_raises_key_error_listing_known_names() -> None:
    register_hand_visual_backend(_record("alpha"))
    with pytest.raises(KeyError, match="alpha"):
        get_hand_visual_backend("nope")


def test_disabled_backend_is_hidden_by_default_but_builds_by_name() -> None:
    register_hand_visual_backend(_record("alpha"))
    register_hand_visual_backend(_record("lab", default_enabled=False))
    assert list_hand_visual_backends() == ("alpha",)
    assert list_hand_visual_backends(include_disabled=True) == ("alpha", "lab")
    out = build_hand_visual(
        GripSpec.legacy(),
        "lead",
        grip_radius_m=0.012,
        backend="lab",
        skin_rgb=(1.0, 0.0, 0.0),
    )
    assert out == ("lab", "lead", 0.012, (1.0, 0.0, 0.0))


def test_myohand_must_be_disabled_by_default() -> None:
    with pytest.raises(ValueError, match="myohand"):
        register_hand_visual_backend(_record("myohand"))
    register_hand_visual_backend(_record("myohand", default_enabled=False))
    assert list_hand_visual_backends() == ()
    assert list_hand_visual_backends(include_disabled=True) == ("myohand",)


def test_builtin_modules_register_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    module = types.ModuleType("_fake_hand_backend_module")
    calls: list[str] = []

    def register() -> None:
        calls.append("register")
        register_hand_visual_backend(_record("fake"))

    module.register = register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(registry, "_BUILTIN_BACKEND_MODULES", (module.__name__,))
    monkeypatch.setattr(registry, "_builtins_loaded", False)
    assert calls == []
    assert list_hand_visual_backends() == ("fake",)
    assert list_hand_visual_backends() == ("fake",)
    assert calls == ["register"]


def test_backend_record_validates_its_fields() -> None:
    with pytest.raises(TypeError):
        HandVisualBackend(name="alpha", builder="not callable", description="")  # type: ignore[arg-type]
    record = _record("alpha")
    assert record.capabilities == frozenset()
    assert record.default_enabled is True


def test_result_rejects_out_of_range_skin_colour() -> None:
    import numpy as np

    from src.shared.python.model_appearance.geometry import Mesh

    mesh = Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]], dtype=np.int64))
    with pytest.raises(ValueError, match="skin_rgb"):
        HandVisualResult(
            backend="alpha",
            side="lead",
            mesh=mesh,
            skin_rgb=(1.2, 0.0, 0.0),
            grip_radius_m=0.012,
        )
