"""The Rust facades must be reachable from the converters that own them.

Both `_urdf_rust_facade` and `_mjcf_rust_facade` lost their only production
call site in the squash `b8d95ad25`, leaving the opt-in `UPSTREAM_URDF_USE_RUST`
path unreachable. Nothing went red: each facade kept its own parity tests, so
the modules were still imported and still passed -- they were simply never
invoked by the code they exist to accelerate.

`tests/unit/urdf/test_rust_facade_parity.py` cannot catch this. It is guarded
by `importorskip("upstream_urdf")` and so skips wherever the Rust wheel is
absent, which is most environments; and even where it runs it exercises the
facade directly rather than through the converter.

These tests therefore assert *wiring*, not parity, and deliberately avoid the
wheel: the facade is stubbed, and the question is only whether the converter
reaches it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

_SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="wiring_probe"><link name="base_link"/></robot>
"""

_SIMPLE_MJCF = """<?xml version="1.0"?>
<mujoco model="wiring_probe"><worldbody><body name="base_link"/></worldbody></mujoco>
"""


def _stub_facade(**overrides: object) -> MagicMock:
    facade = MagicMock()
    facade.should_use_rust.return_value = True
    for key, value in overrides.items():
        setattr(facade, key, value)
    return facade


class TestUrdfRustFastPathWiring:
    def test_parser_consults_the_facade_when_opted_in(self) -> None:
        """`URDFParser.parse` must ask the facade whether to use Rust."""
        from model_generation.converters import _urdf_rust_facade
        from model_generation.converters.urdf_parser import URDFParser

        with (
            patch.object(
                _urdf_rust_facade, "should_use_rust", return_value=False
            ) as gate,
        ):
            URDFParser().parse(_SIMPLE_URDF)

        assert gate.called, (
            "URDFParser.parse never consulted _urdf_rust_facade; the opt-in "
            "UPSTREAM_URDF_USE_RUST path is unreachable"
        )

    def test_parser_falls_back_to_python_when_the_facade_raises(self) -> None:
        """A Rust failure must not fail the parse."""
        from model_generation.converters import _urdf_rust_facade
        from model_generation.converters.urdf_parser import URDFParser

        with (
            patch.object(_urdf_rust_facade, "should_use_rust", return_value=True),
            patch.object(
                _urdf_rust_facade,
                "parse_urdf_to_dict",
                side_effect=RuntimeError("rust exploded"),
            ),
        ):
            model = URDFParser().parse(_SIMPLE_URDF)

        assert model.name == "wiring_probe"
        assert [link.name for link in model.links] == ["base_link"]


class TestMjcfRustFastPathWiring:
    def test_converter_consults_the_facade_when_opted_in(self) -> None:
        """`MJCFConverter.mjcf_to_urdf` must reach `_mjcf_rust_facade`."""
        from model_generation.converters import _mjcf_rust_facade
        from model_generation.converters.mjcf_converter import MJCFConverter

        with patch.object(
            _mjcf_rust_facade, "should_use_rust", return_value=False
        ) as gate:
            MJCFConverter().mjcf_to_urdf(_SIMPLE_MJCF)

        assert gate.called, (
            "mjcf_to_urdf never consulted _mjcf_rust_facade; the opt-in "
            "UPSTREAM_URDF_USE_RUST path is unreachable"
        )

    def test_converter_falls_back_to_python_when_the_facade_raises(self) -> None:
        """The Rust pass is a pre-check; the Python path stays authoritative."""
        from model_generation.converters import _mjcf_rust_facade
        from model_generation.converters.mjcf_converter import MJCFConverter

        with (
            patch.object(_mjcf_rust_facade, "should_use_rust", return_value=True),
            patch.object(
                _mjcf_rust_facade,
                "parse_mjcf_to_dict",
                side_effect=RuntimeError("rust exploded"),
            ),
        ):
            urdf = MJCFConverter().mjcf_to_urdf(_SIMPLE_MJCF)

        assert "wiring_probe" in urdf
