"""Version-tolerant MyoSuite runtime loader (#12635).

MyoSuite 2.x exposes ``myosuite.envs.env_base.MujocoEnv``; 3.x removed it and
builds models through ``myosuite.core.model_builder.ModelBuilder``.  This is
the single place that knows both layouts, so callers hold a runtime object
with ``mj_spec`` / ``mj_model`` / ``mj_data`` whichever line is installed and
the dependency pin can stay unbounded.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from importlib import import_module
from typing import Any

__all__ = [
    "SUPPORTED_LAYOUTS",
    "MyoSuiteLayoutError",
    "MyoSuiteRuntime",
    "detect_runtime_layout",
    "load_runtime_from_xml",
]

# Newest layout first: (layout name, module, attribute that must exist).
_LAYOUTS: tuple[tuple[str, str, str], ...] = (
    ("model_builder", "myosuite.core.model_builder", "ModelBuilder"),
    ("env_base", "myosuite.envs.env_base", "MujocoEnv"),
)
SUPPORTED_LAYOUTS: tuple[str, ...] = tuple(name for name, _, _ in _LAYOUTS)


class MyoSuiteLayoutError(ImportError):
    """MyoSuite is installed but exposes no runtime layout we support."""


@dataclass(frozen=True)
class MyoSuiteRuntime:
    """MyoSuite-held compiled model, data and spec."""

    mj_spec: Any
    mj_model: Any
    mj_data: Any


def detect_runtime_layout() -> str:
    """Name of the installed MyoSuite runtime layout.

    Postcondition: the result is a member of :data:`SUPPORTED_LAYOUTS`.

    Raises:
        MyoSuiteLayoutError: no supported layout imports; lists each failure.
    """
    failures: list[str] = []
    for name, module, attr in _LAYOUTS:
        try:
            getattr(import_module(module), attr)
        except (ImportError, AttributeError) as exc:
            failures.append(f"{module}.{attr}: {exc}")
        else:
            return name
    raise MyoSuiteLayoutError(
        "installed myosuite exposes no supported runtime layout (#12635); tried "
        + "; ".join(failures)
    )


def load_runtime_from_xml(xml: str) -> MyoSuiteRuntime:
    """Compile MJCF text through the installed MyoSuite runtime.

    Precondition: ``xml`` is non-empty MJCF text.
    Postcondition: ``mj_spec``, ``mj_model`` and ``mj_data`` are compiled from
    ``xml``.
    """
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("xml must be non-empty MJCF text")
    layout = detect_runtime_layout()
    handle, path = tempfile.mkstemp(suffix=".xml", prefix="myosuite_spec_")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(xml)
        if layout == "model_builder":
            import mujoco

            builder = import_module("myosuite.core.model_builder").ModelBuilder
            model, spec = builder.from_xml_file(path).build()
            return MyoSuiteRuntime(spec, model, mujoco.MjData(model))
        env = import_module("myosuite.envs.env_base").MujocoEnv(path)
        return MyoSuiteRuntime(env.mj_spec, env.mj_model, env.mj_data)
    finally:
        os.unlink(path)
