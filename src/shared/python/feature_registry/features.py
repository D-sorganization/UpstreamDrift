"""Canonical feature definitions for UpstreamDrift.

This module is the **single source of truth** for the mapping from a
feature name (the user-facing identifier — ``drake``, ``mediapipe``,
``torch-cuda``, ...) to its install metadata: the pip-extra that
provides it, the Docker stage that bakes it in, the rough wheel size
that drives Docker profile budgets, and the engine tier the feature
belongs to.

Adding a new feature is one append here. The registry, the Docker
build script, the CLI table, and the install-prompt dialog all
consume this list — they do not maintain their own copies.

Design by Contract
------------------
* ``FEATURES`` is a tuple (immutable at import time).
* Every entry has a unique ``name``; :func:`get_feature` enforces this.
* Sizes are intentionally rough — they exist to drive *budgets*, not
  to be precise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EngineTier = Literal["core", "extended", "experimental", "tooling"]
"""Tier alignment with :mod:`src.engines.tiers`.

``tooling`` is added here for non-engine features (e.g. pose
backends, RL stacks) that the engine tier policy does not cover.
"""

InstallChannel = Literal["pip", "pip-extra", "conda", "external"]
"""How a feature is installed.

* ``pip`` — direct ``pip install <package>``.
* ``pip-extra`` — ``pip install upstream-drift[<extra>]`` is preferred.
* ``conda`` — wheel is unreliable; conda-forge is the supported path.
* ``external`` — requires a non-Python build (OpenPose, MATLAB).
"""


@dataclass(frozen=True)
class Feature:
    """Metadata describing a single optional feature.

    Attributes:
        name: Stable, lowercase, kebab-or-snake identifier used by the
            CLI, Docker build args, and the registry. Must be unique.
        display_name: Human-readable name for UI surfaces.
        description: One-sentence summary shown in the install dialog.
        probe_key: Key under which a probe is registered with
            :mod:`src.shared.python.feature_registry.registry`. ``None``
            means the feature is always considered available (e.g. the
            pure-Python ``pendulum`` model).
        install_channel: Where the install command comes from. See
            :data:`InstallChannel`.
        install_command: The verbatim shell command shown to the user
            and executed by the install-prompt UX.
        pip_extra: The optional-dependency extra in ``pyproject.toml``,
            if any. Used when ``install_channel == "pip-extra"``.
        docker_stage: Name of the Docker stage that bakes this feature
            in. Multiple features may share a stage. ``None`` means the
            feature is never installed inside Docker (e.g. MATLAB).
        approx_size_mb: Rough installed wheel + native lib footprint.
            Used for Docker-profile budgets and the install-prompt UI.
        tier: :data:`EngineTier` value.
        depends_on: Features that must already be installed for this
            one to function. The registry surfaces dependency gaps.
    """

    name: str
    display_name: str
    description: str
    probe_key: str | None
    install_channel: InstallChannel
    install_command: str
    pip_extra: str | None
    docker_stage: str | None
    approx_size_mb: int
    tier: EngineTier
    depends_on: tuple[str, ...] = ()


FEATURES: tuple[Feature, ...] = (
    # ---- Core engines ----------------------------------------------------
    Feature(
        name="api",
        display_name="API server",
        description="FastAPI/Uvicorn HTTP API and core routes.",
        probe_key=None,
        install_channel="pip-extra",
        install_command="pip install upstream-drift",
        pip_extra=None,
        docker_stage="api",
        approx_size_mb=120,
        tier="core",
    ),
    Feature(
        name="pendulum",
        display_name="Pendulum models",
        description="Pure-Python double pendulum + putting green simulators.",
        probe_key="pendulum",
        install_channel="pip-extra",
        install_command="pip install upstream-drift",
        pip_extra=None,
        docker_stage="api",
        approx_size_mb=0,
        tier="core",
    ),
    Feature(
        name="mujoco",
        display_name="MuJoCo",
        description="Default rigid-body physics engine (lightweight, cross-platform).",
        probe_key="mujoco",
        install_channel="pip",
        install_command="pip install 'mujoco>=3.2.3,<4.0.0'",
        pip_extra=None,
        docker_stage="mujoco",
        approx_size_mb=120,
        tier="core",
    ),
    # ---- Extended engines ------------------------------------------------
    Feature(
        name="drake",
        display_name="Drake",
        description="MIT Drake multibody dynamics; large but feature-rich.",
        probe_key="drake",
        install_channel="pip-extra",
        install_command="pip install 'upstream-drift[drake]'",
        pip_extra="drake",
        docker_stage="drake",
        approx_size_mb=700,
        tier="extended",
    ),
    Feature(
        name="pinocchio",
        display_name="Pinocchio",
        description="Fast rigid-body algorithms (Pin + Pink + qpsolvers).",
        probe_key="pinocchio",
        install_channel="pip-extra",
        install_command="pip install 'upstream-drift[pinocchio]'",
        pip_extra="pinocchio",
        docker_stage="pinocchio",
        approx_size_mb=210,
        tier="extended",
    ),
    # ---- Experimental engines -------------------------------------------
    Feature(
        name="opensim",
        display_name="OpenSim",
        description="Musculoskeletal modeling; conda-forge build recommended.",
        probe_key="opensim",
        install_channel="conda",
        install_command="conda install -c opensim-org opensim",
        pip_extra="biomechanics",
        docker_stage="biomech",
        approx_size_mb=400,
        tier="experimental",
    ),
    Feature(
        name="myosuite",
        display_name="MyoSuite",
        description="Muscle-driven motor control RL environments.",
        probe_key="myosuite",
        install_channel="pip-extra",
        install_command="pip install 'upstream-drift[biomechanics]'",
        pip_extra="biomechanics",
        docker_stage="biomech",
        approx_size_mb=1400,
        tier="experimental",
        depends_on=("mujoco",),
    ),
    Feature(
        name="chrono",
        display_name="PyChrono",
        description="Multibody / DEM / FEM (granular bunker shots, fabrics).",
        probe_key="chrono",
        install_channel="conda",
        install_command="conda install -c projectchrono pychrono",
        pip_extra="chrono",
        docker_stage="chrono",
        approx_size_mb=600,
        tier="experimental",
    ),
    # ---- Pose estimation backends ---------------------------------------
    Feature(
        name="pose-mediapipe",
        display_name="MediaPipe pose",
        description="Google MediaPipe 2-D / 3-D pose estimation.",
        probe_key="mediapipe",
        install_channel="pip-extra",
        install_command="pip install 'upstream-drift[pose]'",
        pip_extra="pose",
        docker_stage="pose",
        approx_size_mb=300,
        tier="tooling",
    ),
    Feature(
        name="pose-openpose",
        display_name="OpenPose",
        description="CMU OpenPose; host-built, requires CUDA & cmake.",
        probe_key="openpose",
        install_channel="external",
        install_command=(
            "# See docs/installation/openpose.md — OpenPose is host-built and "
            "cannot be installed via pip."
        ),
        pip_extra=None,
        docker_stage=None,
        approx_size_mb=0,
        tier="tooling",
    ),
    # ---- ML / training stack --------------------------------------------
    Feature(
        name="torch-cuda",
        display_name="PyTorch (CUDA 12.4)",
        description="GPU PyTorch wheels for RL training and ML pipelines.",
        probe_key="torch",
        install_channel="pip",
        install_command=(
            "pip install 'torch==2.8.0' "
            "--index-url https://download.pytorch.org/whl/cu124"
        ),
        pip_extra=None,
        docker_stage="training",
        approx_size_mb=2800,
        tier="tooling",
    ),
    Feature(
        name="rl-stack",
        display_name="Reinforcement learning stack",
        description="Gymnasium + Stable-Baselines3 + Ray RLlib + TensorBoard.",
        probe_key="rl",
        install_channel="pip-extra",
        install_command="pip install 'upstream-drift[rl]'",
        pip_extra="rl",
        docker_stage="training",
        approx_size_mb=250,
        tier="tooling",
        depends_on=("torch-cuda",),
    ),
)


_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}


def get_feature(name: str) -> Feature:
    """Return the feature with this name.

    Preconditions:
        * ``name`` is a non-empty string.

    Postconditions:
        * Returned feature's ``name`` equals the input name (lowercased).

    Raises:
        KeyError: if ``name`` is not registered.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("name must not be empty")
    try:
        return _BY_NAME[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(_BY_NAME))
        raise KeyError(f"Unknown feature {name!r}. Known features: {known}") from exc


def all_features() -> tuple[Feature, ...]:
    """Return every registered feature in definition order."""
    return FEATURES


def features_for_stage(stage: str) -> tuple[Feature, ...]:
    """Return every feature whose ``docker_stage`` equals ``stage``.

    Used by ``scripts/docker/install_features.py`` to compute the
    install command set for a given Dockerfile stage.
    """
    if not isinstance(stage, str):
        raise TypeError("stage must be a string")
    return tuple(f for f in FEATURES if f.docker_stage == stage)
