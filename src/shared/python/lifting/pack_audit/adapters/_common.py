"""Helpers shared by the engine adapters."""

from __future__ import annotations

import importlib
from collections.abc import Mapping

from ..model import Anthropometry
from ..names import CANONICAL_COORDINATES, engine_coordinate_name
from ..packs import PackLocation


def build_model_text(pack: PackLocation, lift: str, anthro: Anthropometry) -> str:
    """Call the pack's own ``build_<lift>_model`` generator."""
    pack.activate()
    module = importlib.import_module(f"{pack.package}.exercises.{lift}.{lift}_model")
    builder = getattr(module, f"build_{lift}_model")
    return str(
        builder(
            body_mass=anthro.body_mass_kg,
            height=anthro.height_m,
            plate_mass_per_side=anthro.plate_mass_per_side_kg,
        )
    )


def native_angles(engine: str, q: Mapping[str, float]) -> dict[str, float]:
    """Translate canonical angles to the engine's coordinate names.

    Raises:
        ValueError: If *q* names a coordinate outside the canonical set.
    """
    unknown = sorted(set(q) - set(CANONICAL_COORDINATES))
    if unknown:
        raise ValueError(f"unknown canonical coordinate(s): {unknown}")
    return {engine_coordinate_name(engine, name): float(v) for name, v in q.items()}


PELVIS_ORIGIN = (0.0, 0.0, 1.0)
