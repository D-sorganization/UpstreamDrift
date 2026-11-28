"""Visual hair silhouettes built on the existing closed scalp shell.

All measurements use head-length units. Styles describe approximations of
period grooming, not a measurement of any individual person's hair.
"""

import math
from dataclasses import replace
from typing import Any

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.head_hair import (
    build_short_crop,
    load_hair_crop,
)


def build_hair_style(skin: Mesh, part: str, style: str) -> Mesh:
    """Build one of five bounded named hairstyles on an outward-wound skin."""
    base = load_hair_crop()
    # The lower rear hairline adds visible coverage to the nape; a separate
    # crown and forelock field changes the profile and combed part.
    presets: dict[str, dict[str, Any]] = {
        "side_swept": {
            "hairline": (
                (0.0, 0.77),
                (0.5, 0.77),
                (0.85, 0.69),
                (1.1, 0.55),
                (1.6, 0.58),
                (1.85, 0.48),
                (2.05, 0.37),
                (2.45, 0.30),
                (2.85, 0.27),
                (math.pi, 0.27),
            ),
            "hairline_asymmetry": 0.10,
            "side_thickness": 0.025,
            "top_thickness": 0.055,
            "fringe_thickness": 0.035,
            "heavy_side_gain": 0.26,
            "comb_sweep": 0.65,
        },
        "swept_back": {
            "hairline": (
                (0.0, 0.79),
                (0.5, 0.79),
                (0.85, 0.72),
                (1.1, 0.57),
                (1.6, 0.60),
                (1.85, 0.48),
                (2.05, 0.35),
                (2.45, 0.27),
                (2.85, 0.24),
                (math.pi, 0.24),
            ),
            "hairline_asymmetry": 0.10,
            "side_thickness": 0.042,
            "top_thickness": 0.065,
            "fringe_thickness": 0.025,
            "heavy_side_gain": 0.25,
            "comb_sweep": 0.45,
        },
        "slick_part": {
            "hairline": (
                (0.0, 0.84),
                (0.5, 0.83),
                (0.85, 0.76),
                (1.1, 0.61),
                (1.6, 0.63),
                (1.85, 0.52),
                (2.05, 0.42),
                (2.45, 0.36),
                (2.85, 0.33),
                (math.pi, 0.32),
            ),
            "hairline_asymmetry": 0.08,
            "side_thickness": 0.009,
            "top_thickness": 0.035,
            "fringe_thickness": 0.012,
            "part_depth": 0.55,
            "heavy_side_gain": 0.25,
            "comb_amplitude": 0.004,
        },
        "receding_crop": {
            "hairline": (
                (0.0, 0.88),
                (0.5, 0.89),
                (0.85, 0.82),
                (1.1, 0.65),
                (1.6, 0.64),
                (1.85, 0.53),
                (2.05, 0.44),
                (2.45, 0.38),
                (2.85, 0.35),
                (math.pi, 0.34),
            ),
            "side_thickness": 0.008,
            "top_thickness": 0.026,
            "fringe_thickness": 0.006,
            "part_depth": 0.15,
            "heavy_side_gain": 0.08,
            "comb_amplitude": 0.003,
        },
        "tousled_wave": {
            "hairline": (
                (0.0, 0.76),
                (0.5, 0.75),
                (0.85, 0.69),
                (1.1, 0.56),
                (1.6, 0.61),
                (1.85, 0.50),
                (2.05, 0.39),
                (2.45, 0.32),
                (2.85, 0.29),
                (math.pi, 0.29),
            ),
            "hairline_asymmetry": 0.0,
            "hairline_waviness": 0.006,
            "side_thickness": 0.018,
            "top_thickness": 0.065,
            "fringe_thickness": 0.046,
            "fringe_band": 0.035,
            "comb_amplitude": 0.015,
            "comb_wavelength": 0.10,
            "comb_sweep": 0.12,
            "part_depth": 0.0,
            "heavy_side_gain": 0.0,
            "wave_bend": 0.42,
            "wave_wavelength": 0.26,
        },
    }
    if style not in presets:
        raise ValueError(f"Unknown named hairstyle: {style!r}")
    cfg = replace(base, **presets[style])
    if part not in ("left", "right", "none"):
        raise ValueError(f"Unknown hair part: {part!r}")
    sign = {"left": 1.0, "right": -1.0, "none": 0.0}[part]
    return build_short_crop(
        skin, part, replace(cfg, hairline_asymmetry=sign * cfg.hairline_asymmetry)
    )
