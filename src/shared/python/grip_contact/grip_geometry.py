"""One source of truth for club grip geometry (#13891).

Defines GripGeometry with sourced presets (legacy spec standoff, men's standard,
midsize, ladies, undersize), buildup layer accounting, and USGA equipment rules
envelope validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

STATION_UPPER_M: Final[float] = 2.0 * 0.0254  # 2 inches below cap (0.0508 m)
STATION_LOWER_M: Final[float] = 6.0 * 0.0254  # 6 inches below cap (0.1524 m)
USGA_MAX_GRIP_DIAMETER_M: Final[float] = 0.04445  # 1.75 in (44.45 mm)

BUILDUP_INCREMENTS_M: Final[dict[str, float]] = {
    "golf_pride": 0.0254 / 64.0,  # 1 wrap approx +1/64 in (0.000396875 m)
    "hireko": 0.007 * 0.0254,  # 0.007 in per layer (0.0001778 m)
}

VALID_PRESETS: Final[tuple[str, ...]] = (
    "legacy_spec_standoff",
    "mens_standard",
    "midsize",
    "ladies",
    "undersize",
    "custom",
)

VALID_TAPERS: Final[tuple[str, ...]] = (
    "standard",
    "reduced",
    "zero",
    "reverse",
)


@dataclass(frozen=True)
class GripGeometry:
    """Canonical club grip geometry definition.

    Station positions s are measured axially downwards from the grip butt cap:
      s = 0.0508 m (2 in below cap): upper hand measurement station.
      s = 0.1524 m (6 in below cap): lower hand measurement station.

    Preconditions:
        Diameters must be positive, finite, and <= USGA max (44.45 mm).
    """

    d_upper_m: float
    d_lower_m: float
    length_m: float = 0.27
    cap_offset_m: float = 0.0
    preset: str = "legacy_spec_standoff"
    taper: str = "standard"
    buildup_layers: int = 0
    buildup_source: str = "golf_pride"
    source: str = "RHandStandoff"
    vendor: str = "UpstreamDrift legacy"

    def __post_init__(self) -> None:
        if self.preset not in VALID_PRESETS:
            raise ValueError(
                f"preset must be one of {VALID_PRESETS}, got {self.preset!r}"
            )
        if self.taper not in VALID_TAPERS:
            raise ValueError(f"taper must be one of {VALID_TAPERS}, got {self.taper!r}")
        if self.buildup_source not in BUILDUP_INCREMENTS_M:
            raise ValueError(
                f"buildup_source must be one of {tuple(BUILDUP_INCREMENTS_M.keys())}, got {self.buildup_source!r}"
            )
        for name in ("d_upper_m", "d_lower_m", "length_m", "cap_offset_m"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if self.d_upper_m <= 0.0 or self.d_lower_m <= 0.0:
            raise ValueError("diameters must be strictly positive")
        if self.length_m <= 0.0:
            raise ValueError("length_m must be strictly positive")
        if self.buildup_layers < 0:
            raise ValueError("buildup_layers must be non-negative")

    @property
    def buildup_increment_m(self) -> float:
        """Diameter increment per buildup layer for the chosen vendor source."""
        return BUILDUP_INCREMENTS_M[self.buildup_source]

    @property
    def effective_d_upper_m(self) -> float:
        """Effective diameter at upper station including buildup tape layers."""
        return self.d_upper_m + self.buildup_layers * self.buildup_increment_m

    @property
    def effective_d_lower_m(self) -> float:
        """Effective diameter at lower station including buildup tape layers."""
        return self.d_lower_m + self.buildup_layers * self.buildup_increment_m

    def radius_at(self, s_m: float) -> float:
        """Return the effective grip radius at axial distance s_m below cap."""
        if not math.isfinite(s_m):
            raise ValueError("s_m must be finite")
        if self.taper == "zero":
            return 0.5 * self.effective_d_upper_m

        r1 = 0.5 * self.effective_d_upper_m
        r2 = 0.5 * self.effective_d_lower_m
        slope = (r2 - r1) / (STATION_LOWER_M - STATION_UPPER_M)
        return r1 + slope * (s_m - STATION_UPPER_M)

    def with_buildup(self, layers: int, source: str = "golf_pride") -> GripGeometry:
        """Return a copy of this geometry with specified buildup layers."""
        return GripGeometry(
            d_upper_m=self.d_upper_m,
            d_lower_m=self.d_lower_m,
            length_m=self.length_m,
            cap_offset_m=self.cap_offset_m,
            preset=self.preset,
            taper=self.taper,
            buildup_layers=layers,
            buildup_source=source,
            source=f"{self.source}; {layers} layers ({source})",
            vendor=self.vendor,
        )


GRIP_GEOMETRY_PRESETS: Final[dict[str, GripGeometry]] = {
    "legacy_spec_standoff": GripGeometry(
        d_upper_m=0.0254,
        d_lower_m=0.0254,
        length_m=0.27,
        preset="legacy_spec_standoff",
        taper="zero",
        source="RHandStandoff placement z = -0.0127 m (half an inch)",
        vendor="UpstreamDrift legacy",
    ),
    "mens_standard": GripGeometry(
        d_upper_m=0.02286,  # 0.900 in
        d_lower_m=0.01981,  # 0.780 in
        length_m=0.27,
        preset="mens_standard",
        taper="standard",
        source="Hireko Golf fitting guide; Mitchell Golf sizing chart",
        vendor="Hireko/Mitchell",
    ),
    "midsize": GripGeometry(
        d_upper_m=0.02286 + 0.0254 / 16.0,  # 0.9625 in (+1/16 in)
        d_lower_m=0.01981 + 0.0254 / 16.0,
        length_m=0.27,
        preset="midsize",
        taper="standard",
        source="Golf Pride install FAQ; Mitchell Golf sizing chart",
        vendor="Golf Pride",
    ),
    "undersize": GripGeometry(
        d_upper_m=0.02286 - 0.0254 / 64.0,  # -1/64 in
        d_lower_m=0.01981 - 0.0254 / 64.0,
        length_m=0.27,
        preset="undersize",
        taper="standard",
        source="Hireko Golf fitting guide",
        vendor="Hireko",
    ),
    "ladies": GripGeometry(
        d_upper_m=0.02159,  # 0.850 in
        d_lower_m=0.01854,  # 0.730 in
        length_m=0.27,
        preset="ladies",
        taper="standard",
        source="Hireko Golf fitting guide",
        vendor="Hireko",
    ),
}


def get_grip_preset(name: str = "legacy_spec_standoff") -> GripGeometry:
    """Retrieve canonical GripGeometry preset by name."""
    if name not in GRIP_GEOMETRY_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}; must be one of {tuple(GRIP_GEOMETRY_PRESETS.keys())}"
        )
    return GRIP_GEOMETRY_PRESETS[name]


def validate_usga_rules(geom: GripGeometry) -> None:
    """Validate GripGeometry against USGA Equipment Rules (Part 2, Rule 4: The Grip).

    Rules:
      - Max diameter <= 44.45 mm (1.75 in).
      - Cross section must be circular.
      - Taper must be continuous/monotonic (no bulges or waists).
    """
    if geom.effective_d_upper_m > USGA_MAX_GRIP_DIAMETER_M:
        raise ValueError(
            f"Upper diameter {geom.effective_d_upper_m * 1000:.2f} mm exceeds USGA limit "
            f"{USGA_MAX_GRIP_DIAMETER_M * 1000:.2f} mm"
        )
    if geom.effective_d_lower_m > USGA_MAX_GRIP_DIAMETER_M:
        raise ValueError(
            f"Lower diameter {geom.effective_d_lower_m * 1000:.2f} mm exceeds USGA limit "
            f"{USGA_MAX_GRIP_DIAMETER_M * 1000:.2f} mm"
        )
    if geom.taper in ("standard", "reduced"):
        if geom.effective_d_lower_m > geom.effective_d_upper_m:
            raise ValueError(
                f"Standard taper cannot have d_lower ({geom.effective_d_lower_m:.4f} m) > "
                f"d_upper ({geom.effective_d_upper_m:.4f} m); violates continuous taper rules."
            )
    elif geom.taper == "zero":
        if not math.isclose(
            geom.effective_d_lower_m, geom.effective_d_upper_m, rel_tol=1e-6
        ):
            raise ValueError(
                "Zero taper requires d_lower == d_upper to machine precision."
            )


def check_standoff_compatibility(
    geom: GripGeometry,
    standoff_m: float,
    station_s: float = STATION_UPPER_M,
    tol_m: float = 1e-4,
    strict: bool = False,
) -> bool:
    """Check that hand standoff matches the grip radius at the hand station."""
    r = geom.radius_at(station_s)
    diff = abs(r - standoff_m)
    if diff > tol_m:
        if strict:
            raise ValueError(
                f"Hand standoff mismatch at station {station_s:.4f} m: "
                f"radius={r:.4f} m, standoff={standoff_m:.4f} m (diff={diff:.4f} m > tol={tol_m:.4f} m)"
            )
        return False
    return True
