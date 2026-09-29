"""Optional, attributed specifications in the existing club-data authority."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Literal, Self
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sidekick.utils.unit_constants import (
    CENTIMETER_TO_METER,
    GRAM_TO_KILOGRAM,
    INCH_TO_METER,
    MILLIMETER_TO_METER,
)

Property = Literal["length", "mass", "loft", "lie", "moi", "cg_distance"]
Component = Literal["head", "shaft", "grip", "assembled"]
ClaimStatus = Literal[
    "unknown", "unverified", "suggested", "estimated", "published", "measured"
]


class CatalogModel(BaseModel):
    """Immutable records reject misspelled fields instead of dropping data."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class SpecificationSource(CatalogModel):
    """Attribution does not imply that public material has an open license."""

    kind: Literal["manufacturer", "measurement", "player", "legacy", "reference"]
    title: str = Field(min_length=1, max_length=500)
    url: str | None = Field(default=None, max_length=2000)
    retrieved_at: datetime | None = None
    license: str = Field(min_length=1, max_length=500)
    locator: str = Field(default="", max_length=1000)
    method: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def check_attribution(self) -> Self:
        if self.url:
            parsed = urlsplit(self.url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError(
                    "Source URL must be an HTTP(S) address without credentials"
                )
        if self.retrieved_at is not None and self.retrieved_at.utcoffset() is None:
            raise ValueError("Retrieval time must include its timezone")
        if self.kind == "manufacturer" and (not self.url or self.retrieved_at is None):
            raise ValueError("Manufacturer sources require URL and retrieval time")
        return self


class ClubIdentity(CatalogModel):
    """A catalog build, distinct from an individual player's bag item."""

    manufacturer: str = Field(default="", max_length=160)
    model: str = Field(min_length=1, max_length=160)
    club_type: Literal["driver", "wood", "hybrid", "iron", "wedge", "putter", "other"]
    number: str | None = Field(default=None, max_length=40)
    release_year: int | None = Field(default=None, ge=1800, le=2200)
    handedness: Literal["left", "right", "unspecified"] = "unspecified"
    region: str = Field(default="", max_length=100)
    build: str = Field(default="standard-unspecified", min_length=1, max_length=500)


_LENGTH_UNITS = {
    "m": 1.0,
    "cm": CENTIMETER_TO_METER,
    "mm": MILLIMETER_TO_METER,
    "in": INCH_TO_METER,
}
_UNITS = {
    "length": _LENGTH_UNITS,
    "cg_distance": _LENGTH_UNITS,
    "mass": {"kg": 1.0, "g": GRAM_TO_KILOGRAM},
    "loft": {"deg": math.pi / 180, "rad": 1.0},
    "lie": {"deg": math.pi / 180, "rad": 1.0},
    "moi": {"kg*m^2": 1.0, "g*cm^2": GRAM_TO_KILOGRAM * CENTIMETER_TO_METER**2},
}
_BOUNDS = {
    "length": (0.0, 2.0),
    "mass": (0.0, 5.0),
    "moi": (0.0, 1.0),
    "cg_distance": (-1.0, 1.0),
    "loft": (-math.pi / 2, math.pi / 2),
    "lie": (0.0, math.pi / 2),
}


class PropertyClaim(CatalogModel):
    """One source's claim; multiple claims stay separate until resolved."""

    property: Property
    component: Component
    value: float | None
    unit: str
    status: ClaimStatus
    source: SpecificationSource
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    axis: str | None = Field(default=None, min_length=1, max_length=200)
    frame: str | None = Field(default=None, min_length=1, max_length=200)
    origin: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str = Field(default="", max_length=2000)

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("A physical quantity cannot be a boolean")
        return value

    @model_validator(mode="after")
    def check_quantity(self) -> Self:
        if self.unit not in _UNITS[self.property]:
            raise ValueError(f"Unsupported {self.property} unit: {self.unit}")
        if (self.value is None) != (self.status == "unknown"):
            raise ValueError(
                "Unknown values must be null; known values need an evidence status"
            )
        value = self.si_value()
        if value is not None:
            lower, upper = _BOUNDS[self.property]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"{self.property} is outside supported SI bounds {lower}, {upper}"
                )
            if self.property in {"length", "mass", "moi"} and value == 0:
                raise ValueError(f"{self.property} must be positive")
        return self

    def si_value(self) -> float | None:
        """Convert original units to m, kg, radians or kg m² without inference."""
        return (
            None
            if self.value is None
            else self.value * _UNITS[self.property][self.unit]
        )


def _canonical(model: BaseModel) -> str:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class ClubRecord(CatalogModel):
    """Versioned attributed build; absent properties remain unknown."""

    identity: ClubIdentity
    claims: tuple[PropertyClaim, ...] = Field(default=(), max_length=128)
    swing_weight: str | None = Field(
        default=None, pattern=r"^[A-G](?:[0-9](?:\.[0-9]+)?)$"
    )
    notes: str = Field(default="", max_length=10000)

    @property
    def catalog_id(self) -> str:
        """Deterministic build ID; measurements affect revision, not build identity."""
        return str(
            uuid5(NAMESPACE_URL, "upstreamdrift:club:1:" + _canonical(self.identity))
        )

    @property
    def revision(self) -> str:
        """Content digest pins the exact source claims used by a capture."""
        return hashlib.sha256(_canonical(self).encode("utf-8")).hexdigest()

    def physical_value(
        self,
        property_name: Property,
        component: Component,
        *,
        allow_estimates: bool = False,
    ) -> float | None:
        """Return unambiguous SI data or reject insufficient evidence.

        MOI is a scalar about its stated axis, frame and origin, never a tensor.
        Estimates need explicit opt-in; suggestions and unverified data stay out.
        """
        if property_name not in _UNITS or component not in {
            "head",
            "shaft",
            "grip",
            "assembled",
        }:
            raise ValueError("Unknown club property or component")
        claims = [
            c
            for c in self.claims
            if c.property == property_name
            and c.component == component
            and c.value is not None
        ]
        if not claims:
            return None
        if len(claims) != 1:
            raise ValueError(
                "Multiple potentially conflicting specifications require explicit resolution"
            )
        claim = claims[0]
        allowed = (
            {"published", "measured", "estimated"}
            if allow_estimates
            else {"published", "measured"}
        )
        if claim.status not in allowed:
            raise ValueError(
                f"Cannot use {claim.status} club property for physical inference"
            )
        if property_name in {"moi", "cg_distance"} and not (
            claim.axis and claim.frame and claim.origin
        ):
            raise ValueError("Physical use requires an explicit axis, frame and origin")
        return claim.si_value()
