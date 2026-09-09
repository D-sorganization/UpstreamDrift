"""Player-owned clubs and immutable capture evidence in the club-data authority."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal, Self
from uuid import uuid4

from pydantic import Field, model_validator

from .catalog import CatalogModel, ClubIdentity, ClubRecord, PropertyClaim
from .catalog_sources import with_player_overrides


class PlayerClub(CatalogModel):
    """Keep the catalog base and explicit player overrides separately inspectable."""

    club_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=100
    )
    label: str = Field(min_length=1, max_length=200)
    base: ClubRecord
    overrides: tuple[PropertyClaim, ...] = Field(default=(), max_length=128)
    notes: str = Field(default="", max_length=10000)
    archived: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def validate_overrides(self) -> Self:
        self.effective_record()
        return self

    def effective_record(self) -> ClubRecord:
        """Project player claims without discarding their original catalog source."""
        return with_player_overrides(self.base, self.overrides)

    @property
    def identity(self) -> ClubIdentity:
        """Expose the build identity without coupling screens to storage nesting."""
        return self.base.identity

    @property
    def revision(self) -> str:
        """Pin the entire player record, including source, notes and nickname."""
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class PlayerBag(CatalogModel):
    schema_version: Literal["upstreamdrift/player-clubs/1"] = (
        "upstreamdrift/player-clubs/1"
    )
    clubs: tuple[PlayerClub, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [club.club_id for club in self.clubs]
        if len(ids) != len(set(ids)):
            raise ValueError("Player bag contains duplicate club IDs")
        return self

    @property
    def revision(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()


class CaptureClubSnapshot(CatalogModel):
    """A capture owns its copy; catalog/bag changes cannot mutate prior evidence."""

    schema_version: Literal["upstreamdrift/capture-club/1"] = (
        "upstreamdrift/capture-club/1"
    )
    capture_id: str = Field(min_length=1, max_length=200)
    selected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    club: PlayerClub
    club_revision: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_revision(self) -> Self:
        if self.club.revision != self.club_revision:
            raise ValueError("Selected club does not match its saved revision")
        if self.selected_at.utcoffset() is None:
            raise ValueError("Selection time must include its timezone")
        if self.club.archived:
            raise ValueError("An archived club cannot be newly selected for a capture")
        return self
