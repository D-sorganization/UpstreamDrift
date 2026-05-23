"""User-facing error envelope (epic #5968, Phase 5.1).

Today the API surfaces generic strings like
``"Invalid parameters: ValueError(...)"`` to the UI.  This module
introduces :class:`UserFacingError` — a structured envelope with a
title, what happened, why, how to fix, optional field id and docs
url, and a ``retriable`` flag — and a YAML-backed catalog so the copy
lives in one place that non-coders can review.

Front-end widgets in PyQt6 and React consume the same envelope; when
``field_id`` is present they scroll to and highlight that field.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.contracts import require

_CODE_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*$")
_FIELD_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


class UserFacingErrorError(ValueError):
    """Raised for any invalid UserFacingError construction or input."""


@dataclass(frozen=True, slots=True)
class UserFacingError:
    """A structured, user-readable error.

    The strings in ``title``, ``what_happened``, ``why``, and
    ``how_to_fix`` may contain ``{name}`` substitution placeholders;
    callers fill them with :meth:`format`.
    """

    code: str
    title: str
    what_happened: str
    why: str
    how_to_fix: str
    field_id: str | None
    docs_url: str | None
    retriable: bool

    def __post_init__(self) -> None:
        _validate(self)

    def format(self, **substitutions: Any) -> UserFacingError:
        """Return a new instance with placeholders filled.

        Raises :class:`UserFacingErrorError` if a placeholder is
        missing (DbC at the substitution boundary — silent failure
        here would ship ``{value}`` literals to the user).
        """
        try:
            return replace(
                self,
                title=self.title.format(**substitutions),
                what_happened=self.what_happened.format(**substitutions),
                why=self.why.format(**substitutions),
                how_to_fix=self.how_to_fix.format(**substitutions),
            )
        except KeyError as exc:
            raise UserFacingErrorError(
                f"missing substitution for placeholder {exc.args[0]!r} "
                f"in error {self.code!r}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "what_happened": self.what_happened,
            "why": self.why,
            "how_to_fix": self.how_to_fix,
            "field_id": self.field_id,
            "docs_url": self.docs_url,
            "retriable": self.retriable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> UserFacingError:
        require(
            isinstance(payload, Mapping),
            "UserFacingError.from_dict expects a mapping",
            payload,
        )
        try:
            return cls(
                code=str(payload["code"]),
                title=str(payload["title"]),
                what_happened=str(payload["what_happened"]),
                why=str(payload["why"]),
                how_to_fix=str(payload["how_to_fix"]),
                field_id=_optional_str(payload.get("field_id")),
                docs_url=_optional_str(payload.get("docs_url")),
                retriable=_coerce_bool(
                    payload.get("retriable", False),
                    field_name="retriable",
                ),
            )
        except KeyError as exc:
            raise UserFacingErrorError(
                f"UserFacingError missing required key: {exc.args[0]!r}"
            ) from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise UserFacingErrorError(
        f"{field_name} must be a boolean or boolean-like string, got {value!r}"
    )


def _validate(err: UserFacingError) -> None:
    if not isinstance(err.code, str) or not _CODE_PATTERN.match(err.code):
        raise UserFacingErrorError(
            f"code must match {_CODE_PATTERN.pattern!r}, got {err.code!r}"
        )
    if not err.title:
        raise UserFacingErrorError(f"{err.code}: title must be non-empty")
    if not err.what_happened:
        raise UserFacingErrorError(f"{err.code}: what_happened must be non-empty")
    if not err.why:
        raise UserFacingErrorError(f"{err.code}: why must be non-empty")
    if not err.how_to_fix:
        raise UserFacingErrorError(f"{err.code}: how_to_fix must be non-empty")
    if err.field_id is not None and not _FIELD_ID_PATTERN.match(err.field_id):
        raise UserFacingErrorError(
            f"{err.code}: field_id must be a dotted lowercase id or None, "
            f"got {err.field_id!r}"
        )


class ErrorCatalog:
    """A validated collection of :class:`UserFacingError` instances."""

    __slots__ = ("_errors",)

    def __init__(self, errors: Iterable[UserFacingError]) -> None:
        as_tuple = tuple(errors)
        require(
            all(isinstance(e, UserFacingError) for e in as_tuple),
            "ErrorCatalog only accepts UserFacingError instances",
            as_tuple,
        )
        by_code: dict[str, UserFacingError] = {}
        for e in as_tuple:
            if e.code in by_code:
                raise UserFacingErrorError(f"duplicate error code: {e.code!r}")
            by_code[e.code] = e
        self._errors: dict[str, UserFacingError] = dict(
            sorted(by_code.items(), key=lambda kv: kv[0])
        )

    def get(self, code: str) -> UserFacingError:
        try:
            return self._errors[code]
        except KeyError as exc:
            raise KeyError(f"unknown error code: {code!r}") from exc

    def __contains__(self, code: object) -> bool:
        return isinstance(code, str) and code in self._errors

    def __len__(self) -> int:
        return len(self._errors)

    def __iter__(self):
        return iter(self._errors.values())


def load_error_catalog(path: str | Path) -> ErrorCatalog:
    """Load ``path`` (YAML) and return a validated :class:`ErrorCatalog`.

    Schema::

        errors:
          - code: <snake_case_code>
            title: ...
            what_happened: ...
            why: ...
            how_to_fix: ...
            field_id: <dotted.id> | null
            docs_url: <url> | null
            retriable: true | false
    """
    path = Path(path)
    require(path.is_file(), f"error catalog YAML not found: {path}", path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise UserFacingErrorError(
            f"{path}: top-level YAML must be a mapping, got {type(payload).__name__}"
        )
    entries = payload.get("errors", [])
    if not isinstance(entries, Sequence):
        raise UserFacingErrorError(
            f"{path}: 'errors' must be a sequence, got {type(entries).__name__}"
        )
    built = tuple(UserFacingError.from_dict(entry) for entry in entries)
    return ErrorCatalog(built)


__all__ = [
    "ErrorCatalog",
    "UserFacingError",
    "UserFacingErrorError",
    "load_error_catalog",
]
