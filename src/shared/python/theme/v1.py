"""Versioned theme contract — ``theme/v1``.

This module is the *single source of truth* for the role-token vocabulary
that every theme implementation in the fleet must speak. Historically three
separate modules each invented their own palette/role concept:

* ``src/shared/python/theme`` — the canonical fleet ``ThemeManager`` palettes
  (named themes, the 14 base + semantic role tokens).
* ``src/shared/python/pendulum_simulator/gui/theme.py`` — a hand-rolled dark
  palette of ``BG_*`` / ``TEXT_*`` / ``ACCENT_*`` constants plus severity
  colours and a diagnostics stylesheet.

``theme/v1`` reconciles them under one explicit, versioned contract:

* :data:`CONTRACT_VERSION` — the contract identifier (``"v1"``).
* :data:`ROLE_TOKENS` — the ordered tuple of role-token names a palette may
  define. These are exactly the canonical base + semantic colour keys, so the
  contract stays in lock-step with :mod:`src.shared.python.theme.colors`.
* :class:`RolePalette` — a frozen, validated mapping of role token → hex
  colour. Adapter modules (e.g. the pendulum GUI theme) declare their palette
  as a ``RolePalette`` instead of loose module-level constants, so every
  consumer reads the same role names regardless of which app it lives in.
* :func:`role_palette` — build a ``RolePalette`` for any canonical fleet theme.
* :func:`severity_palette` — the canonical diagnostics severity → colour map.

The contract deliberately covers *palette role tokens* only. Per-shape
geometry styling (``body_part_viz.ShapeTheme``: opacity, edge width, shading)
is an orthogonal concern and is not part of this colour-role contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .colors import (
    BUILTIN_THEMES,
    SEMANTIC_COLOR_KEYS,
    THEME_COLOR_KEYS,
    is_valid_hex_color,
    normalise_hex_color,
)

__all__ = [
    "CONTRACT_VERSION",
    "ROLE_TOKENS",
    "RolePalette",
    "SEVERITY_TOKENS",
    "role_palette",
    "severity_palette",
]

# The version identifier for this theme contract. Bump (and add a ``v2``
# module) when the role-token vocabulary changes in a backwards-incompatible
# way; never mutate ``v1`` semantics in place.
CONTRACT_VERSION = "v1"

# The full role-token vocabulary: the 14 required base tokens followed by the
# optional semantic tokens. Sourced from the canonical colour module so this
# contract cannot silently drift from the ThemeManager palettes.
ROLE_TOKENS: tuple[str, ...] = THEME_COLOR_KEYS + SEMANTIC_COLOR_KEYS

# Severity levels recognised by diagnostics surfaces fleet-wide.
SEVERITY_TOKENS: tuple[str, ...] = ("info", "warning", "error", "critical")

# Canonical severity → colour map. Diagnostics views across the fleet read
# these so a "warning" looks the same everywhere.
_SEVERITY_PALETTE: Mapping[str, str] = {
    "info": "#6080c0",
    "warning": "#c0a040",
    "error": "#d06060",
    "critical": "#e03030",
}


@dataclass(frozen=True)
class RolePalette:
    """A validated mapping of contract role token → hex colour.

    Parameters
    ----------
    name:
        Human-readable palette name (e.g. ``"Dark"`` or ``"Pendulum Dark"``).
    tokens:
        Mapping of role-token name → hex colour string. Every key must be a
        member of :data:`ROLE_TOKENS`; every value must be a valid hex colour.
        Values are normalised to ``#rrggbb`` lower-case on construction.

    Notes
    -----
    Frozen and hashable. The required base tokens
    (:data:`~src.shared.python.theme.colors.THEME_COLOR_KEYS`) must all be
    present so consumers can rely on them resolving.
    """

    name: str
    tokens: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.tokens, Mapping):
            raise TypeError(
                f"tokens must be a mapping; got {type(self.tokens).__name__}"
            )

        allowed = set(ROLE_TOKENS)
        normalised: dict[str, str] = {}
        for key, value in self.tokens.items():
            if key not in allowed:
                raise ValueError(
                    f"unknown role token {key!r}; valid tokens are {ROLE_TOKENS}"
                )
            if not isinstance(value, str) or not is_valid_hex_color(value):
                raise ValueError(
                    f"token {key!r} must be a valid hex colour; got {value!r}"
                )
            normalised[key] = normalise_hex_color(value)

        missing = [k for k in THEME_COLOR_KEYS if k not in normalised]
        if missing:
            raise ValueError(
                f"palette {self.name!r} is missing required base tokens: {missing}"
            )

        # Replace the mapping with the normalised, immutable copy.
        object.__setattr__(self, "tokens", dict(normalised))

    def __getitem__(self, token: str) -> str:
        return self.tokens[token]

    def get(self, token: str, default: str | None = None) -> str | None:
        """Return the colour for *token*, or *default* if it is not defined."""
        return self.tokens.get(token, default)

    def as_dict(self) -> dict[str, str]:
        """Return a plain ``dict`` copy of the role tokens."""
        return dict(self.tokens)


def role_palette(theme_name: str) -> RolePalette:
    """Build a :class:`RolePalette` for a canonical fleet theme.

    Parameters
    ----------
    theme_name:
        Name of a theme registered in
        :data:`~src.shared.python.theme.colors.BUILTIN_THEMES`.

    Returns
    -------
    RolePalette
        Palette populated with every role token the theme defines.

    Raises
    ------
    ValueError
        If *theme_name* is not a registered theme.
    """
    if theme_name not in BUILTIN_THEMES:
        raise ValueError(
            f"unknown theme {theme_name!r}; registered themes: {sorted(BUILTIN_THEMES)}"
        )
    theme = BUILTIN_THEMES[theme_name]
    tokens = {k: v for k, v in theme.items() if k in set(ROLE_TOKENS)}
    return RolePalette(name=theme_name, tokens=tokens)


def severity_palette() -> dict[str, str]:
    """Return the canonical diagnostics severity → colour map."""
    return dict(_SEVERITY_PALETTE)
