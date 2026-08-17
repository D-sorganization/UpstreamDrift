"""Canonical configuration hashing for BunkerShot3D runs (issue #8617).

Two runs are "the same experiment" if their *physics* inputs agree, even when
their seed, thread count or output location differ. Answering that mechanically
needs a canonical serialisation, so this module implements RFC 8785 (JSON
Canonicalization Scheme) style encoding:

* object members sorted by UTF-16 code unit,
* no insignificant whitespace,
* ECMAScript ``Number::toString`` number formatting,
* ``NaN``/``Infinity`` rejected (the JSON ``allow_nan=False`` rule).

Two digests are derived from it:

``config_hash``
    SHA-256 over the whole configuration -- identifies the exact run.
``physics_hash``
    SHA-256 over the configuration minus :data:`PHYSICS_EXCLUDED_FIELDS` --
    identifies the experiment, so a seed sweep shares one physics hash.

Known deviation from RFC 8785: Python integers are emitted exactly, including
magnitudes above 2**53 where ECMAScript would lose precision. Configuration
values in this package are small, and exactness is the safer failure mode for a
hash.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import Enum
from typing import Any

__all__ = [
    "PHYSICS_EXCLUDED_FIELDS",
    "FieldClass",
    "canonical_json",
    "classify_field",
    "config_hash",
    "leaf_field_paths",
    "physics_hash",
    "strip_excluded_fields",
]

#: Fields that cannot change the physics of a run. Entries are either a fully
#: dotted path (``"output.rate_hz"``) or a bare leaf name (``"seed"``) that
#: matches at any depth. Kept as a single frozenset so there is exactly one
#: place to look when deciding whether a new field is an experiment input.
#:
#: ``output.rate_hz``/``output.downsample_grains`` are *recording* controls: how
#: densely the run is sampled, not what it simulates. A solver must therefore
#: never derive its integration timestep from ``output.rate_hz`` (finding B30 --
#: the Chrono driver did exactly that); if it does, the two hashes stop meaning
#: what they claim.
PHYSICS_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        # Seeding -- deliberately excluded so a seed sweep shares a physics hash.
        "seed",
        "seeds",
        "random_seed",
        "rng_seed",
        "master_seed",
        "entropy",
        # Diagnostics / logging.
        "log_level",
        "log_file",
        "verbose",
        "progress",
        # Execution resources.
        "n_threads",
        "num_threads",
        "thread_count",
        "n_workers",
        "num_workers",
        "n_jobs",
        "device",
        # Output and scratch locations.
        "output_dir",
        "output_path",
        "output_file",
        "output_prefix",
        "result_path",
        "results_dir",
        "work_dir",
        "scratch_dir",
        "checkpoint_dir",
        # Recording density (see note above).
        "output.rate_hz",
        "output.downsample_grains",
    }
)


class FieldClass(Enum):
    """Whether a configuration field participates in :func:`physics_hash`."""

    PHYSICS = "physics"
    EXCLUDED = "excluded"


def classify_field(path: str) -> FieldClass:
    """Classify a dotted configuration path.

    Args:
        path: Dotted path such as ``"contact_model.friction_coefficient"``.

    Returns:
        :attr:`FieldClass.EXCLUDED` when the full path or its leaf name is in
        :data:`PHYSICS_EXCLUDED_FIELDS`, otherwise :attr:`FieldClass.PHYSICS`.

    Raises:
        TypeError: If ``path`` is not a string.
        ValueError: If ``path`` is empty.
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a str, got {type(path).__name__}")
    if not path:
        raise ValueError("path must be non-empty")
    leaf = path.rsplit(".", 1)[-1]
    if path in PHYSICS_EXCLUDED_FIELDS or leaf in PHYSICS_EXCLUDED_FIELDS:
        return FieldClass.EXCLUDED
    return FieldClass.PHYSICS


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------

#: Below this magnitude ECMAScript switches to exponential notation.
_EXP_LOWER = 1e-6
#: At or above this magnitude ECMAScript switches to exponential notation.
_EXP_UPPER = 1e21

_STRING_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _encode_string(value: str) -> str:
    """Return the RFC 8785 JSON string form of ``value``."""
    out = ['"']
    for char in value:
        escape = _STRING_ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _encode_float(value: float) -> str:
    """Return the ECMAScript ``Number::toString`` form of ``value``."""
    if not math.isfinite(value):
        raise ValueError(
            f"value {value!r} is not finite; canonical JSON forbids NaN/Infinity "
            "(allow_nan=False)"
        )
    if value == 0.0:
        return "0"  # ECMAScript String(-0) is "0".
    magnitude = abs(value)
    if magnitude.is_integer() and magnitude < _EXP_UPPER:
        return str(int(value))
    if _EXP_LOWER <= magnitude < _EXP_UPPER:
        text = format(Decimal(repr(value)), "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    mantissa, _, exponent = repr(value).partition("e")
    if not exponent:  # pragma: no cover - CPython always uses e-notation here
        return mantissa
    sign = "-" if exponent.startswith("-") else "+"
    digits = exponent.lstrip("+-").lstrip("0") or "0"
    return f"{mantissa}e{sign}{digits}"


def _encode_int(value: int) -> str:
    """Return the canonical form of an integer."""
    return str(value)


def _sorted_keys(mapping: Mapping[str, Any]) -> list[str]:
    """Return ``mapping``'s keys ordered by UTF-16 code unit, as RFC 8785 requires."""
    keys = []
    for key in mapping:
        if not isinstance(key, str):
            raise TypeError(
                f"object keys must be str for canonical JSON, got {type(key).__name__}"
            )
        keys.append(key)
    return sorted(keys, key=lambda key: key.encode("utf-16-be"))


def _encode(value: Any) -> str:
    """Recursively encode ``value`` as canonical JSON."""
    if value is None:
        return "null"
    if isinstance(value, bool):  # before int: bool is an int subclass
        return "true" if value else "false"
    if isinstance(value, Enum):
        return _encode(value.value)
    if isinstance(value, str):
        return _encode_string(value)
    if isinstance(value, int):
        return _encode_int(value)
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, Mapping):
        items = ",".join(
            f"{_encode_string(key)}:{_encode(value[key])}"
            for key in _sorted_keys(value)
        )
        return "{" + items + "}"
    if isinstance(value, (bytes, bytearray)):
        raise TypeError("bytes are not representable in canonical JSON")
    if isinstance(value, os.PathLike):
        return _encode_string(os.fspath(value))
    if hasattr(value, "item") and hasattr(value, "shape") and value.shape == ():
        return _encode(value.item())  # numpy scalar
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _encode(value.tolist())  # numpy array
    if isinstance(value, Sequence):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    raise TypeError(
        f"{type(value).__name__} is not representable in canonical JSON; "
        "convert it at the configuration boundary"
    )


def canonical_json(value: Any) -> str:
    """Serialise ``value`` as RFC 8785 style canonical JSON.

    Args:
        value: Any JSON-compatible structure. Mappings, sequences, strings,
            integers, floats, booleans, ``None``, :class:`enum.Enum`,
            :class:`os.PathLike` and numpy scalars/arrays are supported.

    Returns:
        The canonical serialisation: sorted keys, no insignificant whitespace.

    Raises:
        TypeError: If ``value`` contains a type with no canonical form.
        ValueError: If ``value`` contains ``NaN`` or an infinity.
    """
    return _encode(value)


# ---------------------------------------------------------------------------
# Configuration traversal
# ---------------------------------------------------------------------------


def _as_mapping(config: Any) -> Mapping[str, Any]:
    """Return ``config`` as a plain mapping.

    Args:
        config: A pydantic model (anything exposing ``model_dump``) or mapping.

    Returns:
        The mapping form of the configuration.

    Raises:
        TypeError: If ``config`` is neither a pydantic model nor a mapping.
    """
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        result: Mapping[str, Any] = dump()
        return result
    if isinstance(config, Mapping):
        return config
    raise TypeError(
        f"config must be a pydantic model or mapping, got {type(config).__name__}"
    )


def leaf_field_paths(config: Any) -> tuple[str, ...]:
    """Return every leaf field of ``config`` as a sorted tuple of dotted paths.

    Args:
        config: A pydantic model or mapping.

    Returns:
        Dotted paths of all non-mapping leaves, sorted lexicographically.
    """
    paths: list[str] = []

    def walk(node: Mapping[str, Any], prefix: str) -> None:
        for key in node:
            path = f"{prefix}.{key}" if prefix else str(key)
            child = node[key]
            if isinstance(child, Mapping):
                walk(child, path)
            else:
                paths.append(path)

    walk(_as_mapping(config), "")
    return tuple(sorted(paths))


def strip_excluded_fields(config: Any) -> dict[str, Any]:
    """Return ``config`` without any field classified as excluded.

    Args:
        config: A pydantic model or mapping.

    Returns:
        A nested ``dict`` holding only :attr:`FieldClass.PHYSICS` leaves.
    """

    def walk(node: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        kept: dict[str, Any] = {}
        for key in node:
            path = f"{prefix}.{key}" if prefix else str(key)
            if classify_field(path) is FieldClass.EXCLUDED:
                continue
            child = node[key]
            kept[str(key)] = walk(child, path) if isinstance(child, Mapping) else child
        return kept

    return walk(_as_mapping(config), "")


def _sha256(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: Any) -> str:
    """Return the SHA-256 digest of the whole configuration.

    Args:
        config: A pydantic model or mapping.

    Returns:
        64-character lowercase hex digest identifying the exact run inputs.
    """
    return _sha256(canonical_json(_as_mapping(config)))


def physics_hash(config: Any) -> str:
    """Return the SHA-256 digest of the physics-relevant configuration.

    Args:
        config: A pydantic model or mapping.

    Returns:
        64-character lowercase hex digest shared by runs that differ only in
        :data:`PHYSICS_EXCLUDED_FIELDS` (seed, threads, output location, ...).
    """
    return _sha256(canonical_json(strip_excluded_fields(config)))
