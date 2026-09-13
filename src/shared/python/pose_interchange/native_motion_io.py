"""Versioned native motion JSON; independent of initial-pose file formats.

JSON carries SI primitive-conjugate efforts and relative joint-base rotations,
not raw actuator polynomials or complete world body poses. It proves conversion
and persistence only, not engine trajectory equivalence.
"""

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, fields, is_dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .native_joint_state import NativeManifoldState, NativeRotationGroup, RotationState
from .native_motion_sequence import NativeMotionSequence

_FILE_CONVENTION = "native-motion-file-v1"


@dataclass(frozen=True)
class NativeMotionDocument:
    """Loaded sequence and optional raw artifact hash, distinct from model identity.

    raw_model_sha256 identifies original file bytes when supplied. It is optional
    provenance, not proof those bytes are available or equivalent to the canonical
    specification fingerprint checked by NativeJointStateAdapter.
    """

    sequence: NativeMotionSequence
    raw_model_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, NativeMotionSequence):
            raise ValueError("Expected validated native motion sequence")
        value = self.raw_model_sha256
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Expected optional raw model SHA256")


def _json_value(value: Any) -> Any:
    """Serialize owned dataclasses without deepcopying their immutable mappings."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {name: _json_value(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _object(value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Missing or unexpected native motion fields")
    return dict(value)


def _record(value: Any, kind: Any) -> dict[str, Any]:
    return _object(value, {field.name for field in fields(kind)})


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def load_native_motion(path: Path | str) -> NativeMotionDocument:
    """Read strict versioned JSON and delegate semantic validation to the envelope.

    File/permission errors propagate. Unknown fields or versions, duplicate keys,
    invalid numerical values and malformed inventories raise ValueError.
    """
    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique,
        parse_constant=_nonfinite,
    )
    try:
        document = _object(payload, {"file_convention", "raw_model_sha256", "sequence"})
        if document["file_convention"] != _FILE_CONVENTION:
            raise ValueError("Unsupported native motion file convention")
        sequence = _record(document["sequence"], NativeMotionSequence)
        sequence["rotation_groups"] = tuple(
            NativeRotationGroup(**_record(group, NativeRotationGroup))
            for group in sequence["rotation_groups"]
        )
        states = []
        for value in sequence["states"]:
            state = _record(value, NativeManifoldState)
            if not isinstance(state["rotations"], dict) or not isinstance(
                state["scalars"], dict
            ):
                raise ValueError("Expected named native state mappings")
            state["rotations"] = {
                name: RotationState(**_record(rotation, RotationState))
                for name, rotation in state["rotations"].items()
            }
            states.append(NativeManifoldState(**state))
        sequence["states"] = tuple(states)
        return NativeMotionDocument(
            NativeMotionSequence(**sequence), document["raw_model_sha256"]
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"Malformed native motion document: {error}") from error


def save_native_motion(
    sequence: NativeMotionSequence,
    path: Path | str,
    *,
    raw_model_sha256: str | None = None,
) -> Path:
    """Atomically replace a file using a unique temporary sibling.

    The parent directory must exist. Validation/encoding complete before writing.
    A write or replace failure propagates and leaves the previous target intact.
    No initial-pose serializer or engine-specific array convention is involved.
    """
    document = NativeMotionDocument(sequence, raw_model_sha256)
    payload = {"file_convention": _FILE_CONVENTION, **_json_value(document)}
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
    return target
