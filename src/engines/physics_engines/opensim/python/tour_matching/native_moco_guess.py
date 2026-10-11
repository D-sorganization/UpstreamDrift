"""Source-bound native numerical state seed for the existing Moco builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


GUESS_FILENAME = "native_initial_guess.sto"
_PREPARATION = "post-initSystem-native-equilibrated-constant-numerical-seed"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clock(values: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    clock = np.asarray(values, dtype=np.float64)
    if (
        clock.ndim != 1
        or len(clock) < 2
        or not np.isfinite(clock).all()
        or clock[0] != 0.0
        or not np.all(np.diff(clock) > 0)
    ):
        raise ValueError("Native numerical guess clock must start at zero and advance")
    return clock


def _source(model_path: Path, expected_sha256: str) -> bytes:
    payload = Path(model_path).read_bytes()
    if _sha(payload) != expected_sha256:
        raise ValueError("Native numerical guess source SHA-256 differs")
    return payload


def _state_names(model: Any) -> tuple[str, ...]:
    names = model.getStateVariableNames()
    result = tuple(str(names.get(index)) for index in range(names.getSize()))
    if (
        not result
        or len(set(result)) != len(result)
        or not all(name.startswith("/") for name in result)
    ):
        raise ValueError("Native source has incomplete or duplicate state names")
    return result


def _read_table(
    model: Any, guess_path: Path, clock: NDArray[np.float64]
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    import opensim as osim

    try:
        table = osim.TimeSeriesTable(str(guess_path))
        names = tuple(str(name) for name in table.getColumnLabels())
        actual_time = np.asarray(table.getIndependentColumn(), dtype=np.float64)
        states = np.asarray(table.getMatrix().to_numpy(), dtype=np.float64)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Native numerical guess table is invalid") from exc
    native_names = _state_names(model)
    if len(names) != len(native_names) or set(names) != set(native_names):
        raise ValueError("Native numerical guess state names differ from source")
    if not np.array_equal(actual_time, clock):
        raise ValueError("Native numerical guess clock differs from declaration")
    if states.shape != (len(clock), len(names)) or not np.isfinite(states).all():
        raise ValueError("Native numerical guess has missing or nonfinite states")
    return names, states


def _write_sto(
    path: Path,
    names: tuple[str, ...],
    values: NDArray[np.float64],
    clock: NDArray[np.float64],
    runtime_version: str,
) -> None:
    """Write double values with enough digits for exact native readback."""
    state_row = "\t".join(format(float(value), ".17g") for value in values)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("DataType=double\nversion=3\n")
        stream.write(f"OpenSimVersion={runtime_version}\nendheader\n")
        stream.write("time\t" + "\t".join(names) + "\n")
        for instant in clock:
            stream.write(f"{float(instant):.17g}\t{state_row}\n")


@dataclass(frozen=True)
class NativeMocoGuessReceipt:
    """Byte-bound numerical seed, deliberately not a biomechanical readiness claim."""

    source_sha256: str
    loaded_native_model_sha256: str
    runtime_version: str
    provider_sha256: str
    state_names_sha256: str
    state_values_sha256: str
    clock_sha256: str
    guess_sha256: str
    state_count: int
    knot_count: int
    native_equilibration_changes: int | None
    preparation: str = _PREPARATION
    resource_scope: str = "entrypoint-xml-only; external assets unverified"
    qualification: str = "numerical-seed-only"


def _receipt(
    model: Any,
    names: tuple[str, ...],
    states: NDArray[np.float64],
    clock: NDArray[np.float64],
    path: Path,
    source_sha256: str,
    runtime_version: str,
) -> NativeMocoGuessReceipt:
    names_payload = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
    return NativeMocoGuessReceipt(
        source_sha256,
        _sha(model.dump().encode("utf-8")),
        runtime_version,
        _sha(Path(__file__).read_bytes()),
        _sha(names_payload.encode("utf-8")),
        _sha(np.asarray(states, dtype="<f8").tobytes()),
        _sha(np.asarray(clock, dtype="<f8").tobytes()),
        _sha(path.read_bytes()),
        len(names),
        len(clock),
        None,
    )


def audit_native_moco_guess(
    model_path: Path,
    guess_path: Path,
    expected_source_sha256: str,
    time_seconds: Sequence[float] | NDArray[np.float64],
) -> NativeMocoGuessReceipt:
    """Check state order, finite values and exact clock against loaded native model."""
    import opensim as osim

    source = _source(Path(model_path), expected_source_sha256)
    clock = _clock(time_seconds)
    model = osim.Model(str(model_path))
    model.initSystem()
    names, states = _read_table(model, Path(guess_path), clock)
    if Path(model_path).read_bytes() != source:
        raise ValueError("Native numerical guess source changed during audit")
    return _receipt(
        model,
        names,
        states,
        clock,
        Path(guess_path),
        expected_source_sha256,
        str(osim.GetVersionAndDate()),
    )


def materialize_native_moco_guess(
    model_path: Path,
    expected_source_sha256: str,
    time_seconds: Sequence[float] | NDArray[np.float64],
    output_dir: Path,
) -> NativeMocoGuessReceipt:
    """Write one constant native-equilibrated numerical seed; never infer bounds."""
    import opensim as osim

    source = _source(Path(model_path), expected_source_sha256)
    clock = _clock(time_seconds)
    model = osim.Model(str(model_path))
    state = model.initSystem()  # Native initialization includes assembly.
    names = _state_names(model)
    before = np.asarray(
        [model.getStateVariableValue(state, name) for name in names], dtype=np.float64
    )
    model.equilibrateMuscles(state)
    values = np.asarray(
        [model.getStateVariableValue(state, name) for name in names], dtype=np.float64
    )
    if not np.isfinite(before).all() or not np.isfinite(values).all():
        raise ValueError("Native source initialization has nonfinite states")
    if Path(model_path).read_bytes() != source:
        raise ValueError("Native numerical guess source changed during preparation")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / GUESS_FILENAME
    if target.exists() or (directory / "native_initial_guess.json").exists():
        raise FileExistsError("Native numerical guess output already exists")
    with TemporaryDirectory(dir=directory) as temporary:
        interim = Path(temporary) / GUESS_FILENAME
        _write_sto(interim, names, values, clock, str(osim.GetVersionAndDate()))
        actual_names, actual_states = _read_table(model, interim, clock)
        repeated = np.broadcast_to(values, actual_states.shape)
        if actual_names != names or not np.array_equal(actual_states, repeated):
            deviation = float(np.max(np.abs(actual_states - repeated)))
            raise ValueError(
                f"Native numerical guess changed in STO roundtrip: {deviation}"
            )
        receipt = _receipt(
            model,
            names,
            actual_states,
            clock,
            interim,
            expected_source_sha256,
            str(osim.GetVersionAndDate()),
        )
        receipt = replace(
            receipt,
            native_equilibration_changes=int(np.count_nonzero(before != values)),
        )
        if Path(model_path).read_bytes() != source:
            raise ValueError("Native numerical guess source changed before export")
        os.replace(interim, target)
    (directory / "native_initial_guess.json").write_text(
        json.dumps(asdict(receipt), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


__all__ = [
    "GUESS_FILENAME",
    "NativeMocoGuessReceipt",
    "audit_native_moco_guess",
    "materialize_native_moco_guess",
]
