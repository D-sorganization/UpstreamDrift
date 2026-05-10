"""Simscape Multibody :class:`EngineAdapter` — emits a MAT-file blob.

The 3D Golf Model uses Simscape Multibody (MATLAB) and accepts
inertial parameters at model load time via a ``.mat`` workspace
file. **No existing Python loader / writer was found** in
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/`` at
the time this adapter was written, so the on-disk schema below is
defined by this adapter and documented as the canonical
exchange format until a downstream MATLAB consumer is wired up.

On-disk schema (v1)
-------------------
``scipy.io.savemat`` writes a struct-of-arrays. The top-level keys
are:

* ``schema_version``    – uint8, currently ``1``.
* ``subject_id``        – char array.
* ``height_m``          – double scalar.
* ``mass_kg``           – double scalar.
* ``sex``               – char array.
* ``age_years``         – double scalar (``nan`` if not provided).
* ``source_method``     – char array.
* ``segment_names``     – cell array of char arrays (one per segment).
* ``body_part_ids``     – cell array of char arrays.
* ``proximal_markers``  – cell array of char arrays (empty cell ⇒ ``None``).
* ``distal_markers``    – cell array of char arrays.
* ``segment_methods``   – cell array of char arrays.
* ``length_m``          – ``(N,)`` double array.
* ``mass``              – ``(N,)`` double array (segment mass).
* ``source_height_m``   – ``(N,)`` double array.
* ``source_mass_kg``    – ``(N,)`` double array.
* ``com_xyz``           – ``(N, 3)`` double array.
* ``inertia``           – ``(N, 3, 3)`` double array.

The schema is intentionally flat so a MATLAB consumer can ingest
it with a single ``load('subject.mat')`` call and pick the columns
it needs without needing to evaluate any Python objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from .._subject_anthropometrics import SubjectAnthropometrics
from ..segment_properties import SegmentProperties

_SENTINEL_NONE = ""  # empty MATLAB char array marks an absent optional marker.


class SimscapeAdapter:
    """Round-trip a :class:`SubjectAnthropometrics` through a Simscape ``.mat``."""

    engine_name: str = "simscape"

    def export(
        self, anthropometrics: SubjectAnthropometrics, output_path: Path
    ) -> None:
        if not isinstance(anthropometrics, SubjectAnthropometrics):
            raise TypeError(
                "anthropometrics must be a SubjectAnthropometrics, got "
                f"{type(anthropometrics).__name__}"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(anthropometrics.segments)
        segment_names = [name for name, _ in anthropometrics.segments]
        props_list = [props for _, props in anthropometrics.segments]

        com = np.zeros((n, 3), dtype=float)
        inertia = np.zeros((n, 3, 3), dtype=float)
        length_m = np.zeros(n, dtype=float)
        mass = np.zeros(n, dtype=float)
        source_height = np.zeros(n, dtype=float)
        source_mass = np.zeros(n, dtype=float)
        for i, props in enumerate(props_list):
            com[i] = np.asarray(props.com_xyz_m, dtype=float)
            inertia[i] = np.asarray(props.inertia_tensor, dtype=float)
            length_m[i] = float(props.length_m)
            mass[i] = float(props.mass_kg)
            source_height[i] = float(props.source_subject_height_m)
            source_mass[i] = float(props.source_subject_mass_kg)

        payload: dict[str, Any] = {
            "schema_version": np.uint8(1),
            "subject_id": anthropometrics.subject_id,
            "height_m": float(anthropometrics.height_m),
            "mass_kg": float(anthropometrics.mass_kg),
            "sex": anthropometrics.sex,
            "age_years": float(
                anthropometrics.age_years
                if anthropometrics.age_years is not None
                else np.nan
            ),
            "source_method": anthropometrics.source_method,
            "segment_names": np.array(segment_names, dtype=object).reshape(1, -1),
            "body_part_ids": np.array(
                [p.body_part_id for p in props_list], dtype=object
            ).reshape(1, -1),
            "proximal_markers": np.array(
                [
                    (
                        p.proximal_marker
                        if p.proximal_marker is not None
                        else _SENTINEL_NONE
                    )
                    for p in props_list
                ],
                dtype=object,
            ).reshape(1, -1),
            "distal_markers": np.array(
                [
                    p.distal_marker if p.distal_marker is not None else _SENTINEL_NONE
                    for p in props_list
                ],
                dtype=object,
            ).reshape(1, -1),
            "segment_methods": np.array(
                [p.source_method for p in props_list], dtype=object
            ).reshape(1, -1),
            "length_m": length_m,
            "mass": mass,
            "source_height_m": source_height,
            "source_mass_kg": source_mass,
            "com_xyz": com,
            "inertia": inertia,
        }
        savemat(str(output_path), payload, do_compression=False, oned_as="row")

    def import_back(self, input_path: Path) -> SubjectAnthropometrics:
        input_path = Path(input_path)
        raw = loadmat(str(input_path), squeeze_me=False, mat_dtype=False)

        subject_id = _scalar_str(raw, "subject_id")
        height_m = float(_scalar_num(raw, "height_m"))
        mass_kg = float(_scalar_num(raw, "mass_kg"))
        sex = _scalar_str(raw, "sex")
        source_method = _scalar_str(raw, "source_method")
        age_raw = float(_scalar_num(raw, "age_years"))
        age_years: float | None = None if np.isnan(age_raw) else age_raw

        segment_names = _cell_strings(raw, "segment_names")
        body_part_ids = _cell_strings(raw, "body_part_ids")
        proximal_markers = _cell_strings(raw, "proximal_markers")
        distal_markers = _cell_strings(raw, "distal_markers")
        segment_methods = _cell_strings(raw, "segment_methods")
        length_m = np.asarray(raw["length_m"], dtype=float).ravel()
        seg_mass = np.asarray(raw["mass"], dtype=float).ravel()
        src_height = np.asarray(raw["source_height_m"], dtype=float).ravel()
        src_mass = np.asarray(raw["source_mass_kg"], dtype=float).ravel()
        com = np.asarray(raw["com_xyz"], dtype=float)
        inertia = np.asarray(raw["inertia"], dtype=float)

        n = len(segment_names)
        if not (
            len(body_part_ids)
            == n
            == len(segment_methods)
            == len(length_m)
            == len(seg_mass)
            == len(src_height)
            == len(src_mass)
            == com.shape[0]
            == inertia.shape[0]
        ):
            raise ValueError(f"inconsistent column counts in Simscape MAT {input_path}")

        segments: list[tuple[str, SegmentProperties]] = []
        for i in range(n):
            prox = proximal_markers[i] or None
            dist = distal_markers[i] or None
            props = SegmentProperties(
                name=segment_names[i],
                body_part_id=body_part_ids[i],
                length_m=float(length_m[i]),
                proximal_marker=prox,
                distal_marker=dist,
                mass_kg=float(seg_mass[i]),
                com_xyz_m=com[i].astype(float),
                inertia_tensor=inertia[i].astype(float),
                source_method=segment_methods[i],
                source_subject_height_m=float(src_height[i]),
                source_subject_mass_kg=float(src_mass[i]),
            )
            segments.append((segment_names[i], props))

        return SubjectAnthropometrics(
            subject_id=subject_id,
            height_m=height_m,
            mass_kg=mass_kg,
            segments=tuple(segments),
            source_method=source_method,
            age_years=age_years,
            sex=sex,
        )


# --------------------------------------------------------------------------- #
# Helpers — MATLAB scipy.io quirks all in one place.                          #
# --------------------------------------------------------------------------- #
def _scalar_str(raw: dict[str, Any], key: str) -> str:
    if key not in raw:
        raise ValueError(f"Simscape MAT missing required key {key!r}")
    val = raw[key]
    # scipy.io.loadmat wraps strings as 1x1 ndarray of dtype '<U...'.
    arr = np.asarray(val)
    if arr.dtype.kind == "U":
        return str(arr.item() if arr.size == 1 else arr.flat[0])
    if arr.dtype.kind == "O":
        item = arr.flat[0]
        return str(item)
    raise ValueError(f"Simscape MAT key {key!r} is not a string: {val!r}")


def _scalar_num(raw: dict[str, Any], key: str) -> float:
    if key not in raw:
        raise ValueError(f"Simscape MAT missing required key {key!r}")
    arr = np.asarray(raw[key], dtype=float).ravel()
    if arr.size != 1:
        raise ValueError(f"Simscape MAT key {key!r} must be scalar, got {arr!r}")
    return float(arr[0])


def _cell_strings(raw: dict[str, Any], key: str) -> list[str]:
    if key not in raw:
        raise ValueError(f"Simscape MAT missing required key {key!r}")
    arr = np.asarray(raw[key]).reshape(-1)
    out: list[str] = []
    for entry in arr:
        cell = np.asarray(entry)
        if cell.size == 0:
            # 0-length char array (the sentinel we wrote for ``None``).
            out.append("")
        elif cell.dtype.kind == "U":
            out.append(str(cell.item() if cell.size == 1 else cell.flat[0]))
        else:
            out.append(str(cell.flat[0]))
    return out
