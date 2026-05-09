"""High-level orchestrator for the anthropometrics subsystem.

End-to-end glue around the canonical building blocks already on
``main`` (contracts, types, estimators, readers, persistence,
engine adapters). Closes #4822 — Child #10 of EPIC #4797.

The single public entry point :func:`run_pipeline` performs the
full chain in one call:

1. Load a C3D mocap file via :mod:`ezc3d`. If subject height /
   mass were not supplied, fall back to whatever is encoded in the
   C3D ``SUBJECT_INFO`` / ``PROCESSING`` groups.
2. Estimate available per-segment lengths from the mocap marker
   trajectories (best-effort — a default segment-definition list
   is applied; absent markers are silently skipped).
3. Materialise a :class:`SubjectAnthropometrics` from the chosen
   regression estimator (de Leva / Dempster / Zatsiorsky-Seluyanov).
4. Export to one or more physics-engine native formats by looking
   up :data:`ADAPTER_REGISTRY`.
5. Persist the canonical record to ``output_dir/subject.json``.
6. Generate a self-contained ``output_dir/report.html`` validation
   report covering mass closure, inertia spectral health, and
   length closure.

Design-by-contract: invalid arguments raise :class:`ValueError`,
a missing mocap file raises :class:`FileNotFoundError`, the
output directory is created automatically.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._subject_anthropometrics import SubjectAnthropometrics
from .engine_adapters import ADAPTER_REGISTRY
from .estimators.from_de_leva import DeLevaEstimator
from .estimators.from_dempster import DempsterEstimator
from .estimators.from_mocap import SegmentDef, estimate_segment_lengths_from_markers
from .estimators.from_zatsiorsky import ZatsiorskyEstimator
from .persistence import save_subject
from .readers.c3d_subject_info import read_c3d_subject_metadata

if TYPE_CHECKING:
    from .contracts import Estimator

__all__ = ["run_pipeline"]

logger = logging.getLogger(__name__)

EstimatorName = Literal["de_leva", "dempster", "zatsiorsky"]

_VALID_ESTIMATORS: tuple[str, ...] = ("de_leva", "dempster", "zatsiorsky")


# Per-engine output extension. The canonical ADAPTER_REGISTRY key is
# its ``engine_name``. ``mujoco`` is accepted as an alias for the
# MuJoCo-based MyoSuite adapter so callers can request the engine by
# the more common identifier.
_ENGINE_EXTENSIONS: dict[str, str] = {
    "drake": "urdf",
    "pinocchio": "urdf",
    "myosuite": "xml",
    "mujoco": "xml",
    "opensim": "osim",
    "simscape": "mat",
}

_ENGINE_ALIASES: dict[str, str] = {"mujoco": "myosuite"}


# A small default mocap segment-definition list. Markers absent
# from a given C3D file are simply skipped — the lengths produced
# here are informational only; the regression estimator owns the
# canonical segment lengths in the returned record.
_DEFAULT_MOCAP_SEGMENTS: tuple[SegmentDef, ...] = (
    SegmentDef("left_upper_arm", "LShoulderTop", "LElbowOut"),
    SegmentDef("left_forearm", "LElbowOut", "LWristTop"),
    SegmentDef("right_upper_arm", "RShoulderTop", "RElbowOut"),
    SegmentDef("right_forearm", "RElbowOut", "RWristTop"),
    SegmentDef("left_thigh", "WaistLeft", "LKneeOut"),
    SegmentDef("left_shin", "LKneeOut", "LAnkleOut"),
    SegmentDef("right_thigh", "WaistRight", "RKneeOut"),
    SegmentDef("right_shin", "RKneeOut", "RAnkleOut"),
)


# --------------------------------------------------------------------------- #
# Public entry point.                                                         #
# --------------------------------------------------------------------------- #
def run_pipeline(
    mocap_file: Path | str,
    *,
    subject_height_m: float | None = None,
    subject_mass_kg: float | None = None,
    estimator: EstimatorName = "de_leva",
    target_engines: Sequence[str] = ("drake", "mujoco", "pinocchio", "opensim"),
    output_dir: Path | str,
) -> SubjectAnthropometrics:
    """Drive the full mocap → anthropometrics → engine-export pipeline.

    Parameters
    ----------
    mocap_file
        Path to a ``.c3d`` motion-capture file.
    subject_height_m, subject_mass_kg
        Optional subject scalars in SI units. When omitted, the
        C3D ``SUBJECT_INFO`` / ``PROCESSING`` parameter groups are
        consulted; if those are also missing the call raises
        :class:`ValueError`.
    estimator
        Regression estimator to apply: one of ``"de_leva"``,
        ``"dempster"``, ``"zatsiorsky"``.
    target_engines
        Iterable of engine identifiers as registered in
        :data:`ADAPTER_REGISTRY`. ``"mujoco"`` is accepted as an
        alias for the MyoSuite (MuJoCo-based) adapter. Unknown
        engines emit a warning and are skipped — callers may
        request optional engines without precomputing what is
        installed.
    output_dir
        Destination directory for engine exports, the canonical
        ``subject.json`` record, and the ``report.html`` validation
        report. Created automatically (including parents) if it
        does not yet exist.

    Returns
    -------
    SubjectAnthropometrics
        The fully-validated canonical record produced by the
        chosen estimator.

    Raises
    ------
    FileNotFoundError
        If *mocap_file* does not exist.
    ValueError
        If *estimator* is unknown, if no subject height / mass can
        be resolved, or if a downstream contract fails.
    """
    mocap_path, output_path, sex, age_years, subject_id = _validate_and_resolve_inputs(
        mocap_file=mocap_file,
        subject_height_m=subject_height_m,
        subject_mass_kg=subject_mass_kg,
        estimator=estimator,
        output_dir=output_dir,
    )
    height_m, mass_kg = _resolve_subject_scalars(
        mocap_path,
        provided_height=subject_height_m,
        provided_mass=subject_mass_kg,
    )

    # 2. Best-effort mocap-derived segment lengths (informational).
    mocap_lengths = _estimate_mocap_lengths_safely(mocap_path)

    # 3. Apply the chosen regression estimator.
    record = _build_subject_record(
        estimator_name=estimator,
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        sex=sex,
        age_years=age_years,
    )

    # 4. Export per engine.
    _export_to_engines(record, target_engines, output_path)

    # 5. Persist canonical JSON.
    save_subject(record, output_path / "subject.json")

    # 6. Validation report.
    report_html = _build_validation_report(record, mocap_lengths)
    (output_path / "report.html").write_text(report_html, encoding="utf-8")

    return record


# --------------------------------------------------------------------------- #
# Validation / argument resolution.                                           #
# --------------------------------------------------------------------------- #
def _validate_and_resolve_inputs(
    *,
    mocap_file: Path | str,
    subject_height_m: float | None,
    subject_mass_kg: float | None,
    estimator: str,
    output_dir: Path | str,
) -> tuple[Path, Path, str, float | None, str]:
    """Validate every public argument and resolve subject metadata.

    Returns ``(mocap_path, output_path, sex, age_years, subject_id)``.
    The numeric ``subject_height_m`` / ``subject_mass_kg`` are not
    returned here — they are resolved later because the C3D file may
    need to be opened twice (once for metadata, once for markers).
    """
    if estimator not in _VALID_ESTIMATORS:
        raise ValueError(
            f"estimator must be one of {_VALID_ESTIMATORS}, got {estimator!r}"
        )

    mocap_path = Path(mocap_file)
    if not mocap_path.exists():
        raise FileNotFoundError(f"mocap file not found: {mocap_path}")

    for label, value in (
        ("subject_height_m", subject_height_m),
        ("subject_mass_kg", subject_mass_kg),
    ):
        if value is None:
            continue
        if not (
            isinstance(value, (int, float)) and np.isfinite(float(value)) and value > 0
        ):
            raise ValueError(f"{label} must be a positive finite number, got {value!r}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pull session-level metadata (sex, age_years, subject_id) from
    # the C3D file once. Numeric height / mass are handled separately
    # by ``_resolve_subject_scalars`` so missing-but-overridden
    # cases don't raise here.
    try:
        meta = read_c3d_subject_metadata(mocap_path)
        sex = meta.sex.value
        age_years = meta.age_years
        subject_id = meta.subject_id or _slugify(mocap_path.stem)
    except (ImportError, OSError) as error:  # pragma: no cover
        logger.warning(
            "Falling back to defaults — could not read C3D metadata from %s: %s",
            mocap_path,
            error,
        )
        sex = "unspecified"
        age_years = None
        subject_id = _slugify(mocap_path.stem)

    return mocap_path, output_path, sex, age_years, subject_id


def _resolve_subject_scalars(
    mocap_path: Path,
    *,
    provided_height: float | None,
    provided_mass: float | None,
) -> tuple[float, float]:
    """Return ``(height_m, mass_kg)``, falling back to the C3D metadata."""
    if provided_height is not None and provided_mass is not None:
        return float(provided_height), float(provided_mass)

    try:
        meta = read_c3d_subject_metadata(mocap_path)
    except (ImportError, OSError) as error:
        raise ValueError(
            "subject_height_m and subject_mass_kg were not supplied and the "
            f"C3D metadata could not be read from {mocap_path}: {error}"
        ) from error

    height = float(provided_height) if provided_height is not None else meta.height_m
    mass = float(provided_mass) if provided_mass is not None else meta.mass_kg

    if height is None or mass is None:
        raise ValueError(
            "subject height/mass were not supplied and the C3D file at "
            f"{mocap_path} does not encode them in SUBJECT_INFO/PROCESSING. "
            "Pass subject_height_m and subject_mass_kg explicitly."
        )
    return float(height), float(mass)


def _slugify(value: str) -> str:
    """Return a non-empty filesystem-safe identifier derived from *value*."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "subject"


# --------------------------------------------------------------------------- #
# Mocap segment-length helpers.                                               #
# --------------------------------------------------------------------------- #
def _estimate_mocap_lengths_safely(mocap_path: Path) -> dict[str, float]:
    """Return whatever segment lengths can be derived from the C3D file.

    The returned mapping is informational — the regression estimator
    owns the canonical segment lengths on the returned
    :class:`SubjectAnthropometrics`. This function never raises:
    failures (missing markers, C3D-read errors, ``ezc3d`` import
    error, ...) are logged and the method returns an empty dict.
    """
    try:
        markers = _load_marker_trajectories(mocap_path)
    except (ImportError, OSError, ValueError, KeyError) as error:
        logger.warning(
            "Could not load mocap markers for length estimation from %s: %s",
            mocap_path,
            error,
        )
        return {}

    if not markers:
        return {}

    available = {
        seg
        for seg in _DEFAULT_MOCAP_SEGMENTS
        if seg.proximal_marker in markers and seg.distal_marker in markers
    }
    if not available:
        return {}

    try:
        return estimate_segment_lengths_from_markers(
            markers, list(available), method="median_distance"
        )
    except ValueError as error:
        logger.warning("Mocap segment-length estimation skipped: %s", error)
        return {}


def _load_marker_trajectories(mocap_path: Path) -> dict[str, np.ndarray]:
    """Open *mocap_path* via ``ezc3d`` and return ``{label: (T, 3) array}``.

    NaN samples are preserved — :func:`estimate_segment_lengths_from_markers`
    is NaN-tolerant.
    """
    try:
        import ezc3d  # local import keeps the package importable without ezc3d
    except ImportError as error:  # pragma: no cover - exercised via mocks
        raise ImportError(
            "ezc3d is required to load C3D marker trajectories. "
            "Install it with: pip install ezc3d"
        ) from error

    c3d_data = ezc3d.c3d(str(mocap_path))
    params = c3d_data["parameters"]
    points = np.asarray(c3d_data["data"]["points"])  # (4, N, T)
    if points.ndim != 3 or points.shape[0] < 3:
        return {}

    labels_raw = params["POINT"]["LABELS"]["value"]
    labels = [str(label).strip() for label in labels_raw]

    units_raw = params["POINT"]["UNITS"]["value"]
    units = str(units_raw[0]).strip().lower() if units_raw else "mm"
    scale = 1.0 if units.startswith("m") and not units.startswith("mm") else 1.0e-3

    n_markers = points.shape[1]
    n_frames = points.shape[2]
    out: dict[str, np.ndarray] = {}
    for idx, label in enumerate(labels):
        if idx >= n_markers or not label:
            continue
        traj = np.empty((n_frames, 3), dtype=float)
        traj[:, 0] = points[0, idx, :] * scale
        traj[:, 1] = points[1, idx, :] * scale
        traj[:, 2] = points[2, idx, :] * scale
        out[label] = traj
    return out


# --------------------------------------------------------------------------- #
# Estimator dispatch.                                                         #
# --------------------------------------------------------------------------- #
def _select_estimator(name: str) -> Estimator:
    """Return the concrete :class:`Estimator` for *name*."""
    if name == "de_leva":
        return DeLevaEstimator()
    if name == "dempster":
        return DempsterEstimator()
    if name == "zatsiorsky":
        return ZatsiorskyEstimator()
    raise ValueError(  # pragma: no cover - guarded earlier
        f"estimator must be one of {_VALID_ESTIMATORS}, got {name!r}"
    )


def _build_subject_record(
    *,
    estimator_name: str,
    subject_id: str,
    height_m: float,
    mass_kg: float,
    sex: str,
    age_years: float | None,
) -> SubjectAnthropometrics:
    """Apply the chosen estimator and return a :class:`SubjectAnthropometrics`."""
    estimator = _select_estimator(estimator_name)
    return estimator.estimate(
        subject_id=subject_id,
        height_m=height_m,
        mass_kg=mass_kg,
        sex=sex,
        age_years=age_years,
    )


# --------------------------------------------------------------------------- #
# Engine export.                                                              #
# --------------------------------------------------------------------------- #
def _export_to_engines(
    record: SubjectAnthropometrics,
    target_engines: Sequence[str],
    output_dir: Path,
) -> None:
    """Export *record* to every engine in *target_engines* it can.

    Unknown engines log a warning and are skipped. The on-disk
    file name is ``output_dir / f"{engine}.{ext}"`` where ``ext``
    is taken from :data:`_ENGINE_EXTENSIONS` (falling back to
    ``"out"`` for engines registered without a known extension).
    """
    for raw_name in target_engines:
        canonical = _ENGINE_ALIASES.get(raw_name, raw_name)
        if canonical not in ADAPTER_REGISTRY:
            logger.warning(
                "Engine %r not in ADAPTER_REGISTRY (%s) — skipping export.",
                raw_name,
                sorted(ADAPTER_REGISTRY),
            )
            continue
        adapter = ADAPTER_REGISTRY[canonical]
        ext = _ENGINE_EXTENSIONS.get(raw_name, _ENGINE_EXTENSIONS.get(canonical, "out"))
        out_path = output_dir / f"{raw_name}.{ext}"
        adapter.export(record, out_path)


# --------------------------------------------------------------------------- #
# Validation report.                                                          #
# --------------------------------------------------------------------------- #
_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Anthropometrics validation report — {subject_id}</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2em; color: #222; }}
h1 {{ font-size: 1.6em; }}
h2 {{ font-size: 1.2em; margin-top: 1.6em; }}
table {{ border-collapse: collapse; margin-top: 0.5em; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.ok {{ color: #117a0d; font-weight: bold; }}
.fail {{ color: #b30000; font-weight: bold; }}
.note {{ color: #555; font-style: italic; }}
code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Anthropometrics validation report</h1>
<p>
Subject <code>{subject_id}</code> — height {height_m:.4f} m,
mass {mass_kg:.4f} kg, source method <code>{source_method}</code>,
{n_segments} segments.
</p>

<h2>1. Mass closure</h2>
<p>
Sum of segment masses ÷ subject mass =
<strong>{mass_ratio:.6f}</strong> (target 1.000000 ± 1%).
Status: <span class="{mass_class}">{mass_status}</span>.
</p>

<h2>2. Inertia spectral check</h2>
<p>
Per-segment principal moments and triangle-inequality status.
</p>
<table>
<thead>
<tr><th>Segment</th><th>I<sub>1</sub></th><th>I<sub>2</sub></th><th>I<sub>3</sub></th><th>PD</th><th>Triangle</th></tr>
</thead>
<tbody>
{segment_rows}
</tbody>
</table>
<p>
Overall positive-definite: <span class="{pd_class}">{pd_status}</span>.
Overall triangle inequality: <span class="{tri_class}">{tri_status}</span>.
</p>

<h2>3. Length closure (sanity)</h2>
<p>
Sum of axial segment lengths ÷ subject height =
<strong>{length_ratio:.6f}</strong>. No hard threshold —
this is a sanity figure only.
</p>

<h2>4. Mocap-derived segment lengths (informational)</h2>
{mocap_block}

</body>
</html>
"""


def _build_validation_report(
    record: SubjectAnthropometrics,
    mocap_lengths: dict[str, float],
) -> str:
    """Render the canonical HTML validation report.

    The report is a deterministic string so callers can snapshot it
    against ``tests/fixtures/anthropometrics/expected_report.html``.
    All numeric fields are formatted with explicit precision; no
    timestamps or non-deterministic content are included.
    """
    seg_masses = [float(props.mass_kg) for _, props in record.segments]
    total_mass = float(sum(seg_masses))
    mass_ratio = total_mass / float(record.mass_kg)
    mass_ok = abs(mass_ratio - 1.0) <= 0.01

    seg_lengths = [float(props.length_m) for _, props in record.segments]
    length_ratio = float(sum(seg_lengths)) / float(record.height_m)

    rows: list[str] = []
    pd_ok = True
    tri_ok = True
    for seg_name, props in record.segments:
        eigenvalues = sorted(
            float(v) for v in np.linalg.eigvalsh(np.asarray(props.inertia_tensor))
        )
        i1, i2, i3 = eigenvalues
        pd_seg = i1 > 0
        tri_seg = (
            (i1 + i2) >= i3 - 1e-12
            and (i1 + i3) >= i2 - 1e-12
            and (i2 + i3) >= i1 - 1e-12
        )
        pd_ok = pd_ok and pd_seg
        tri_ok = tri_ok and tri_seg
        rows.append(
            "<tr>"
            f"<td>{escape(seg_name)}</td>"
            f"<td>{i1:.6e}</td>"
            f"<td>{i2:.6e}</td>"
            f"<td>{i3:.6e}</td>"
            f'<td class="{"ok" if pd_seg else "fail"}">'
            f"{'OK' if pd_seg else 'FAIL'}</td>"
            f'<td class="{"ok" if tri_seg else "fail"}">'
            f"{'OK' if tri_seg else 'FAIL'}</td>"
            "</tr>"
        )

    if mocap_lengths:
        mocap_rows = "".join(
            f"<tr><td>{escape(name)}</td><td>{length:.6f}</td></tr>"
            for name, length in sorted(mocap_lengths.items())
        )
        mocap_block = (
            "<table><thead><tr><th>Segment</th>"
            "<th>Length (m)</th></tr></thead>"
            f"<tbody>{mocap_rows}</tbody></table>"
        )
    else:
        mocap_block = (
            '<p class="note">No mocap-derived segment lengths were '
            "available for this subject (markers absent or C3D could "
            "not be read).</p>"
        )

    return _REPORT_TEMPLATE.format(
        subject_id=escape(record.subject_id),
        height_m=float(record.height_m),
        mass_kg=float(record.mass_kg),
        source_method=escape(record.source_method),
        n_segments=len(record.segments),
        mass_ratio=mass_ratio,
        mass_status="OK" if mass_ok else "FAIL",
        mass_class="ok" if mass_ok else "fail",
        segment_rows="\n".join(rows),
        pd_status="OK" if pd_ok else "FAIL",
        pd_class="ok" if pd_ok else "fail",
        tri_status="OK" if tri_ok else "FAIL",
        tri_class="ok" if tri_ok else "fail",
        length_ratio=length_ratio,
        mocap_block=mocap_block,
    )
