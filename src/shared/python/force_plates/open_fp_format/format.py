"""Constants of the open-fp swing file layout (OF-02, epic #13061).

Everything here is taken from the documented file format (open-fp
``PROJECT_INIT.md`` section 9.7, commit :data:`SOURCE_REVISION`), not from any
source code of that AGPL-3.0 project.  The format carries no version field, so
the pinned revision is the only version anchor.
"""

from __future__ import annotations

import re

__all__ = [
    "ARRAY_KEYS",
    "CHANNEL_IDS",
    "CSV_HEADER",
    "CSV_LOAD_DECIMALS",
    "EVENT_EDGE_TOL_S",
    "EVENT_NAMES",
    "HTTP_COP_QUANTUM_MM",
    "HTTP_EVENT_EDGE_TOL_S",
    "HTTP_EVENT_QUANTUM_S",
    "HTTP_FORCE_QUANTUM_N",
    "HTTP_LEAD_PCT_QUANTUM",
    "HTTP_META_KEYS",
    "HTTP_SOURCE_FORMAT",
    "HTTP_SUMMARY_KEYS",
    "HTTP_TIME_QUANTUM_S",
    "ID_PATTERN",
    "IMPACT_SOURCES",
    "LEAD_PCT_TOL",
    "LEAD_PCT_TOL_CSV",
    "LEAD_PCT_TOL_HTTP",
    "LOAD_SLACK_N",
    "LOAD_SLACK_N_CSV",
    "LOAD_SLACK_N_HTTP",
    "MAX_CSV_BYTES",
    "MAX_JSON_BYTES",
    "MAX_NPZ_BYTES",
    "META_KEYS",
    "MM_PER_M",
    "SOURCE_COP_MIN_LOAD_N",
    "SOURCE_FORMAT",
    "SOURCE_REVISION",
]

#: open-fp revision whose documented format this reader implements.
SOURCE_REVISION = "53d682ce70b34bd2d85918b210ebdc81937e33f8"
#: ``PlateProvenance.source_format`` label (the format is unversioned).
SOURCE_FORMAT = "open-fp-swing (unversioned)"

#: Swing id = file stem: board-local ``YYYYMMDD-HHMMSS-mmm`` plus optional ``-k``.
ID_PATTERN = re.compile(r"^\d{8}-\d{6}-\d{3}(-\d{1,4})?$")

#: Corner channel order of the ``forces`` array (plate L then plate R).
CHANNEL_IDS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")

#: Exact NPZ key set.  Shapes are ``(n,)`` or ``(n, k)``; ``k`` listed here.
#: Units: ``t`` s; ``forces``, ``fz_*`` N; ``cop_*`` mm; ``lead_pct`` percent.
ARRAY_KEYS: dict[str, int | None] = {
    "t": None,
    "forces": len(CHANNEL_IDS),
    "fz_L": None,
    "fz_R": None,
    "cop_L": 2,
    "cop_R": 2,
    "cop_C": 2,
    "lead_pct": None,
}

#: Exact top-level JSON key set; ``metrics`` and ``path`` are opaque and ignored.
META_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "created",
        "golfer",
        "bw_n",
        "t0_host",
        "events",
        "metrics",
        "impact_source",
        "path",
    }
)
#: Exact event names; each maps to seconds from ``t[0]`` or ``null``.
EVENT_NAMES: frozenset[str] = frozenset(
    {
        "motion_start",
        "max_trail",
        "transition",
        "max_lead",
        "finish",
        "impact",
        "trigger",
    }
)
#: Slack [s] when testing an event against the trial end: ``t[-1] - t[0]`` is a
#: float difference, so an event stored at exactly the last sample can exceed it.
EVENT_EDGE_TOL_S = 1e-9
#: Allowed ``impact_source`` values (besides ``null``).
IMPACT_SOURCES: frozenset[str] = frozenset({"shot", "manual"})

#: A plate reports no CoP below this load [N] (a property of the source plates).
SOURCE_COP_MIN_LOAD_N = 20.0
#: Millimetres per metre (the files store CoP in mm; divide, which is correctly
#: rounded, rather than multiplying by an inexact 1e-3).
MM_PER_M = 1000.0

#: Size caps, checked before a file is parsed (and, for the npz, on its expanded
#: size and every array header).  A swing is tens of kB (6 s at 200 Hz with 18
#: columns is about 100 kB); these leave a wide margin while keeping a hostile
#: file from exhausting memory.
MAX_JSON_BYTES = 1 << 20  # 1 MiB
MAX_NPZ_BYTES = 8 << 20  # 8 MiB
MAX_CSV_BYTES = 8 << 20  # 8 MiB

#: CSV export header (exact) and its fixed number of decimals for loads/CoP/%.
CSV_HEADER = (
    "t_s,lead_pct,fz_L_n,fz_R_n,cop_L_x_mm,cop_L_y_mm,cop_R_x_mm,cop_R_y_mm,"
    "cop_C_x_mm,cop_C_y_mm,L1_n,L2_n,L3_n,L4_n,R1_n,R2_n,R3_n,R4_n"
)
CSV_LOAD_DECIMALS = 2
#: Rounding quantum of CSV loads [N] and CoP [mm] (both to CSV_LOAD_DECIMALS).
CSV_FORCE_QUANTUM_N = 10.0**-CSV_LOAD_DECIMALS
CSV_COP_QUANTUM_MM = 10.0**-CSV_LOAD_DECIMALS

#: ``lead_pct`` cross-check tolerance in percentage points for NPZ data: float
#: round-off only (the source stores float64 computed from the same loads).
LEAD_PCT_TOL = 1e-6
#: Load slack [N] around the 20 N threshold inside which the NaN/finite status
#: of ``lead_pct`` is not enforced (NPZ: float round-off).
LOAD_SLACK_N = 1e-6

_HALF_LOAD_ULP_N = 0.5 * 10.0**-CSV_LOAD_DECIMALS
#: CSV loads are rounded to 0.01 N, so each load is off by up to 0.005 N and a
#: total by up to 0.01 N: the status of ``lead_pct`` is ambiguous that close to
#: the threshold.
LOAD_SLACK_N_CSV = 2.0 * _HALF_LOAD_ULP_N
#: CSV ``lead_pct`` tolerance [percentage points]: its own 0.005 rounding plus the
#: propagated load rounding, to first order ``d(100 L / T) <= 100 * 0.005 *
#: (L + R) / T**2 = 0.5 / T`` at the smallest defined total ``T = 20 N``, plus
#: the NPZ float allowance.
LEAD_PCT_TOL_CSV = (
    0.5 * 10.0**-CSV_LOAD_DECIMALS
    + 100.0 * _HALF_LOAD_ULP_N / SOURCE_COP_MIN_LOAD_N
    + LEAD_PCT_TOL
)

# ------------------------------------------------------------------ HTTP --
#: ``PlateProvenance.source_format`` label of the LAN JSON form (OF-06).
HTTP_SOURCE_FORMAT = "open-fp-http-json"
#: Top-level keys of ``GET /api/swings/{id}``: the file metadata plus ``arrays``.
HTTP_META_KEYS: frozenset[str] = META_KEYS | {"arrays"}
#: Keys of one summary object of ``GET /api/swings`` (no ``t0_host``/``arrays``).
HTTP_SUMMARY_KEYS: frozenset[str] = META_KEYS - {"t0_host"}

#: Quantisation of the JSON form, from the documented rounding in open-fp's
#: ``swing.py::to_dict``: ``t`` to 1e-4 s, forces [N] and CoP [mm] to 0.1,
#: ``lead_pct`` to 0.01 %, events to 1e-3 s.  Each value is rounded to the
#: nearest multiple, so the error is at most half of it.
HTTP_TIME_QUANTUM_S = 1e-4
HTTP_FORCE_QUANTUM_N = 0.1
HTTP_COP_QUANTUM_MM = 0.1
HTTP_LEAD_PCT_QUANTUM = 0.01
HTTP_EVENT_QUANTUM_S = 1e-3

#: Event range slack [s] for the JSON form: an event rounded to 1e-3 s may exceed
#: the span ``t[-1] - t[0]`` (rounded to 1e-4 s) by half an event quantum plus
#: the time rounding of both ends.
HTTP_EVENT_EDGE_TOL_S = 0.5 * HTTP_EVENT_QUANTUM_S + HTTP_TIME_QUANTUM_S

_HTTP_HALF_LOAD_N = 0.5 * HTTP_FORCE_QUANTUM_N
#: As for the CSV: loads off by up to half a quantum, a total by up to a full one.
LOAD_SLACK_N_HTTP = 2.0 * _HTTP_HALF_LOAD_N
#: ``lead_pct`` tolerance [percentage points] for the JSON form: its own rounding
#: plus the propagated load rounding ``100 * (0.05 N) / 20 N`` (see the CSV
#: derivation above) plus the NPZ float allowance.
LEAD_PCT_TOL_HTTP = (
    0.5 * HTTP_LEAD_PCT_QUANTUM
    + 100.0 * _HTTP_HALF_LOAD_N / SOURCE_COP_MIN_LOAD_N
    + LEAD_PCT_TOL
)
