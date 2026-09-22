"""Full-swing qualification matrix for all six engines (MS-104, #10378).

Maintains explicit engine × club × gate rows for flagship full-body models,
evaluates them against the matched-swing ledger and evidence links, and
fail-closes release status when any required row is incomplete.

This module is a software contract: it never invents a six-engine native pass.
Native engine work remains owned by the named per-engine issues recorded on
each row's blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from src.shared.python.contracts import ensure, postcondition, precondition, require
from src.shared.python.motion_matching.acceptance import Horizon
from src.shared.python.motion_matching.contact_law import CONFORMANCE_VERSION
from src.shared.python.motion_matching.ledger_schema import Ledger, LedgerRow
from src.shared.python.motion_matching.matching_strategy import ALL_ENGINES

__all__ = [
    "SCHEMA_VERSION",
    "TARGET_ENGINES",
    "FullSwingQualificationError",
    "NamedBlocker",
    "QualificationEvidenceLinks",
    "QualificationRow",
    "QualificationRowSpec",
    "FullSwingQualificationReport",
    "build_required_row_specs",
    "evaluate_matrix",
    "load_report",
    "render_blocker_report",
    "row_key",
    "write_report",
]

SCHEMA_VERSION = "full-swing-qualification/1.0.0"
TARGET_ENGINES: frozenset[str] = frozenset(ALL_ENGINES)
_CLUBS: tuple[str, ...] = ("driver", "iron")
_GATES: tuple[str, ...] = tuple(h.value for h in Horizon)

# Per-engine owner issues for flagship G1/G2/G3 work (folded MS-109/110/112 → #10378).
_OWNER_BLOCKERS: dict[str, dict[str, tuple[int, str]]] = {
    "mujoco": {
        "G1": (10336, "MS-21: MuJoCo G1 native dynamic match and replay"),
        "G2": (10336, "MS-21: MuJoCo G2 through-impact ladder"),
        "G3": (10336, "MS-21: MuJoCo G3 full-capture ladder"),
    },
    "drake": {
        "G1": (10337, "MS-30: Drake native G1 fit"),
        "G2": (10378, "MS-104 folded MS-110: Drake driver/iron G2 native match"),
        "G3": (10378, "MS-104 folded MS-110: Drake driver/iron G3 native match"),
    },
    "pinocchio": {
        "G1": (10381, "MS-107: Pinocchio/Crocoddyl replayable G1"),
        "G2": (10385, "MS-111: Pinocchio driver/iron G2 continuation"),
        "G3": (10385, "MS-111: Pinocchio driver/iron G3 continuation"),
    },
    "opensim": {
        "G1": (10341, "MS-42: OpenSim Moco G1 on shared document"),
        "G2": (10341, "MS-42 phase B: OpenSim G2"),
        "G3": (10341, "MS-42 phase B: OpenSim G3"),
    },
    "myosuite": {
        "G1": (10346, "MS-53: MyoSuite excitation-driven G1"),
        "G2": (10378, "MS-104 folded MS-112: MyoSuite muscle-driven G2"),
        "G3": (10378, "MS-104 folded MS-112: MyoSuite muscle-driven G3"),
    },
    "simscape": {
        "G1": (10378, "MS-104 folded MS-109: Simscape full-body flagship G1"),
        "G2": (10378, "MS-104 folded MS-109: Simscape full-body flagship G2"),
        "G3": (10378, "MS-104 folded MS-109: Simscape full-body flagship G3"),
    },
}


class FullSwingQualificationError(ValueError):
    """Raised when a qualification contract is violated."""


@dataclass(frozen=True)
class NamedBlocker:
    """Named GitHub issue blocking a qualification row."""

    issue: int
    title: str
    kind: str = "native_engine"

    def __post_init__(self) -> None:
        require(self.issue > 0, "blocker issue must be positive", self.issue)
        require(bool(self.title.strip()), "blocker title must be non-empty", self.title)
        require(bool(self.kind.strip()), "blocker kind must be non-empty", self.kind)

    def as_dict(self) -> dict[str, Any]:
        return {"issue": self.issue, "title": self.title, "kind": self.kind}


@dataclass(frozen=True)
class QualificationEvidenceLinks:
    """Content-addressed links required for a qualified row (MS-100/72/replay)."""

    ms100_acceptance_path: str | None = None
    ms100_acceptance_sha256: str | None = None
    ms72_conformance_version: str | None = None
    native_replay_receipt_path: str | None = None
    native_replay_sha256: str | None = None
    numerical_convergence_receipt_path: str | None = None
    numerical_convergence_sha256: str | None = None
    ms70_parity_receipt_path: str | None = None
    candidate_sha: str | None = None
    model_sha: str | None = None
    runtime_hash: str | None = None
    gate_hash: str | None = None

    def is_complete(self) -> bool:
        required = (
            self.ms100_acceptance_path,
            self.ms100_acceptance_sha256,
            self.ms72_conformance_version,
            self.native_replay_receipt_path,
            self.native_replay_sha256,
            self.numerical_convergence_receipt_path,
            self.numerical_convergence_sha256,
            self.candidate_sha,
            self.model_sha,
            self.runtime_hash,
            self.gate_hash,
        )
        return all(v is not None and str(v).strip() for v in required)

    def missing_fields(self) -> tuple[str, ...]:
        names = (
            "ms100_acceptance_path",
            "ms100_acceptance_sha256",
            "ms72_conformance_version",
            "native_replay_receipt_path",
            "native_replay_sha256",
            "numerical_convergence_receipt_path",
            "numerical_convergence_sha256",
            "candidate_sha",
            "model_sha",
            "runtime_hash",
            "gate_hash",
        )
        return tuple(n for n in names if not getattr(self, n))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ms100_acceptance_path": self.ms100_acceptance_path,
            "ms100_acceptance_sha256": self.ms100_acceptance_sha256,
            "ms72_conformance_version": self.ms72_conformance_version,
            "native_replay_receipt_path": self.native_replay_receipt_path,
            "native_replay_sha256": self.native_replay_sha256,
            "numerical_convergence_receipt_path": self.numerical_convergence_receipt_path,
            "numerical_convergence_sha256": self.numerical_convergence_sha256,
            "ms70_parity_receipt_path": self.ms70_parity_receipt_path,
            "candidate_sha": self.candidate_sha,
            "model_sha": self.model_sha,
            "runtime_hash": self.runtime_hash,
            "gate_hash": self.gate_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> QualificationEvidenceLinks:
        if not data:
            return cls()
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class QualificationRowSpec:
    """Static required-row definition before ledger evaluation."""

    engine: str
    club: str
    gate: str
    model_class: str
    platform: str
    owner_blocker: NamedBlocker
    is_required: bool = True


@dataclass(frozen=True)
class QualificationRow:
    """Evaluated engine × club × gate qualification cell."""

    engine: str
    club: str
    gate: str
    model_class: str
    platform: str
    status: str
    evidence: QualificationEvidenceLinks
    blockers: tuple[NamedBlocker, ...]
    is_required: bool
    ledger_receipt_path: str | None = None

    @property
    def is_qualified(self) -> bool:
        return self.status == "qualified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "club": self.club,
            "gate": self.gate,
            "model_class": self.model_class,
            "platform": self.platform,
            "status": self.status,
            "evidence": self.evidence.as_dict(),
            "blockers": [b.as_dict() for b in self.blockers],
            "is_required": self.is_required,
            "ledger_receipt_path": self.ledger_receipt_path,
            "is_qualified": self.is_qualified,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualificationRow:
        blockers = tuple(
            NamedBlocker(**b)
            for b in data.get("blockers", ())
            if isinstance(b, Mapping)
        )
        return cls(
            engine=str(data["engine"]),
            club=str(data["club"]),
            gate=str(data["gate"]),
            model_class=str(data["model_class"]),
            platform=str(data["platform"]),
            status=str(data["status"]),
            evidence=QualificationEvidenceLinks.from_dict(data.get("evidence")),
            blockers=blockers,
            is_required=bool(data.get("is_required", True)),
            ledger_receipt_path=data.get("ledger_receipt_path"),
        )


@dataclass(frozen=True)
class FullSwingQualificationReport:
    """Fail-closed report over required and partial qualification rows."""

    schema_version: str
    generated_at: str
    release_status: str
    required_rows: tuple[QualificationRow, ...]
    partial_rows: tuple[QualificationRow, ...]
    incomplete_required_count: int
    blockers: tuple[NamedBlocker, ...]
    qualification_note: str = ""

    def require_release_ready(self) -> None:
        if self.release_status != "ready" or self.incomplete_required_count:
            raise FullSwingQualificationError(
                f"release not ready: status={self.release_status}, "
                f"incomplete_required={self.incomplete_required_count}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "release_status": self.release_status,
            "required_rows": [r.as_dict() for r in self.required_rows],
            "partial_rows": [r.as_dict() for r in self.partial_rows],
            "incomplete_required_count": self.incomplete_required_count,
            "blockers": [b.as_dict() for b in self.blockers],
            "qualification_note": self.qualification_note,
        }


def row_key(engine: str, club: str, gate: str) -> str:
    """Stable key for an engine × club × gate cell."""
    return f"{engine}:{club}:{gate}"


def build_required_row_specs() -> tuple[QualificationRowSpec, ...]:
    """Return the 36 required full-body flagship rows (6×2×3)."""
    specs: list[QualificationRowSpec] = []
    for engine in sorted(TARGET_ENGINES):
        platform = "deskcomputer" if engine == "simscape" else "any"
        for club in _CLUBS:
            for gate in _GATES:
                issue, title = _OWNER_BLOCKERS[engine][gate]
                specs.append(
                    QualificationRowSpec(
                        engine=engine,
                        club=club,
                        gate=gate,
                        model_class="full_body_flagship",
                        platform=platform,
                        owner_blocker=NamedBlocker(issue, title, "native_engine"),
                    )
                )
    ensure(len(specs) == 36, "required flagship matrix must be 36 rows", len(specs))
    return tuple(specs)


def _acceptance_passed(row: LedgerRow, gate: str) -> bool:
    acc = row.acceptance
    if not isinstance(acc, Mapping):
        return False
    if acc.get("is_physically_accepted") is not True:
        return False
    if str(acc.get("horizon", "")).upper() != gate:
        return False
    gates = acc.get("gates")
    return isinstance(gates, list) and bool(gates)


def _find_ledger_match(
    ledger: Ledger, engine: str, club: str, gate: str
) -> LedgerRow | None:
    for row in ledger.rows:
        if row.engine.lower() != engine:
            continue
        if (row.capture or "").lower() != club:
            continue
        if _acceptance_passed(row, gate):
            return row
    # Prefer a non-accepted row for linking MS-100 rejection evidence.
    for row in ledger.rows:
        if row.engine.lower() == engine and (row.capture or "").lower() == club:
            return row
    return None


def _evidence_blockers(links: QualificationEvidenceLinks) -> tuple[NamedBlocker, ...]:
    missing = links.missing_fields()
    if not missing:
        return ()
    kinds: dict[str, tuple[int, str, str]] = {
        "ms100_acceptance_path": (
            10374,
            "acceptance",
            "MS-100 acceptance receipt path missing",
        ),
        "ms100_acceptance_sha256": (
            10374,
            "acceptance",
            "MS-100 acceptance receipt hash missing",
        ),
        "ms72_conformance_version": (
            10352,
            "conformance",
            "MS-72 conformance version missing",
        ),
        "native_replay_receipt_path": (
            10378,
            "replay",
            "Native replay receipt path missing",
        ),
        "native_replay_sha256": (10378, "replay", "Native replay receipt hash missing"),
        "numerical_convergence_receipt_path": (
            10378,
            "convergence",
            "Numerical-convergence receipt path missing",
        ),
        "numerical_convergence_sha256": (
            10378,
            "convergence",
            "Numerical-convergence receipt hash missing",
        ),
        "candidate_sha": (10378, "provenance", "Candidate hash missing"),
        "model_sha": (10378, "provenance", "Model hash missing"),
        "runtime_hash": (10378, "provenance", "Runtime hash missing"),
        "gate_hash": (10378, "provenance", "Gate hash missing"),
    }
    out: list[NamedBlocker] = []
    for field_name in missing:
        issue, kind, title = kinds[field_name]
        out.append(NamedBlocker(issue, title, kind))
    return tuple(out)


def _evaluate_one(
    spec: QualificationRowSpec,
    ledger: Ledger,
    overrides: Mapping[str, QualificationEvidenceLinks],
    reduced_keys: frozenset[str],
) -> QualificationRow:
    key = row_key(spec.engine, spec.club, spec.gate)
    match = _find_ledger_match(ledger, spec.engine, spec.club, spec.gate)
    links = overrides.get(key, QualificationEvidenceLinks())
    if match is not None and not links.ms100_acceptance_path:
        links = QualificationEvidenceLinks(
            ms100_acceptance_path=match.receipt_path,
            ms100_acceptance_sha256=match.sha256,
            ms72_conformance_version=links.ms72_conformance_version,
            native_replay_receipt_path=links.native_replay_receipt_path,
            native_replay_sha256=links.native_replay_sha256,
            numerical_convergence_receipt_path=links.numerical_convergence_receipt_path,
            numerical_convergence_sha256=links.numerical_convergence_sha256,
            ms70_parity_receipt_path=links.ms70_parity_receipt_path,
            candidate_sha=links.candidate_sha or match.candidate_sha,
            model_sha=links.model_sha,
            runtime_hash=links.runtime_hash,
            gate_hash=links.gate_hash,
        )

    blockers: list[NamedBlocker] = [spec.owner_blocker]
    accepted = match is not None and _acceptance_passed(match, spec.gate)
    if key in reduced_keys:
        blockers.append(
            NamedBlocker(
                10378,
                "Reduced-model oracle cannot satisfy full-body flagship row",
                "model_class",
            )
        )
        accepted = False
    if not accepted:
        blockers.append(
            NamedBlocker(
                10374,
                f"No MS-100 physically accepted {spec.gate} receipt for "
                f"{spec.engine}/{spec.club}",
                "acceptance",
            )
        )
    blockers.extend(_evidence_blockers(links))

    if accepted and links.is_complete() and key not in reduced_keys:
        status = "qualified"
        blockers = ()
    elif accepted:
        status = "incomplete"
    else:
        status = "blocked"

    return QualificationRow(
        engine=spec.engine,
        club=spec.club,
        gate=spec.gate,
        model_class=spec.model_class,
        platform=spec.platform,
        status=status,
        evidence=links,
        blockers=tuple(blockers),
        is_required=spec.is_required,
        ledger_receipt_path=match.receipt_path if match else None,
    )


def _partial_oracle_row() -> QualificationRow:
    return QualificationRow(
        engine="simscape",
        club="driver",
        gate="G1",
        model_class="reduced_oracle",
        platform="deskcomputer",
        status="blocked",
        evidence=QualificationEvidenceLinks(
            ms72_conformance_version=CONFORMANCE_VERSION
        ),
        blockers=(
            NamedBlocker(
                10378,
                "27-coordinate Simscape remains a reduced-model oracle, not flagship G3",
                "model_class",
            ),
        ),
        is_required=False,
        ledger_receipt_path=None,
    )


@dataclass(frozen=True)
class _EvalOptions:
    evidence_overrides: Mapping[str, QualificationEvidenceLinks] = field(
        default_factory=dict
    )
    reduced_oracle_keys: frozenset[str] = field(default_factory=frozenset)
    generated_at: str | None = None


@precondition(
    lambda ledger, evidence_overrides=None, reduced_oracle_keys=None, options=None: (
        isinstance(ledger, Ledger)
    ),
    "ledger must be a Ledger",
)
@postcondition(
    lambda result: (
        result.schema_version == SCHEMA_VERSION
        and result.release_status in {"blocked", "ready"}
        and len(result.required_rows) == 36
    ),
    "report must be schema-valid with 36 required rows",
)
def evaluate_matrix(
    ledger: Ledger,
    *,
    evidence_overrides: Mapping[str, QualificationEvidenceLinks] | None = None,
    reduced_oracle_keys: set[str] | frozenset[str] | None = None,
    options: _EvalOptions | None = None,
) -> FullSwingQualificationReport:
    """Evaluate the required flagship matrix against ledger + evidence links."""
    opts = options or _EvalOptions(
        evidence_overrides=evidence_overrides or {},
        reduced_oracle_keys=frozenset(reduced_oracle_keys or ()),
    )
    overrides = opts.evidence_overrides
    reduced = opts.reduced_oracle_keys
    required = tuple(
        _evaluate_one(spec, ledger, overrides, reduced)
        for spec in build_required_row_specs()
    )
    incomplete = sum(1 for r in required if not r.is_qualified)
    release = "ready" if incomplete == 0 else "blocked"
    blocker_map: dict[tuple[int, str], NamedBlocker] = {}
    for row in required:
        for blocker in row.blockers:
            blocker_map[(blocker.issue, blocker.title)] = blocker
    note = (
        "Software-contract matrix only; no invented six-engine native pass. "
        "Release stays blocked until every required row links MS-100 acceptance, "
        "MS-72 conformance, native replay, and numerical-convergence receipts."
    )
    return FullSwingQualificationReport(
        schema_version=SCHEMA_VERSION,
        generated_at=opts.generated_at
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        release_status=release,
        required_rows=required,
        partial_rows=(_partial_oracle_row(),),
        incomplete_required_count=incomplete,
        blockers=tuple(blocker_map.values()),
        qualification_note=note,
    )


def render_blocker_report(report: FullSwingQualificationReport) -> str:
    """Human-readable enumeration of remaining blockers (never claims ready early)."""
    lines = [
        f"MS-104 full-swing qualification ({report.schema_version})",
        f"release_status={report.release_status}",
        f"incomplete_required={report.incomplete_required_count}/36",
        "",
        "Remaining blockers:",
    ]
    for blocker in report.blockers:
        lines.append(f"- #{blocker.issue} [{blocker.kind}] {blocker.title}")
    lines.append("")
    lines.append("Per-engine incomplete cells:")
    for engine in sorted(TARGET_ENGINES):
        cells = [
            f"{r.club}/{r.gate}={r.status}"
            for r in report.required_rows
            if r.engine == engine and not r.is_qualified
        ]
        lines.append(f"- {engine}: " + (", ".join(cells) if cells else "all qualified"))
    return "\n".join(lines)


def write_report(report: FullSwingQualificationReport, path: Path) -> None:
    """Persist a JSON report (content-addressed snapshot for evidence)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_report(path: Path) -> FullSwingQualificationReport:
    """Load a previously written qualification report."""
    require(path.is_file(), f"report not found: {path}", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, Mapping), "report root must be an object", data)
    required = tuple(QualificationRow.from_dict(r) for r in data["required_rows"])
    partial = tuple(QualificationRow.from_dict(r) for r in data.get("partial_rows", ()))
    blockers = tuple(
        NamedBlocker(**b) for b in data.get("blockers", ()) if isinstance(b, Mapping)
    )
    return FullSwingQualificationReport(
        schema_version=str(data["schema_version"]),
        generated_at=str(data["generated_at"]),
        release_status=str(data["release_status"]),
        required_rows=required,
        partial_rows=partial,
        incomplete_required_count=int(data["incomplete_required_count"]),
        blockers=blockers,
        qualification_note=str(data.get("qualification_note", "")),
    )
