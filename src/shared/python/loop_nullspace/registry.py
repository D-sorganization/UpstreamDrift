"""NSX-11: Hypothesis, Convention, and Preset Registry Loader With CI Citation Gate.

Provides immutable typed records and Design-by-Contract validation for
``src/config/loop_nullspace_registry.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

VerificationStatus = Literal["VERIFIED", "UNVERIFIED", "REMOVED"]
VALID_VERIFICATION_STATUSES: frozenset[str] = frozenset(
    {"VERIFIED", "UNVERIFIED", "REMOVED"}
)

UNVERIFIED_CITATION_TOKENS: tuple[str, ...] = (
    "Thornton",
    "Mehta",
    "Choi",
    "Sorokin",
    "Bosch",
    "Nesbit",
    "two-thirds",
    "15-40 ms",
    "MDPI",
    "SPAR",
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "src" / "config" / "loop_nullspace_registry.json"


def _validate_verification_status(value: Any, context: str) -> VerificationStatus:
    """Validate that ``value`` is an allowed verification status string."""
    if not isinstance(value, str) or value not in VALID_VERIFICATION_STATUSES:
        raise ValueError(
            f"{context}: invalid verification_status {value!r}; "
            f"must be one of {sorted(VALID_VERIFICATION_STATUSES)}"
        )
    return value  # type: ignore[return-value]


def _enforce_no_unverified_tokens(
    text: str,
    context: str,
    extra_unverified_tokens: Sequence[str] = (),
) -> None:
    """Raise ``ValueError`` if ``text`` cites any unverified or removed source token."""
    lowered = text.lower()
    for token in (*UNVERIFIED_CITATION_TOKENS, *extra_unverified_tokens):
        if token and token.lower() in lowered:
            raise ValueError(
                f"{context} cites unverified/removed source token {token!r}: {text!r}"
            )


@dataclass(frozen=True)
class SourceRecord:
    """Verified or explicitly flagged literature source record."""

    citation: str
    doi_or_url: str | None
    table_or_figure: str | None
    verification_status: VerificationStatus
    source_id: str | None = None

    @classmethod
    def from_raw(
        cls,
        raw: str | Mapping[str, Any],
        citations_by_id: Mapping[str, Mapping[str, Any]],
        context: str,
    ) -> SourceRecord:
        """Parse a source record from either an inline mapping or a citation ID."""
        if isinstance(raw, str):
            cit = citations_by_id.get(raw)
            if cit is None:
                raise ValueError(f"{context}: unknown source citation id {raw!r}")
            status = _validate_verification_status(
                cit.get("verification_status"), f"{context}.source({raw})"
            )
            return cls(
                citation=str(cit.get("reference", raw)),
                doi_or_url=cit.get("doi_or_url"),
                table_or_figure=cit.get("table_or_figure"),
                verification_status=status,
                source_id=raw,
            )
        if not isinstance(raw, Mapping):
            raise TypeError(f"{context}: source must be a dict or citation id string")

        citation_str = raw.get("citation") or raw.get("reference")
        if not isinstance(citation_str, str) or not citation_str.strip():
            raise ValueError(f"{context}: source must provide non-empty 'citation'")

        status = _validate_verification_status(
            raw.get("verification_status"), f"{context}.source"
        )
        src_id = raw.get("id")
        if isinstance(src_id, str) and src_id in citations_by_id:
            cat_status = citations_by_id[src_id].get("verification_status")
            if cat_status != status:
                raise ValueError(
                    f"{context}: source status {status!r} mismatches catalog {cat_status!r}"
                )
        return cls(
            citation=citation_str.strip(),
            doi_or_url=raw.get("doi_or_url"),
            table_or_figure=raw.get("table_or_figure"),
            verification_status=status,
            source_id=src_id if isinstance(src_id, str) else None,
        )


@dataclass(frozen=True)
class HypothesisRecord:
    """A physiological or modeling prior hypothesis entry."""

    id: str
    statement: str
    parameters: Mapping[str, Any]
    source: SourceRecord
    verification_status: VerificationStatus

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        citations_by_id: Mapping[str, Mapping[str, Any]],
        unverified_ids: Sequence[str],
    ) -> HypothesisRecord:
        """Construct and validate a ``HypothesisRecord`` from a JSON dict."""
        if not isinstance(data, Mapping):
            raise TypeError("Hypothesis entry must be a JSON object")
        hid = data.get("id")
        if not isinstance(hid, str) or not hid.strip():
            raise ValueError("Hypothesis entry missing non-empty 'id'")
        statement = data.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"Hypothesis {hid!r} missing non-empty 'statement'")

        params = data.get("parameters", {})
        if not isinstance(params, Mapping):
            raise TypeError(f"Hypothesis {hid!r} 'parameters' must be a mapping")

        if "source" not in data:
            raise ValueError(f"Hypothesis {hid!r} missing 'source'")
        source = SourceRecord.from_raw(
            data["source"], citations_by_id, f"Hypothesis {hid!r}"
        )
        status = _validate_verification_status(
            data.get("verification_status"), f"Hypothesis {hid!r}"
        )
        if status != source.verification_status:
            raise ValueError(
                f"Hypothesis {hid!r} verification_status {status!r} does not match "
                f"source verification_status {source.verification_status!r}"
            )
        _enforce_no_unverified_tokens(
            statement, f"Hypothesis {hid!r} statement", unverified_ids
        )
        return cls(
            id=hid.strip(),
            statement=statement.strip(),
            parameters=MappingProxyType(dict(params)),
            source=source,
            verification_status=status,
        )


@dataclass(frozen=True)
class ConventionRecord:
    """A mathematical frame, metric, or cone convention entry."""

    id: str
    statement: str
    parameters: Mapping[str, Any]

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], unverified_ids: Sequence[str]
    ) -> ConventionRecord:
        """Construct and validate a ``ConventionRecord`` from a JSON dict."""
        if not isinstance(data, Mapping):
            raise TypeError("Convention entry must be a JSON object")
        cid = data.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("Convention entry missing non-empty 'id'")
        statement = data.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError(f"Convention {cid!r} missing non-empty 'statement'")
        params = data.get("parameters", {})
        if not isinstance(params, Mapping):
            raise TypeError(f"Convention {cid!r} 'parameters' must be a mapping")
        _enforce_no_unverified_tokens(
            statement, f"Convention {cid!r} statement", unverified_ids
        )
        return cls(
            id=cid.strip(),
            statement=statement.strip(),
            parameters=MappingProxyType(dict(params)),
        )


@dataclass(frozen=True)
class PresetRecord:
    """Named bundle of hypothesis IDs and convention IDs."""

    name: str
    label: str
    description: str
    hypothesis_ids: tuple[str, ...]
    convention_ids: tuple[str, ...]

    @property
    def hypotheses(self) -> tuple[str, ...]:
        """Compatibility alias returning ``hypothesis_ids``."""
        return self.hypothesis_ids

    @property
    def conventions(self) -> tuple[str, ...]:
        """Compatibility alias returning ``convention_ids``."""
        return self.convention_ids

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], unverified_ids: Sequence[str]
    ) -> PresetRecord:
        """Construct and validate a ``PresetRecord`` from a JSON dict."""
        if not isinstance(data, Mapping):
            raise TypeError("Preset entry must be a JSON object")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Preset entry missing non-empty 'name'")
        label = data.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Preset {name!r} missing non-empty 'label'")
        description = data.get("description", "")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Preset {name!r} missing non-empty 'description'")

        raw_hyps = data.get("hypothesis_ids") or data.get("hypotheses")
        if not isinstance(raw_hyps, list) or not raw_hyps:
            raise ValueError(f"Preset {name!r} must specify non-empty hypothesis_ids")
        raw_convs = data.get("convention_ids") or data.get("conventions")
        if not isinstance(raw_convs, list) or not raw_convs:
            raise ValueError(f"Preset {name!r} must specify non-empty convention_ids")

        _enforce_no_unverified_tokens(label, f"Preset {name!r} label", unverified_ids)
        _enforce_no_unverified_tokens(
            description, f"Preset {name!r} description", unverified_ids
        )
        return cls(
            name=name.strip(),
            label=label.strip(),
            description=description.strip(),
            hypothesis_ids=tuple(str(h) for h in raw_hyps),
            convention_ids=tuple(str(c) for c in raw_convs),
        )


@dataclass(frozen=True)
class HypothesisSet:
    """Resolved preset containing full ``HypothesisRecord`` and ``ConventionRecord`` objects."""

    preset: PresetRecord
    hypotheses: tuple[HypothesisRecord, ...]
    conventions: tuple[ConventionRecord, ...]

    @property
    def name(self) -> str:
        """Preset identifier."""
        return self.preset.name

    @property
    def label(self) -> str:
        """User-facing preset label."""
        return self.preset.label

    @property
    def hypothesis_ids(self) -> tuple[str, ...]:
        """Tuple of resolved hypothesis IDs."""
        return tuple(h.id for h in self.hypotheses)

    @property
    def convention_ids(self) -> tuple[str, ...]:
        """Tuple of resolved convention IDs."""
        return tuple(c.id for c in self.conventions)


@dataclass(frozen=True)
class LoopNullspaceRegistry:
    """Validated, immutable Closed-Loop Null-Space Exploration registry."""

    schema_version: str
    description: str
    hypotheses: tuple[HypothesisRecord, ...] = field(default_factory=tuple)
    conventions: tuple[ConventionRecord, ...] = field(default_factory=tuple)
    presets: tuple[PresetRecord, ...] = field(default_factory=tuple)
    raw_torque_capacity: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisRecord:
        """Look up a hypothesis by ID or raise ``KeyError``."""
        for item in self.hypotheses:
            if item.id == hypothesis_id:
                return item
        raise KeyError(f"Unknown hypothesis id: {hypothesis_id!r}")

    def get_convention(self, convention_id: str) -> ConventionRecord:
        """Look up a convention by ID or raise ``KeyError``."""
        for item in self.conventions:
            if item.id == convention_id:
                return item
        raise KeyError(f"Unknown convention id: {convention_id!r}")

    def get_preset(self, name: str) -> HypothesisSet:
        """Resolve a preset by name into a ``HypothesisSet`` or raise ``KeyError``."""
        for item in self.presets:
            if item.name == name:
                hyps = tuple(self.get_hypothesis(hid) for hid in item.hypothesis_ids)
                convs = tuple(self.get_convention(cid) for cid in item.convention_ids)
                return HypothesisSet(preset=item, hypotheses=hyps, conventions=convs)
        raise KeyError(f"Unknown preset name: {name!r}")


RegistryCatalog = LoopNullspaceRegistry


def _validate_torque_capacity_entries(
    entries: Sequence[Any],
    citations_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate ``torque_capacity[]`` source statuses and ASSUMED slot warning flags."""
    seen_coords: set[str] = set()
    validated: list[Mapping[str, Any]] = []
    for idx, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            raise TypeError(f"torque_capacity[{idx}] must be an object")
        coord = raw.get("coordinate")
        if not isinstance(coord, str) or not coord.strip():
            raise ValueError(f"torque_capacity[{idx}] missing 'coordinate'")
        if coord in seen_coords:
            raise ValueError(f"Duplicate torque_capacity coordinate: {coord!r}")
        seen_coords.add(coord)

        if "source" not in raw:
            raise ValueError(f"torque_capacity[{coord!r}] missing 'source'")
        src = SourceRecord.from_raw(
            raw["source"], citations_by_id, f"torque_capacity[{coord!r}]"
        )
        ver_status = _validate_verification_status(
            raw.get("verification_status"), f"torque_capacity[{coord!r}]"
        )
        if ver_status != src.verification_status:
            raise ValueError(
                f"torque_capacity[{coord!r}] verification_status {ver_status!r} "
                f"mismatches source {src.verification_status!r}"
            )
        slot_status = raw.get("status", ver_status)
        if slot_status == "ASSUMED" or ver_status != "VERIFIED":
            if ver_status != "UNVERIFIED":
                raise ValueError(
                    f"ASSUMED torque_capacity[{coord!r}] must have "
                    f"verification_status='UNVERIFIED', got {ver_status!r}"
                )
            warning = raw.get("warning")
            if not isinstance(warning, str) or not warning.strip():
                raise ValueError(
                    f"ASSUMED/UNVERIFIED torque_capacity[{coord!r}] requires "
                    "a non-empty 'warning' string"
                )
        validated.append(MappingProxyType(dict(raw)))
    return tuple(validated)


def _ensure_unique_ids(ids: Sequence[str], label: str) -> set[str]:
    """Verify that ``ids`` contains no duplicates and return the set of IDs."""
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            raise ValueError(f"Duplicate {label} in registry: {item_id!r}")
        seen.add(item_id)
    return seen


def _validate_preset_references(
    presets: Sequence[PresetRecord],
    hyp_ids: set[str],
    conv_ids: set[str],
    hypotheses: Sequence[HypothesisRecord],
) -> None:
    """Verify every preset references existing verified hypotheses and conventions."""
    hyp_by_id = {h.id: h for h in hypotheses}
    for p in presets:
        for hid in p.hypothesis_ids:
            if hid not in hyp_ids:
                raise ValueError(
                    f"Preset {p.name!r} references unknown hypothesis id {hid!r}"
                )
            if hyp_by_id[hid].verification_status != "VERIFIED":
                raise ValueError(
                    f"Preset {p.name!r} references non-VERIFIED hypothesis {hid!r}"
                )
        for cid in p.convention_ids:
            if cid not in conv_ids:
                raise ValueError(
                    f"Preset {p.name!r} references unknown convention id {cid!r}"
                )


def load_registry(path: Path | str | None = None) -> LoopNullspaceRegistry:
    """Load and validate the Closed-Loop Null-Space Exploration registry."""
    reg_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not reg_path.is_file():
        raise FileNotFoundError(f"Loop nullspace registry not found: {reg_path}")

    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Registry root must be a JSON object")

    schema_version = raw.get("schema_version")
    if schema_version != "1.0.0":
        raise ValueError(f"Unsupported registry schema_version: {schema_version!r}")

    citations_raw = raw.get("citations", [])
    citations_by_id: dict[str, Mapping[str, Any]] = {}
    unverified_ids: list[str] = []
    for cit in citations_raw:
        if isinstance(cit, Mapping) and isinstance(cit.get("id"), str):
            cid = cit["id"]
            if cid in citations_by_id:
                raise ValueError(f"Duplicate citation id: {cid!r}")
            c_status = _validate_verification_status(
                cit.get("verification_status"), f"Citation {cid!r}"
            )
            citations_by_id[cid] = cit
            if c_status != "VERIFIED":
                unverified_ids.append(cid)

    hyps_raw = raw.get("hypotheses")
    convs_raw = raw.get("conventions")
    presets_raw = raw.get("presets")
    if not isinstance(hyps_raw, list) or not hyps_raw:
        raise ValueError("Registry must contain non-empty 'hypotheses' list")
    if not isinstance(convs_raw, list) or not convs_raw:
        raise ValueError("Registry must contain non-empty 'conventions' list")
    if not isinstance(presets_raw, list) or not presets_raw:
        raise ValueError("Registry must contain non-empty 'presets' list")

    hypotheses = tuple(
        HypothesisRecord.from_dict(h, citations_by_id, unverified_ids) for h in hyps_raw
    )
    conventions = tuple(
        ConventionRecord.from_dict(c, unverified_ids) for c in convs_raw
    )
    presets = tuple(PresetRecord.from_dict(p, unverified_ids) for p in presets_raw)

    hyp_ids = _ensure_unique_ids([h.id for h in hypotheses], "hypothesis id")
    conv_ids = _ensure_unique_ids([c.id for c in conventions], "convention id")
    _ensure_unique_ids([p.name for p in presets], "preset name")
    _validate_preset_references(presets, hyp_ids, conv_ids, hypotheses)

    cap_raw = raw.get("torque_capacity", [])
    if not isinstance(cap_raw, list):
        raise ValueError("'torque_capacity' must be a list")
    validated_caps = _validate_torque_capacity_entries(cap_raw, citations_by_id)

    return LoopNullspaceRegistry(
        schema_version=schema_version,
        description=str(raw.get("description", "")),
        hypotheses=hypotheses,
        conventions=conventions,
        presets=presets,
        raw_torque_capacity=validated_caps,
    )


def hypothesis(
    hypothesis_id: str, *, registry: LoopNullspaceRegistry | None = None
) -> HypothesisRecord:
    """Return the ``HypothesisRecord`` for ``hypothesis_id``."""
    if not isinstance(hypothesis_id, str) or not hypothesis_id.strip():
        raise ValueError("hypothesis_id must be a non-empty string")
    reg = registry if registry is not None else load_registry()
    try:
        return reg.get_hypothesis(hypothesis_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def convention(
    convention_id: str, *, registry: LoopNullspaceRegistry | None = None
) -> ConventionRecord:
    """Return the ``ConventionRecord`` for ``convention_id``."""
    if not isinstance(convention_id, str) or not convention_id.strip():
        raise ValueError("convention_id must be a non-empty string")
    reg = registry if registry is not None else load_registry()
    try:
        return reg.get_convention(convention_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def preset(
    name: str, *, registry: LoopNullspaceRegistry | None = None
) -> HypothesisSet:
    """Resolve preset ``name`` into a ``HypothesisSet``."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("preset name must be a non-empty string")
    reg = registry if registry is not None else load_registry()
    try:
        return reg.get_preset(name)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def require_verified_for_display(
    hypothesis_ids: Iterable[str],
    *,
    registry: LoopNullspaceRegistry | None = None,
) -> tuple[HypothesisRecord, ...]:
    """Verify every referenced hypothesis is ``VERIFIED`` and safe for display."""
    reg = registry if registry is not None else load_registry()
    ids_tuple = tuple(hypothesis_ids)
    if not ids_tuple:
        raise ValueError("hypothesis_ids must not be empty")
    records: list[HypothesisRecord] = []
    for hid in ids_tuple:
        rec = hypothesis(hid, registry=reg)
        if rec.verification_status != "VERIFIED":
            raise ValueError(
                f"Hypothesis {hid!r} has verification_status={rec.verification_status!r} "
                "and cannot be displayed as a verified prior"
            )
        if rec.source.verification_status != "VERIFIED":
            raise ValueError(
                f"Hypothesis {hid!r} source has verification_status="
                f"{rec.source.verification_status!r}"
            )
        _enforce_no_unverified_tokens(rec.statement, f"Hypothesis {hid!r}")
        records.append(rec)
    return tuple(records)


__all__ = [
    "ConventionRecord",
    "DEFAULT_REGISTRY_PATH",
    "HypothesisRecord",
    "HypothesisSet",
    "LoopNullspaceRegistry",
    "PresetRecord",
    "RegistryCatalog",
    "SourceRecord",
    "VerificationStatus",
    "convention",
    "hypothesis",
    "load_registry",
    "preset",
    "require_verified_for_display",
]
