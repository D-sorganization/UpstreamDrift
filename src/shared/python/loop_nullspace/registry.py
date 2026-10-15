"""Typed loader and citation gate for the NSX registry (NSX-11, R1/R13).

Every prior behind a closed-loop allocation is a named, versioned object in
``src/config/loop_nullspace_registry.json``: hypotheses (physiological or
modelling priors), conventions (metric, scale or frame choices), presets
(named hypothesis sets) and torque capacities.  This module parses that file
into frozen records and enforces the citation gate:

* ids are unique across hypotheses, conventions and presets;
* every preset references existing hypothesis and convention ids;
* every citation and every capacity source carries a ``verification_status``;
* no user-facing string (``statement``, preset ``label``) mentions a citation
  that is not VERIFIED, and no preset contains a hypothesis resting on one;
* a capacity without a VERIFIED source must be flagged ``assumed``.

Design by Contract:
    Preconditions: input is a mapping of the documented schema.
    Postconditions: a returned :class:`Registry` is internally consistent.
    Invariants: all records are immutable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REGISTRY_PATH = (
    Path(__file__).resolve().parents[3] / "config" / ("loop_nullspace_registry.json")
)
VALID_STATUSES = frozenset({"VERIFIED", "UNVERIFIED", "REMOVED", "NOT_APPLICABLE"})
_UNSOUND = frozenset({"UNVERIFIED", "REMOVED"})


def _req(data: Mapping[str, Any], key: str, kind: type, where: str) -> Any:
    """Return ``data[key]`` after checking presence and type."""
    if key not in data:
        raise ValueError(f"{where}: missing required key {key!r}")
    value = data[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise ValueError(f"{where}: {key!r} must be {kind.__name__}")
    return value


def _status(data: Mapping[str, Any], where: str) -> str:
    value = _req(data, "verification_status", str, where)
    if value not in VALID_STATUSES:
        raise ValueError(
            f"{where}: verification_status {value!r} not in {sorted(VALID_STATUSES)}"
        )
    return value


@dataclass(frozen=True)
class Citation:
    """A checked literature reference."""

    id: str
    reference: str
    verification_status: str


@dataclass(frozen=True)
class Source:
    """Where a prior comes from; ``citation_id`` is None for modelling choices."""

    citation_id: str | None
    note: str


@dataclass(frozen=True)
class Hypothesis:
    """A physiological or modelling prior conditioning FAMILY_DEPENDENT output."""

    id: str
    statement: str
    source: Source
    verification_status: str


@dataclass(frozen=True)
class Convention:
    """A choice of metric, scale or frame (never physiology)."""

    id: str
    statement: str


@dataclass(frozen=True)
class TorqueCapacity:
    """One joint-direction torque capacity; ``assumed`` flags an unsourced value."""

    joint_slot: str
    direction: str
    tau_max_nm: float
    source: Source
    verification_status: str
    assumed: bool


@dataclass(frozen=True)
class HypothesisSet:
    """A resolved preset: its hypotheses and conventions as records."""

    id: str
    label: str
    hypotheses: tuple[Hypothesis, ...]
    conventions: tuple[Convention, ...]

    def hypothesis_ids(self) -> tuple[str, ...]:
        """Return the ids of the member hypotheses, in preset order."""
        return tuple(h.id for h in self.hypotheses)


@dataclass(frozen=True)
class Preset:
    """A named list of hypothesis and convention ids."""

    id: str
    label: str
    hypothesis_ids: tuple[str, ...]
    convention_ids: tuple[str, ...]


@dataclass(frozen=True)
class Registry:
    """The validated registry."""

    schema_version: int
    registry_version: str
    citations: tuple[Citation, ...]
    hypotheses: tuple[Hypothesis, ...]
    conventions: tuple[Convention, ...]
    presets: tuple[Preset, ...]
    torque_capacity: tuple[TorqueCapacity, ...]

    def hypothesis(self, hypothesis_id: str) -> Hypothesis:
        """Return a hypothesis by id.  Raises ``KeyError`` if unknown."""
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                return h
        raise KeyError(f"unknown hypothesis id {hypothesis_id!r}")

    def preset(self, name: str) -> HypothesisSet:
        """Resolve a preset to records.  Raises ``KeyError`` if unknown."""
        for p in self.presets:
            if p.id == name:
                conv = {c.id: c for c in self.conventions}
                return HypothesisSet(
                    id=p.id,
                    label=p.label,
                    hypotheses=tuple(self.hypothesis(i) for i in p.hypothesis_ids),
                    conventions=tuple(conv[i] for i in p.convention_ids),
                )
        raise KeyError(f"unknown preset {name!r}")

    def citation_status(self, citation_id: str | None) -> str:
        """Return a citation's status; ``NOT_APPLICABLE`` for no citation."""
        if citation_id is None:
            return "NOT_APPLICABLE"
        for c in self.citations:
            if c.id == citation_id:
                return c.verification_status
        raise KeyError(f"unknown citation id {citation_id!r}")

    def assumed_capacities(self) -> tuple[TorqueCapacity, ...]:
        """Return the capacities flagged ASSUMED."""
        return tuple(c for c in self.torque_capacity if c.assumed)

    def require_verified_for_display(self, ids: Iterable[str]) -> tuple[str, ...]:
        """Return ``ids`` unchanged if each may be shown to a user.

        Raises:
            TypeError: If ``ids`` is a bare string or not iterable.
            ValueError: If an id is unknown, is UNVERIFIED or REMOVED, or
                rests on a citation that is not VERIFIED.
        """
        if isinstance(ids, str) or not isinstance(ids, Iterable):
            raise TypeError("ids must be an iterable of hypothesis id strings")
        out = tuple(ids)
        for hid in out:
            try:
                h = self.hypothesis(hid)
            except KeyError:
                raise ValueError(f"unknown hypothesis id {hid!r}") from None
            bad = _unsound_reason(self, h)
            if bad:
                raise ValueError(f"{hid} cannot be displayed: {bad}")
        return out


def _unsound_reason(reg: Registry, h: Hypothesis) -> str:
    """Describe why ``h`` is not displayable, or return an empty string."""
    if h.verification_status in _UNSOUND:
        return f"verification_status is {h.verification_status}"
    cstat = reg.citation_status(h.source.citation_id)
    if h.source.citation_id is not None and cstat != "VERIFIED":
        return f"cited source {h.source.citation_id} is {cstat}"
    return ""


def _parse_source(data: Mapping[str, Any], where: str) -> Source:
    src = _req(data, "source", dict, where)
    if "citation_id" not in src:
        raise ValueError(f"{where}: source missing 'citation_id'")
    cid = src["citation_id"]
    if cid is not None and not isinstance(cid, str):
        raise ValueError(f"{where}: citation_id must be a string or null")
    return Source(cid, _req(src, "note", str, f"{where}.source"))


def _check_unique(kind: str, ids: list[str], seen: set[str]) -> None:
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate id {i!r} ({kind})")
        seen.add(i)


def _parse_capacity(item: Mapping[str, Any], idx: int) -> TorqueCapacity:
    where = f"torque_capacity[{idx}]"
    tau = item.get("tau_max_nm")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not tau > 0:
        raise ValueError(f"{where}: tau_max_nm must be a positive number")
    return TorqueCapacity(
        joint_slot=_req(item, "joint_slot", str, where),
        direction=_req(item, "direction", str, where),
        tau_max_nm=float(tau),
        source=_parse_source(item, where),
        verification_status=_status(item, where),
        assumed=_req(item, "assumed", bool, where),
    )


def _check_capacity_flags(reg: Registry) -> None:
    for i, cap in enumerate(reg.torque_capacity):
        where = f"torque_capacity[{i}] {cap.joint_slot}/{cap.direction}"
        sound = (
            cap.source.citation_id is not None
            and reg.citation_status(cap.source.citation_id) == "VERIFIED"
            and cap.verification_status == "VERIFIED"
        )
        if sound and cap.assumed:
            raise ValueError(f"{where}: verified capacity must not be assumed")
        if not sound:
            if not cap.assumed:
                raise ValueError(f"{where}: unsourced capacity must set assumed=true")
            if cap.verification_status == "VERIFIED":
                raise ValueError(
                    f"{where}: cannot be VERIFIED without a verified source"
                )


def _check_citation_gate(reg: Registry) -> None:
    """Enforce the rules tying user-facing text to verified sources."""
    for h in reg.hypotheses:
        cid = h.source.citation_id
        if cid is not None:
            if all(c.id != cid for c in reg.citations):
                raise ValueError(f"{h.id}: unknown citation id {cid!r}")
        if h.verification_status == "VERIFIED" and (
            cid is None or reg.citation_status(cid) != "VERIFIED"
        ):
            raise ValueError(f"{h.id}: VERIFIED requires a VERIFIED citation")
    bad = {
        c.id: c.verification_status
        for c in reg.citations
        if c.verification_status != "VERIFIED"
    }
    texts = [(h.id, h.statement) for h in reg.hypotheses]
    texts += [(c.id, c.statement) for c in reg.conventions]
    texts += [(p.id, p.label) for p in reg.presets]
    for owner, text in texts:
        for cid, status in bad.items():
            if cid in text:
                raise ValueError(
                    f"{owner}: user-facing text cites {status} source {cid}"
                )
    for p in reg.presets:
        for hid in p.hypothesis_ids:
            reason = _unsound_reason(reg, reg.hypothesis(hid))
            if reason:
                raise ValueError(f"preset {p.id}: {hid} {reason}")


def parse_registry(data: Mapping[str, Any]) -> Registry:
    """Parse and validate a registry mapping.

    Raises:
        TypeError: If ``data`` is not a mapping.
        ValueError: On any schema or citation-gate violation.
    """
    if not isinstance(data, Mapping):
        raise TypeError("registry data must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    citations = tuple(
        Citation(
            _req(c, "id", str, f"citations[{i}]"),
            _req(c, "reference", str, f"citations[{i}]"),
            _status(c, f"citations[{i}]"),
        )
        for i, c in enumerate(_req(data, "citations", list, "registry"))
    )
    hyps = tuple(
        Hypothesis(
            _req(h, "id", str, f"hypotheses[{i}]"),
            _req(h, "statement", str, f"hypotheses[{i}]"),
            _parse_source(h, f"hypotheses[{i}]"),
            _status(h, f"hypotheses[{i}]"),
        )
        for i, h in enumerate(_req(data, "hypotheses", list, "registry"))
    )
    convs = tuple(
        Convention(
            _req(c, "id", str, f"conventions[{i}]"),
            _req(c, "statement", str, f"conventions[{i}]"),
        )
        for i, c in enumerate(_req(data, "conventions", list, "registry"))
    )
    presets = tuple(
        Preset(
            _req(p, "id", str, f"presets[{i}]"),
            _req(p, "label", str, f"presets[{i}]"),
            tuple(_req(p, "hypotheses", list, f"presets[{i}]")),
            tuple(_req(p, "conventions", list, f"presets[{i}]")),
        )
        for i, p in enumerate(_req(data, "presets", list, "registry"))
    )
    caps = tuple(
        _parse_capacity(c, i)
        for i, c in enumerate(_req(data, "torque_capacity", list, "registry"))
    )
    seen: set[str] = set()
    _check_unique("citation", [c.id for c in citations], set())
    _check_unique("hypothesis", [h.id for h in hyps], seen)
    _check_unique("convention", [c.id for c in convs], seen)
    _check_unique("preset", [p.id for p in presets], seen)
    reg = Registry(
        SCHEMA_VERSION,
        _req(data, "registry_version", str, "registry"),
        citations,
        hyps,
        convs,
        presets,
        caps,
    )
    hid, cid = {h.id for h in hyps}, {c.id for c in convs}
    for p in presets:
        for i in p.hypothesis_ids:
            if i not in hid:
                raise ValueError(f"preset {p.id}: unknown hypothesis id {i!r}")
        for i in p.convention_ids:
            if i not in cid:
                raise ValueError(f"preset {p.id}: unknown convention id {i!r}")
    _check_citation_gate(reg)
    _check_capacity_flags(reg)
    return reg


def load_raw_registry(path: Path | None = None) -> dict[str, Any]:
    """Return the registry JSON as a plain dict (fresh object each call)."""
    text = (path or REGISTRY_PATH).read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def _shipped() -> Registry:
    return parse_registry(load_raw_registry())


def load_registry(path: Path | None = None) -> Registry:
    """Load and validate the registry (the shipped file is cached).

    Postcondition: the returned registry passed every gate in this module.
    """
    if path is None:
        return _shipped()
    return parse_registry(load_raw_registry(path))


def hypothesis(hypothesis_id: str) -> Hypothesis:
    """Return a shipped hypothesis by id (``KeyError`` if unknown)."""
    return load_registry().hypothesis(hypothesis_id)


def preset(name: str) -> HypothesisSet:
    """Return a shipped preset as a :class:`HypothesisSet`."""
    return load_registry().preset(name)


def require_verified_for_display(ids: Iterable[str]) -> tuple[str, ...]:
    """Gate shipped hypothesis ids for display; see :meth:`Registry.require_verified_for_display`."""
    return load_registry().require_verified_for_display(ids)
