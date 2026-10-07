"""Headless drag-and-drop assembly session for the Frankenstein editor (CMB-8/9).

``AssemblySession`` owns a working :class:`URDFModel` plus the typed ports of
every placed part. It decides whether a drop is allowed (port compatibility,
occupancy, live composition validation), performs attach/detach, keeps an
undo/redo history, and round-trips through URDF. It has no Qt dependency so the
canvas widget stays thin and every rule is unit-testable.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET  # stdlib retained for Element  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from collections.abc import Callable
from dataclasses import dataclass

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks

from src.shared.python.model_generation.editor.attachment_ports import PortPolarity
from src.tools.model_explorer.attachment_manifest import (
    AttachmentPoint,
    parse_attachment_point,
)
from src.tools.model_explorer.composition_flow import AttachmentSelection
from src.tools.model_explorer.composition_ux import (
    CompositionDragPayload,
    CompositionUxController,
)
from src.tools.model_explorer.composition_validator import (
    CompositionFinding,
    CompositionValidationResult,
)
from src.tools.model_explorer.frankenstein_editor.model import URDFModel
from src.tools.model_explorer.part_catalog import (
    PartCatalog,
    PartSpec,
    find_mating_plug,
)

ASSEMBLY_TAG = "ud_assembly"
ASSEMBLY_SCHEMA = 1


class AssemblyError(ValueError):
    """Raised when an assembly operation is rejected."""


@dataclass(frozen=True)
class PlacedPart:
    """One part instance inside the assembly."""

    instance_id: str
    part_id: str
    prefix: str
    host_port: str | None
    host_instance: str | None
    links: tuple[str, ...]
    joints: tuple[str, ...]
    ports: tuple[AttachmentPoint, ...]


@dataclass(frozen=True)
class DropDecision:
    """Result of asking whether a part may be dropped on a host port."""

    accepted: bool
    reason: str
    part_id: str
    host_port: str
    findings: tuple[CompositionFinding, ...] = ()


@dataclass(frozen=True)
class _Snapshot:
    model: URDFModel
    placed: tuple[PlacedPart, ...]
    counter: int


class AssemblySession:
    """Mutable assembly with typed-port validation and undo/redo."""

    def __init__(self, catalog: PartCatalog, base_part_id: str) -> None:
        if catalog is None:
            raise ValueError("catalog must be provided")
        self.catalog = catalog
        self._ux = CompositionUxController()
        self._listeners: list[Callable[[], None]] = []
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._counter = 0
        base = catalog.get(base_part_id)
        self.model = base.load_model()
        self.model.robot_name = f"assembly_{base_part_id}"
        self._placed: tuple[PlacedPart, ...] = (self._make_base(base),)
        self._sync_metadata()

    # ------------------------------------------------------------------ views
    @property
    def placed_parts(self) -> tuple[PlacedPart, ...]:
        """Placed instances, base first."""
        return self._placed

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Call ``listener`` after every change to the assembly."""
        if listener is None:
            raise ValueError("listener must be provided")
        self._listeners.append(listener)

    def link_names(self) -> tuple[str, ...]:
        """Names of every link in the assembled model."""
        return tuple(self.model.links)

    def link_edges(self) -> tuple[tuple[str, str], ...]:
        """(parent, child) link pairs for every joint."""
        return self.model.joint_edges()

    def validate(self) -> CompositionValidationResult:
        """Run composition validation on the current model."""
        return self.model.validate_composition()

    def instance(self, instance_id: str) -> PlacedPart:
        """Return a placed part by instance id."""
        for placed in self._placed:
            if placed.instance_id == instance_id:
                return placed
        raise KeyError(f"unknown instance: {instance_id}")

    def all_ports(self) -> tuple[AttachmentPoint, ...]:
        """Every port of every placed part, with model-level link names."""
        return tuple(port for placed in self._placed for port in placed.ports)

    def occupied_ports(self) -> frozenset[str]:
        """Names of host ports that already have a part mated to them."""
        return frozenset(p.host_port for p in self._placed if p.host_port)

    def free_sockets(self) -> tuple[AttachmentPoint, ...]:
        """Typed sockets that nothing is mated to yet."""
        taken = self.occupied_ports()
        return tuple(
            port
            for port in self.all_ports()
            if port.polarity is PortPolarity.SOCKET and port.name not in taken
        )

    def host_port(self, name: str) -> AttachmentPoint | None:
        """Find a host port by its model-level name."""
        return next((p for p in self.all_ports() if p.name == name), None)

    # --------------------------------------------------------------- decisions
    def evaluate_drop(self, part_id: str, host_port_name: str) -> DropDecision:
        """Say whether ``part_id`` may be dropped on ``host_port_name``.

        Never mutates the assembly. Unknown ports and parts are rejections
        with a reason rather than exceptions, so drag hover can call this.
        """
        if part_id is None or host_port_name is None:
            raise ValueError("part_id and host_port_name must be provided")
        try:
            part = self.catalog.get(part_id)
        except KeyError:
            return self._reject(part_id, host_port_name, f"unknown part {part_id!r}")
        host = self.host_port(host_port_name)
        if host is None:
            return self._reject(part_id, host_port_name, "unknown host port")
        if host_port_name in self.occupied_ports():
            return self._reject(part_id, host_port_name, "port is already occupied")
        plug, verdict = find_mating_plug(host, part)
        if plug is None:
            return self._reject(part_id, host_port_name, verdict.reason)
        if plug.link_name != part.root_link():
            return self._reject(
                part_id, host_port_name, "part plug is not on the part's root link"
            )
        return self._preview(part, host)

    def attach(self, part_id: str, host_port_name: str) -> PlacedPart:
        """Drop a part onto a host port; raises :class:`AssemblyError`."""
        decision = self.evaluate_drop(part_id, host_port_name)
        if not decision.accepted:
            raise AssemblyError(decision.reason)
        part = self.catalog.get(part_id)
        host = self.host_port(host_port_name)
        if host is None:
            raise AssemblyError("unknown host port")
        self._push_undo()
        self._counter += 1
        instance_id = f"{part_id}_{self._counter}"
        prefix = f"{instance_id}__"
        before_links, before_joints = set(self.model.links), set(self.model.joints)
        self._ux.commit_drop(
            payload=_payload(part, prefix),
            target_model=self.model,
            source_model=part.load_model(),
            selection=_selection(host, prefix),
        )
        placed = PlacedPart(
            instance_id=instance_id,
            part_id=part_id,
            prefix=prefix,
            host_port=host_port_name,
            host_instance=self._owner_of(host_port_name),
            links=tuple(sorted(set(self.model.links) - before_links)),
            joints=tuple(sorted(set(self.model.joints) - before_joints)),
            ports=_prefixed_ports(part, prefix, self.model),
        )
        self._placed = (*self._placed, placed)
        self._finish_change()
        return placed

    def detach(self, instance_id: str) -> tuple[str, ...]:
        """Remove a part and everything mated below it; returns removed ids."""
        target = self.instance(instance_id)
        if target.host_port is None:
            raise AssemblyError("the base part cannot be detached")
        doomed = self._descendants(instance_id)
        self._push_undo()
        for placed in self._placed:
            if placed.instance_id in doomed:
                for link in placed.links:
                    self.model.remove_link(link)
                for joint in placed.joints:
                    self.model.remove_joint(joint)
        self._placed = tuple(p for p in self._placed if p.instance_id not in doomed)
        self._finish_change()
        return tuple(sorted(doomed))

    # ------------------------------------------------------------ undo / redo
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        """Revert the last change; False when there is nothing to undo."""
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        """Re-apply an undone change; False when there is nothing to redo."""
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    # ------------------------------------------------------------ persistence
    def to_urdf(self, *, force: bool = False) -> str:
        """Serialize to URDF with the port/instance record embedded."""
        self._sync_metadata()
        return self.model.to_xml(force=force)

    @classmethod
    def from_urdf(cls, urdf_xml: str, catalog: PartCatalog) -> AssemblySession:
        """Rebuild a session from :meth:`to_urdf` output."""
        if not urdf_xml:
            raise ValueError("urdf_xml must be provided")
        model = URDFModel.from_element(DefusedET.fromstring(urdf_xml))
        record = _read_record(model)
        session = cls.__new__(cls)
        session.catalog = catalog
        session._ux = CompositionUxController()
        session._listeners, session._undo, session._redo = [], [], []
        session._counter = int(record["counter"])
        session.model = model
        session._placed = tuple(_placed_from_dict(d) for d in record["placed"])
        return session

    # --------------------------------------------------------------- internal
    def _make_base(self, base: PartSpec) -> PlacedPart:
        return PlacedPart(
            instance_id=f"{base.part_id}_0",
            part_id=base.part_id,
            prefix="",
            host_port=None,
            host_instance=None,
            links=tuple(self.model.links),
            joints=tuple(self.model.joints),
            ports=base.ports,
        )

    @staticmethod
    def _reject(
        part_id: str, host_port: str, reason: str, findings: tuple = ()
    ) -> DropDecision:
        return DropDecision(False, reason, part_id, host_port, findings)

    def _preview(self, part: PartSpec, host: AttachmentPoint) -> DropDecision:
        prefix = f"{part.part_id}_preview__"
        preview = self._ux.preview_drop(
            payload=_payload(part, prefix),
            target_model=self.model,
            source_model=part.load_model(),
            selection=_selection(host, prefix),
        )
        findings = preview.validation.findings if preview.validation else ()
        if preview.state != "ready":
            errors = [f.message for f in findings if f.severity == "error"]
            reason = "; ".join(errors) or preview.message
            return self._reject(part.part_id, host.name, reason, findings)
        return DropDecision(True, "compatible", part.part_id, host.name, findings)

    def _owner_of(self, port_name: str) -> str | None:
        for placed in self._placed:
            if any(p.name == port_name for p in placed.ports):
                return placed.instance_id
        return None

    def _descendants(self, instance_id: str) -> set[str]:
        doomed = {instance_id}
        changed = True
        while changed:
            changed = False
            for placed in self._placed:
                if placed.host_instance in doomed and placed.instance_id not in doomed:
                    doomed.add(placed.instance_id)
                    changed = True
        return doomed

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(self.model.clone(), self._placed, self._counter)

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _restore(self, snapshot: _Snapshot) -> None:
        self.model = snapshot.model
        self._placed = snapshot.placed
        self._counter = snapshot.counter
        self._finish_change()

    def _finish_change(self) -> None:
        self._sync_metadata()
        for listener in list(self._listeners):
            listener()

    def _sync_metadata(self) -> None:
        record = {
            "schema": ASSEMBLY_SCHEMA,
            "counter": self._counter,
            "placed": [_placed_to_dict(p) for p in self._placed],
        }
        element = ET.Element(ASSEMBLY_TAG)
        element.text = json.dumps(record, sort_keys=True)
        self.model.replace_extension(element)


def _payload(part: PartSpec, prefix: str) -> CompositionDragPayload:
    return CompositionDragPayload(
        category=part.category,
        key=part.part_id,
        name=part.name,
        format_badge="URDF",
        source_prefix=prefix,
    )


def _selection(host: AttachmentPoint, prefix: str) -> AttachmentSelection:
    return AttachmentSelection(
        target_link=host.link_name,
        attachment_name=host.name,
        interface_xyz=host.interface_frame.xyz,
        interface_rpy=host.interface_frame.rpy,
        source_prefix=prefix,
    )


def _prefixed_ports(
    part: PartSpec, prefix: str, model: URDFModel
) -> tuple[AttachmentPoint, ...]:
    """Ports of a placed part, renamed to the model's prefixed link names."""
    ports = []
    for port in part.ports:
        if port.polarity is PortPolarity.PLUG:
            continue
        link = f"{prefix}{port.link_name}"
        if link not in model.links:
            continue
        ports.append(
            AttachmentPoint(
                name=f"{prefix}{port.name}",
                link_name=link,
                role=port.role,
                interface_frame=port.interface_frame,
                max_payload_kg=port.max_payload_kg,
                tags=port.tags,
                port_type=port.port_type,
                polarity=port.polarity,
            )
        )
    return tuple(ports)


def _placed_to_dict(placed: PlacedPart) -> dict:
    return {
        "instance_id": placed.instance_id,
        "part_id": placed.part_id,
        "prefix": placed.prefix,
        "host_port": placed.host_port,
        "host_instance": placed.host_instance,
        "links": list(placed.links),
        "joints": list(placed.joints),
        "ports": [p.to_dict() for p in placed.ports],
    }


def _placed_from_dict(data: dict) -> PlacedPart:
    ports = []
    for raw in data["ports"]:
        point, warnings = parse_attachment_point(raw, "port")
        if point is None or warnings:
            raise AssemblyError(f"invalid embedded port: {warnings}")
        ports.append(point)
    return PlacedPart(
        instance_id=data["instance_id"],
        part_id=data["part_id"],
        prefix=data["prefix"],
        host_port=data["host_port"],
        host_instance=data["host_instance"],
        links=tuple(data["links"]),
        joints=tuple(data["joints"]),
        ports=tuple(ports),
    )


def _read_record(model: URDFModel) -> dict:
    for element in model.other_elements:
        if element.tag == ASSEMBLY_TAG and element.text:
            record = json.loads(element.text)
            if record.get("schema") != ASSEMBLY_SCHEMA:
                raise AssemblyError("unsupported assembly record schema")
            return record
    raise AssemblyError("URDF has no embedded assembly record")


__all__ = [
    "AssemblyError",
    "AssemblySession",
    "DropDecision",
    "PlacedPart",
]
