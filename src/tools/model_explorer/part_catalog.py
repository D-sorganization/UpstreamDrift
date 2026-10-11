"""Browsable library of assembly parts with typed attachment ports (CMB-8).

The catalog is the pure-data half of the drag-and-drop component library. A
:class:`PartSpec` bundles a URDF with the typed ports it exposes: *sockets*
other parts can mate with and *plugs* that mate with someone else's socket.
Parts come from two places: bundled synthetic parts (limbs, heads, shoes,
torso, robot arm, pedestal) generated here with valid inertials, and existing
URDF files in the repository (for example the driver) with ports declared in
code or a ``*.attachments.json`` sidecar.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks

from src.shared.python.model_generation.editor.attachment_ports import (
    PortCompatibility,
    PortPolarity,
    PortType,
    check_port_compatibility,
)
from src.tools.model_explorer._part_builders import (
    BUNDLED_CATEGORY_LABELS,
    bundled_parts,
)
from src.tools.model_explorer.attachment_manifest import (
    AttachmentInterfaceFrame,
    AttachmentPoint,
    load_attachment_manifest,
)
from src.tools.model_explorer.frankenstein_editor.model import URDFModel

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER_URDF = REPO_ROOT / "src/shared/urdf/golf_clubs/driver/driver.urdf"


@dataclass(frozen=True)
class PartSpec:
    """One library part: a URDF plus the typed ports it exposes."""

    part_id: str
    name: str
    category: str
    description: str
    urdf_xml: str
    ports: tuple[AttachmentPoint, ...]
    source: str = "bundled"

    def __post_init__(self) -> None:
        if not self.part_id.strip():
            raise ValueError("part_id must be non-empty")
        if not self.category.strip():
            raise ValueError("category must be non-empty")

    def load_model(self) -> URDFModel:
        """Parse a fresh, independent model for this part."""
        model = URDFModel.from_element(DefusedET.fromstring(self.urdf_xml))
        model.attachment_points = tuple(p.to_dict() for p in self.ports)
        return model

    def root_link(self) -> str:
        """Name of the single root link of this part."""
        model = self.load_model()
        children = {
            joint.find("child").get("link")  # type: ignore[union-attr]
            for joint in model.joints.values()
            if joint.find("child") is not None
        }
        roots = [name for name in model.links if name not in children]
        if len(roots) != 1:
            raise ValueError(f"part {self.part_id} must have exactly one root link")
        return roots[0]

    def mass_kg(self) -> float:
        """Sum of the link masses declared in the part's inertial blocks."""
        total = 0.0
        for link in self.load_model().links.values():
            mass = link.find("inertial/mass")
            if mass is not None:
                total += float(mass.get("value", "0"))
        return total

    def plug_ports(self) -> tuple[AttachmentPoint, ...]:
        """Typed plugs: how this part mates with a host socket."""
        return tuple(p for p in self.ports if p.polarity is PortPolarity.PLUG)

    def socket_ports(self) -> tuple[AttachmentPoint, ...]:
        """Typed sockets: where further parts can be dropped onto this part."""
        return tuple(p for p in self.ports if p.polarity is PortPolarity.SOCKET)

    def port(self, name: str) -> AttachmentPoint | None:
        """Look a declared port up by name."""
        return next((p for p in self.ports if p.name == name), None)


def find_mating_plug(
    host_port: AttachmentPoint,
    part: PartSpec,
) -> tuple[AttachmentPoint | None, PortCompatibility]:
    """Find the plug of ``part`` that mates with ``host_port``.

    Returns ``(plug, ok)`` for the first compatible plug, otherwise
    ``(None, reason)`` carrying the first plug's rejection. Untyped host
    ports never mate.
    """
    if host_port is None:
        raise ValueError("host_port must be provided")
    if part is None:
        raise ValueError("part must be provided")
    host = host_port.typed_port()
    if host is None:
        return None, PortCompatibility(False, f"port {host_port.name!r} is untyped")
    plugs = part.plug_ports()
    if not plugs:
        return None, PortCompatibility(False, f"{part.name} has no plug port")
    first: PortCompatibility | None = None
    for plug in plugs:
        typed = plug.typed_port()
        if typed is None:
            continue
        verdict = check_port_compatibility(host, typed, part_mass_kg=part.mass_kg())
        if verdict.ok:
            return plug, verdict
        first = first or verdict
    return None, first or PortCompatibility(False, "no compatible plug")


def can_mate(host_port: AttachmentPoint, part: PartSpec) -> PortCompatibility:
    """Whether ``part`` has a plug compatible with ``host_port``."""
    return find_mating_plug(host_port, part)[1]


class PartCatalog:
    """Searchable, category-grouped collection of :class:`PartSpec`."""

    def __init__(self, parts: Iterable[PartSpec] = ()) -> None:
        self._parts: dict[str, PartSpec] = {}
        self._category_labels: dict[str, str] = dict(BUNDLED_CATEGORY_LABELS)
        for part in parts:
            self.register(part)

    @classmethod
    def bundled(cls) -> PartCatalog:
        """Catalog of the bundled parts plus repository URDF parts."""
        catalog = cls(bundled_parts())
        if DRIVER_URDF.exists():
            catalog.register(driver_part(DRIVER_URDF))
        return catalog

    def register(self, part: PartSpec) -> None:
        """Add a part; ids must be unique."""
        if part is None:
            raise ValueError("part must be provided")
        if part.part_id in self._parts:
            raise ValueError(f"duplicate part id: {part.part_id}")
        self._parts[part.part_id] = part
        self._category_labels.setdefault(
            part.category, part.category.replace("_", " ").title()
        )

    def register_urdf_file(
        self,
        path: Path,
        *,
        part_id: str,
        name: str,
        category: str,
        description: str = "",
    ) -> PartSpec:
        """Register a URDF whose ports come from its attachments sidecar."""
        if path is None:
            raise ValueError("path must be provided")
        manifest = load_attachment_manifest(path)
        part = PartSpec(
            part_id=part_id,
            name=name,
            category=category,
            description=description,
            urdf_xml=Path(path).read_text(encoding="utf-8"),
            ports=manifest.attachment_points,
            source=str(path),
        )
        self.register(part)
        return part

    def get(self, part_id: str) -> PartSpec:
        """Return a part by id; raises ``KeyError`` when unknown."""
        if part_id not in self._parts:
            raise KeyError(f"unknown part id: {part_id}")
        return self._parts[part_id]

    def categories(self) -> tuple[tuple[str, str], ...]:
        """(category, label) pairs that have at least one part, sorted."""
        used = sorted({p.category for p in self._parts.values()})
        return tuple((c, self._category_labels[c]) for c in used)

    def list_parts(
        self,
        *,
        category: str | None = None,
        query: str = "",
        compatible_with: AttachmentPoint | None = None,
    ) -> tuple[PartSpec, ...]:
        """Browse parts, optionally by category, search text and port fit."""
        if query is None:
            raise ValueError("query must be provided")
        tokens = [t for t in query.lower().split() if t]
        result = []
        for part in self._parts.values():
            if category is not None and part.category != category:
                continue
            haystack = f"{part.name} {part.category} {part.description}".lower()
            if not all(token in haystack for token in tokens):
                continue
            if compatible_with is not None and not can_mate(compatible_with, part).ok:
                continue
            result.append(part)
        return tuple(sorted(result, key=lambda p: (p.category, p.name)))

    def all_ids(self) -> tuple[str, ...]:
        """Every registered part id in insertion order."""
        return tuple(self._parts)


def driver_part(path: Path) -> PartSpec:
    """Golf driver from the repository URDF; its grip plug mates with a hand."""
    plug = AttachmentPoint(
        name="grip_plug",
        link_name="base_link",
        role="club-grip",
        interface_frame=AttachmentInterfaceFrame(),
        tags=("club",),
        port_type=PortType.GRIP,
        polarity=PortPolarity.PLUG,
    )
    return PartSpec(
        part_id="club_driver",
        name="Driver",
        category="club",
        description="Golf driver from src/shared/urdf/golf_clubs/driver",
        urdf_xml=Path(path).read_text(encoding="utf-8"),
        ports=(plug,),
        source=str(path),
    )


__all__ = ["PartCatalog", "PartSpec", "can_mate", "driver_part", "find_mating_plug"]
