"""Composition validation for first-party Frankenstein editor URDF models."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.tools.model_explorer.frankenstein_editor.model import URDFModel

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class CompositionFinding:
    """A structured validation finding for a composed URDF model."""

    code: str
    severity: Severity
    message: str
    elements: tuple[str, ...] = ()
    category: str = "topology"


@dataclass(frozen=True)
class CompositionValidationResult:
    """Validation result with convenience accessors for gate decisions."""

    findings: tuple[CompositionFinding, ...]

    @property
    def errors(self) -> tuple[CompositionFinding, ...]:
        """Findings that must block export unless forced."""
        return tuple(
            finding for finding in self.findings if finding.severity == "error"
        )

    @property
    def warnings(self) -> tuple[CompositionFinding, ...]:
        """Findings that should be surfaced without blocking export."""
        return tuple(
            finding for finding in self.findings if finding.severity == "warning"
        )

    @property
    def ok(self) -> bool:
        """Whether validation found no blocking errors."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise an actionable exception when blocking findings exist."""
        if self.errors:
            raise CompositionValidationError(self.errors)


class CompositionValidationError(ValueError):
    """Raised when composition validation blocks export."""

    def __init__(self, findings: tuple[CompositionFinding, ...]) -> None:
        self.findings = findings
        messages = "; ".join(finding.message for finding in findings)
        super().__init__(f"Invalid Frankenstein composition: {messages}")


@dataclass(frozen=True)
class _JointEdge:
    name: str
    parent: str
    child: str
    origin: tuple[float, float, float]


@dataclass(frozen=True)
class _Aabb:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]


class CompositionValidator:
    """Validate URDF composition invariants before first-party editor export."""

    _FLOAT_TOLERANCE = 1e-9
    _FIXED_JOINT_TYPES = {"fixed"}
    _SUBTREE_MASS_RATIO_WARNING = 2.0

    def validate_model(self, model: URDFModel) -> CompositionValidationResult:
        """Validate a first-party Frankenstein editor model."""
        if model is None:
            raise ValueError("model must be provided")
        root = ET.Element("robot", name=model.robot_name)
        root.extend(copy.deepcopy(material) for material in model.materials.values())
        root.extend(copy.deepcopy(link) for link in model.links.values())
        root.extend(copy.deepcopy(joint) for joint in model.joints.values())
        root.extend(copy.deepcopy(element) for element in model.other_elements)
        return self.validate_xml_root(root)

    def validate_xml_root(self, root: ET.Element) -> CompositionValidationResult:
        """Validate a URDF XML root before it is collapsed into dictionaries."""
        if root is None:
            raise ValueError("root must be provided")
        findings: list[CompositionFinding] = []

        link_elements = list(root.findall("link"))
        joint_elements = list(root.findall("joint"))
        link_names = [self._name(link) for link in link_elements]
        joint_names = [self._name(joint) for joint in joint_elements]

        self._check_unique_names("link", link_names, findings)
        self._check_unique_names("joint", joint_names, findings)
        self._check_topology(link_elements, joint_elements, findings)
        self._check_moving_link_inertials(link_elements, joint_elements, findings)
        self._check_subtree_mass_ratios(link_elements, joint_elements, findings)
        self._check_attachment_geometry_overlap(link_elements, joint_elements, findings)

        return CompositionValidationResult(tuple(findings))

    def _check_unique_names(
        self,
        element_type: Literal["link", "joint"],
        names: list[str],
        findings: list[CompositionFinding],
    ) -> None:
        missing = [index for index, name in enumerate(names) if not name]
        if missing:
            findings.append(
                CompositionFinding(
                    code=f"missing_{element_type}_name",
                    severity="error",
                    message=(
                        f"{element_type.title()} elements at indexes "
                        f"{missing} are missing names."
                    ),
                    category="topology",
                )
            )

        counts = Counter(name for name in names if name)
        for name, count in sorted(counts.items()):
            if count > 1:
                findings.append(
                    CompositionFinding(
                        code=f"duplicate_{element_type}_name",
                        severity="error",
                        message=(
                            f"Duplicate {element_type} name '{name}' appears "
                            f"{count} times."
                        ),
                        elements=(name,),
                        category="topology",
                    )
                )

    def _check_topology(
        self,
        link_elements: list[ET.Element],
        joint_elements: list[ET.Element],
        findings: list[CompositionFinding],
    ) -> None:
        links = {self._name(link) for link in link_elements if self._name(link)}
        if not links:
            findings.append(
                CompositionFinding(
                    code="missing_links",
                    severity="error",
                    message="Composition must contain at least one named link.",
                    category="topology",
                )
            )
            return

        children: set[str] = set()
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        parent_counts: dict[str, list[str]] = defaultdict(list)

        for joint in joint_elements:
            joint_name = self._name(joint) or "<unnamed joint>"
            parent = self._joint_endpoint(joint, "parent")
            child = self._joint_endpoint(joint, "child")
            missing_refs = [
                ref for ref in (parent, child) if not ref or ref not in links
            ]
            if missing_refs:
                refs = ", ".join(repr(ref or "<missing>") for ref in missing_refs)
                findings.append(
                    CompositionFinding(
                        code="orphan_joint",
                        severity="error",
                        message=(
                            f"Joint '{joint_name}' references missing link(s): {refs}."
                        ),
                        elements=(
                            joint_name,
                            *tuple(ref for ref in missing_refs if ref),
                        ),
                        category="topology",
                    )
                )
                continue

            assert parent is not None
            assert child is not None
            children.add(child)
            adjacency[parent].append((child, joint_name))
            parent_counts[child].append(joint_name)

        for child, joint_names in sorted(parent_counts.items()):
            if len(joint_names) > 1:
                findings.append(
                    CompositionFinding(
                        code="multiple_link_parents",
                        severity="error",
                        message=(
                            f"Link '{child}' has multiple parent joints: "
                            f"{', '.join(joint_names)}."
                        ),
                        elements=(child, *tuple(joint_names)),
                        category="topology",
                    )
                )

        self._check_roots_and_connectivity(links, children, adjacency, findings)
        self._check_cycles(links, adjacency, findings)

    def _check_roots_and_connectivity(
        self,
        links: set[str],
        children: set[str],
        adjacency: dict[str, list[tuple[str, str]]],
        findings: list[CompositionFinding],
    ) -> None:
        roots = sorted(links - children)
        if len(roots) != 1:
            findings.append(
                CompositionFinding(
                    code="invalid_root_count",
                    severity="error",
                    message=(
                        "Composition must have exactly one root link; "
                        f"found {len(roots)}: {', '.join(roots) or '<none>'}."
                    ),
                    elements=tuple(roots),
                    category="topology",
                )
            )
            return

        visited: set[str] = set()
        stack = [roots[0]]
        while stack:
            link = stack.pop()
            if link in visited:
                continue
            visited.add(link)
            stack.extend(child for child, _joint_name in adjacency.get(link, ()))

        disconnected = sorted(links - visited)
        if disconnected:
            findings.append(
                CompositionFinding(
                    code="disconnected_links",
                    severity="error",
                    message=(
                        "Composition has links disconnected from root "
                        f"'{roots[0]}': {', '.join(disconnected)}."
                    ),
                    elements=tuple(disconnected),
                    category="topology",
                )
            )

    def _check_cycles(
        self,
        links: set[str],
        adjacency: dict[str, list[tuple[str, str]]],
        findings: list[CompositionFinding],
    ) -> None:
        state: dict[str, Literal["visiting", "visited"]] = {}
        path: list[tuple[str, str | None]] = []

        def visit(link: str) -> bool:
            state[link] = "visiting"
            path.append((link, None))
            for child, joint_name in adjacency.get(link, ()):
                if state.get(child) == "visiting":
                    cycle_links, cycle_joints = self._format_cycle(path, child)
                    cycle_joints.append(joint_name)
                    findings.append(
                        CompositionFinding(
                            code="topology_cycle",
                            severity="error",
                            message=(
                                "Kinematic cycle detected through links "
                                f"{' -> '.join(cycle_links)} using joints "
                                f"{', '.join(cycle_joints)}."
                            ),
                            elements=tuple(cycle_links + cycle_joints),
                            category="topology",
                        )
                    )
                    return True
                if child not in state:
                    path[-1] = (link, joint_name)
                    if visit(child):
                        return True
                    path[-1] = (link, None)
            path.pop()
            state[link] = "visited"
            return False

        for link in sorted(links):
            if link not in state and visit(link):
                return

    def _format_cycle(
        self,
        path: list[tuple[str, str | None]],
        repeated_link: str,
    ) -> tuple[list[str], list[str]]:
        start = next(
            index
            for index, (link, _joint_name) in enumerate(path)
            if link == repeated_link
        )
        cycle_path = path[start:]
        cycle_links = [link for link, _joint_name in cycle_path] + [repeated_link]
        cycle_joints = [
            joint_name for _link, joint_name in cycle_path if joint_name is not None
        ]
        return cycle_links, cycle_joints

    def _check_moving_link_inertials(
        self,
        link_elements: list[ET.Element],
        joint_elements: list[ET.Element],
        findings: list[CompositionFinding],
    ) -> None:
        links_by_name = {
            self._name(link): link for link in link_elements if self._name(link)
        }
        moving_children = {
            child
            for joint in joint_elements
            if joint.get("type", "fixed") not in self._FIXED_JOINT_TYPES
            for child in [self._joint_endpoint(joint, "child")]
            if child
        }

        for link_name in sorted(moving_children):
            link = links_by_name.get(link_name)
            if link is None:
                continue
            inertial = link.find("inertial")
            if inertial is None:
                findings.append(
                    CompositionFinding(
                        code="missing_inertial",
                        severity="error",
                        message=(
                            f"Moving link '{link_name}' must include an inertial block."
                        ),
                        elements=(link_name,),
                        category="inertial",
                    )
                )
                continue

            self._check_mass(link_name, inertial, findings)
            self._check_inertia(link_name, inertial, findings)

    def _check_subtree_mass_ratios(
        self,
        link_elements: list[ET.Element],
        joint_elements: list[ET.Element],
        findings: list[CompositionFinding],
    ) -> None:
        links_by_name = self._links_by_name(link_elements)
        edges = self._valid_joint_edges(joint_elements, set(links_by_name))
        if not edges:
            return

        adjacency = self._adjacency(edges)
        parent_by_child = {edge.child: edge.parent for edge in edges}
        masses: dict[str, float] = {}
        for name, link in links_by_name.items():
            mass = self._link_mass(link)
            if mass is not None:
                masses[name] = mass

        for edge in edges:
            parent_chain_mass = self._parent_chain_mass(
                edge.parent,
                parent_by_child,
                masses,
            )
            subtree_mass = self._subtree_mass(edge.child, adjacency, masses)
            if parent_chain_mass <= 0 or subtree_mass <= 0:
                continue
            ratio = subtree_mass / parent_chain_mass
            if ratio > self._SUBTREE_MASS_RATIO_WARNING:
                findings.append(
                    CompositionFinding(
                        code="subtree_mass_ratio",
                        severity="warning",
                        message=(
                            f"Attached subtree rooted at '{edge.child}' has mass "
                            f"{subtree_mass:g}, which is {ratio:.2f}x the parent "
                            f"chain mass {parent_chain_mass:g} at '{edge.parent}'."
                        ),
                        elements=(edge.parent, edge.child, edge.name),
                        category="inertial",
                    )
                )

    def _check_attachment_geometry_overlap(
        self,
        link_elements: list[ET.Element],
        joint_elements: list[ET.Element],
        findings: list[CompositionFinding],
    ) -> None:
        links_by_name = self._links_by_name(link_elements)
        aabbs = {
            name: aabb
            for name, link in links_by_name.items()
            for aabb in [self._link_aabb(link)]
            if aabb is not None
        }

        for edge in self._valid_joint_edges(joint_elements, set(links_by_name)):
            parent_aabb = aabbs.get(edge.parent)
            child_aabb = aabbs.get(edge.child)
            if parent_aabb is None or child_aabb is None:
                continue
            child_in_parent = self._translate_aabb(child_aabb, edge.origin)
            if self._aabbs_overlap(parent_aabb, child_in_parent):
                findings.append(
                    CompositionFinding(
                        code="geometry_overlap",
                        severity="warning",
                        message=(
                            f"Attachment joint '{edge.name}' places child link "
                            f"'{edge.child}' geometry inside parent link "
                            f"'{edge.parent}' geometry; inspect mount offset or "
                            "use a deliberate collision allowance."
                        ),
                        elements=(edge.parent, edge.child, edge.name),
                        category="geometry",
                    )
                )

    def _check_mass(
        self,
        link_name: str,
        inertial: ET.Element,
        findings: list[CompositionFinding],
    ) -> None:
        mass_element = inertial.find("mass")
        mass = self._float_attr(mass_element, "value")
        if mass is None or not math.isfinite(mass) or mass <= 0:
            findings.append(
                CompositionFinding(
                    code="invalid_mass",
                    severity="error",
                    message=(
                        f"Moving link '{link_name}' must have positive finite mass."
                    ),
                    elements=(link_name,),
                    category="inertial",
                )
            )

    def _check_inertia(
        self,
        link_name: str,
        inertial: ET.Element,
        findings: list[CompositionFinding],
    ) -> None:
        inertia = inertial.find("inertia")
        values = {
            name: self._float_attr(inertia, name)
            for name in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        }
        if any(value is None or not math.isfinite(value) for value in values.values()):
            findings.append(
                CompositionFinding(
                    code="invalid_inertia",
                    severity="error",
                    message=(
                        f"Moving link '{link_name}' must have finite inertia values."
                    ),
                    elements=(link_name,),
                    category="inertial",
                )
            )
            return

        ixx = values["ixx"]
        iyy = values["iyy"]
        izz = values["izz"]
        ixy = values["ixy"]
        ixz = values["ixz"]
        iyz = values["iyz"]
        assert ixx is not None
        assert iyy is not None
        assert izz is not None
        assert ixy is not None
        assert ixz is not None
        assert iyz is not None
        principal_minors = (
            ixx,
            iyy,
            izz,
            ixx * iyy - ixy * ixy,
            ixx * izz - ixz * ixz,
            iyy * izz - iyz * iyz,
            (
                ixx * iyy * izz
                + 2 * ixy * ixz * iyz
                - ixx * iyz * iyz
                - iyy * ixz * ixz
                - izz * ixy * ixy
            ),
        )
        if any(value < -self._FLOAT_TOLERANCE for value in principal_minors):
            findings.append(
                CompositionFinding(
                    code="invalid_inertia",
                    severity="error",
                    message=(
                        f"Moving link '{link_name}' must have positive "
                        "semidefinite inertia."
                    ),
                    elements=(link_name,),
                    category="inertial",
                )
            )

    @staticmethod
    def _links_by_name(link_elements: list[ET.Element]) -> dict[str, ET.Element]:
        return {
            link.get("name", "").strip(): link
            for link in link_elements
            if link.get("name", "").strip()
        }

    def _valid_joint_edges(
        self,
        joint_elements: list[ET.Element],
        links: set[str],
    ) -> tuple[_JointEdge, ...]:
        edges: list[_JointEdge] = []
        for joint in joint_elements:
            parent = self._joint_endpoint(joint, "parent")
            child = self._joint_endpoint(joint, "child")
            if parent not in links or child not in links:
                continue
            assert parent is not None
            assert child is not None
            edges.append(
                _JointEdge(
                    name=self._name(joint) or "<unnamed joint>",
                    parent=parent,
                    child=child,
                    origin=self._origin_xyz(joint),
                )
            )
        return tuple(edges)

    @staticmethod
    def _adjacency(edges: tuple[_JointEdge, ...]) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.parent].append(edge.child)
        return adjacency

    def _parent_chain_mass(
        self,
        parent: str,
        parent_by_child: dict[str, str],
        masses: dict[str, float],
    ) -> float:
        total = 0.0
        current: str | None = parent
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            total += masses.get(current, 0.0)
            current = parent_by_child.get(current)
        return total

    def _subtree_mass(
        self,
        root: str,
        adjacency: dict[str, list[str]],
        masses: dict[str, float],
    ) -> float:
        total = 0.0
        stack = [root]
        seen: set[str] = set()
        while stack:
            link = stack.pop()
            if link in seen:
                continue
            seen.add(link)
            total += masses.get(link, 0.0)
            stack.extend(adjacency.get(link, ()))
        return total

    def _link_mass(self, link: ET.Element) -> float | None:
        mass = self._float_attr(link.find("inertial/mass"), "value")
        if mass is None or not math.isfinite(mass) or mass <= 0:
            return None
        return mass

    def _link_aabb(self, link: ET.Element) -> _Aabb | None:
        aabbs = [
            aabb
            for tag in ("visual", "collision")
            for geometry in link.findall(tag)
            for aabb in [self._geometry_aabb(geometry)]
            if aabb is not None
        ]
        if not aabbs:
            return None
        minimum = (
            min(aabb.minimum[0] for aabb in aabbs),
            min(aabb.minimum[1] for aabb in aabbs),
            min(aabb.minimum[2] for aabb in aabbs),
        )
        maximum = (
            max(aabb.maximum[0] for aabb in aabbs),
            max(aabb.maximum[1] for aabb in aabbs),
            max(aabb.maximum[2] for aabb in aabbs),
        )
        return _Aabb(minimum=minimum, maximum=maximum)

    def _geometry_aabb(self, geometry_parent: ET.Element) -> _Aabb | None:
        origin = self._origin_xyz(geometry_parent)
        geometry = geometry_parent.find("geometry")
        if geometry is None:
            return None
        box = geometry.find("box")
        if box is not None:
            size = self._vector_attr(box, "size")
            if size is None:
                return None
            half_extents = (size[0] / 2, size[1] / 2, size[2] / 2)
            return self._aabb_from_half_extents(origin, half_extents)

        sphere = geometry.find("sphere")
        radius = self._float_attr(sphere, "radius")
        if radius is not None and math.isfinite(radius) and radius >= 0:
            return self._aabb_from_half_extents(origin, (radius, radius, radius))

        cylinder = geometry.find("cylinder")
        radius = self._float_attr(cylinder, "radius")
        length = self._float_attr(cylinder, "length")
        if (
            radius is not None
            and length is not None
            and math.isfinite(radius)
            and math.isfinite(length)
            and radius >= 0
            and length >= 0
        ):
            return self._aabb_from_half_extents(origin, (radius, radius, length / 2))

        return None

    @staticmethod
    def _aabb_from_half_extents(
        center: tuple[float, float, float],
        half_extents: tuple[float, float, float],
    ) -> _Aabb:
        return _Aabb(
            minimum=(
                center[0] - half_extents[0],
                center[1] - half_extents[1],
                center[2] - half_extents[2],
            ),
            maximum=(
                center[0] + half_extents[0],
                center[1] + half_extents[1],
                center[2] + half_extents[2],
            ),
        )

    @staticmethod
    def _translate_aabb(
        aabb: _Aabb,
        offset: tuple[float, float, float],
    ) -> _Aabb:
        return _Aabb(
            minimum=(
                aabb.minimum[0] + offset[0],
                aabb.minimum[1] + offset[1],
                aabb.minimum[2] + offset[2],
            ),
            maximum=(
                aabb.maximum[0] + offset[0],
                aabb.maximum[1] + offset[1],
                aabb.maximum[2] + offset[2],
            ),
        )

    @staticmethod
    def _aabbs_overlap(first: _Aabb, second: _Aabb) -> bool:
        return all(
            first.minimum[index] <= second.maximum[index]
            and second.minimum[index] <= first.maximum[index]
            for index in range(3)
        )

    @staticmethod
    def _name(element: ET.Element) -> str:
        return element.get("name", "").strip()

    @staticmethod
    def _joint_endpoint(
        joint: ET.Element, tag: Literal["parent", "child"]
    ) -> str | None:
        endpoint = joint.find(tag)
        if endpoint is None:
            return None
        link = endpoint.get("link")
        return link.strip() if link else None

    @staticmethod
    def _float_attr(element: ET.Element | None, attr: str) -> float | None:
        if element is None:
            return None
        value = element.get(attr)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _origin_xyz(self, element: ET.Element) -> tuple[float, float, float]:
        origin = element.find("origin")
        return self._vector_attr(origin, "xyz") or (0.0, 0.0, 0.0)

    @staticmethod
    def _vector_attr(
        element: ET.Element | None,
        attr: str,
    ) -> tuple[float, float, float] | None:
        if element is None:
            return None
        value = element.get(attr)
        if value is None:
            return None
        parts = value.split()
        if len(parts) != 3:
            return None
        try:
            vector = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            return None
        if not all(math.isfinite(component) for component in vector):
            return None
        return vector
