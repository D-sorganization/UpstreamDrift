from __future__ import annotations

import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from dataclasses import dataclass, field

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ChainNode:
    """Represents a node in the kinematic chain."""

    name: str
    link_element: ET.Element | None = None
    joint_to_parent: ET.Element | None = None
    parent: ChainNode | None = None
    children: list[ChainNode] = field(default_factory=list)
    depth: int = 0

    def get_chain_to_root(self) -> list[ChainNode]:
        """Get the chain from this node to the root."""
        chain = [self]
        current = self.parent
        while current:
            chain.append(current)
            current = current.parent
        return list(reversed(chain))

    def get_all_descendants(self) -> list[ChainNode]:
        """Get all descendant nodes."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    def is_end_effector(self) -> bool:
        """Check if this could be an end effector (leaf with common naming)."""
        if not self.is_leaf():
            return False
        lower_name = self.name.lower()
        end_effector_hints = [
            "hand",
            "gripper",
            "tool",
            "effector",
            "finger",
            "tip",
            "end",
            "head",
            "foot",
            "palm",
        ]
        return any(hint in lower_name for hint in end_effector_hints)


class KinematicTree:
    """Represents the kinematic tree structure of a URDF."""

    def __init__(self) -> None:
        """Initialize the kinematic tree."""
        self.root: ChainNode | None = None
        self.nodes: dict[str, ChainNode] = {}

    @precondition(
        lambda self, urdf_content: (
            urdf_content is not None and len(urdf_content.strip()) > 0
        ),
        "URDF content must be a non-empty string",
    )
    def build_from_urdf(self, urdf_content: str) -> None:  # noqa: C901
        """Build the tree from URDF XML content."""
        if not (urdf_content is not None):
            raise ValueError("urdf_content must be provided")
        if not (urdf_content is not None):
            raise ValueError("urdf_content must be provided")
        try:
            root_elem = DefusedET.fromstring(urdf_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse URDF: {e}")
            return

        self.nodes.clear()
        self.root = None

        # Extract links
        links = {}
        for link in root_elem.findall("link"):
            name = link.get("name", "")
            links[name] = link
            self.nodes[name] = ChainNode(name=name, link_element=link)

        # Extract joints and build hierarchy
        child_links = set()
        for joint in root_elem.findall("joint"):
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")

            if parent_elem is None or child_elem is None:
                continue

            parent_name = parent_elem.get("link", "")
            child_name = child_elem.get("link", "")

            if parent_name in self.nodes and child_name in self.nodes:
                parent_node = self.nodes[parent_name]
                child_node = self.nodes[child_name]

                child_node.parent = parent_node
                child_node.joint_to_parent = joint
                parent_node.children.append(child_node)
                child_links.add(child_name)

        # Find root (link that is never a child)
        for name, node in self.nodes.items():
            if name not in child_links:
                if self.root is None:
                    self.root = node
                else:
                    # Multiple roots - use first one
                    logger.warning(
                        f"Multiple root links found. Using '{self.root.name}'"
                    )

        # Calculate depths
        self._calculate_depths()

    def _calculate_depths(self) -> None:
        """Calculate depth for each node."""
        if self.root is None:
            return

        def set_depth(node: ChainNode, depth: int) -> None:
            """Recursively assign depth values to each node."""
            if not (node is not None):
                raise ValueError("node must be provided")
            if not (node is not None):
                raise ValueError("node must be provided")
            node.depth = depth
            for child in node.children:
                set_depth(child, depth + 1)

        set_depth(self.root, 0)

    @precondition(
        lambda self, from_link, to_link: (
            from_link is not None and len(from_link.strip()) > 0
        ),
        "Source link name must be a non-empty string",
    )
    @precondition(
        lambda self, from_link, to_link: (
            to_link is not None and len(to_link.strip()) > 0
        ),
        "Target link name must be a non-empty string",
    )
    def get_chain(self, from_link: str, to_link: str) -> list[ChainNode]:  # noqa: C901
        """Get the chain between two links.

        Args:
            from_link: Starting link name
            to_link: Ending link name

        Returns:
            List of nodes in the chain (may be empty if no path exists)
        """
        if not (from_link is not None):
            raise ValueError("from_link must be provided")
        if not (from_link is not None):
            raise ValueError("from_link must be provided")
        if from_link not in self.nodes or to_link not in self.nodes:
            return []

        from_node = self.nodes[from_link]
        to_node = self.nodes[to_link]

        # Get paths to root
        from_path = from_node.get_chain_to_root()
        to_path = to_node.get_chain_to_root()

        # Find common ancestor
        from_set = {n.name for n in from_path}
        common_ancestor = None
        for node in to_path:
            if node.name in from_set:
                common_ancestor = node
                break

        if common_ancestor is None:
            return []

        # Build path: from_link -> ... -> common_ancestor -> ... -> to_link.
        # from_path / to_path are ordered root -> self, so we walk from_path in
        # reverse (self up to common ancestor) and to_path forward starting
        # just after the common ancestor.
        chain: list[ChainNode] = []

        for node in reversed(from_path):
            chain.append(node)
            if node.name == common_ancestor.name:
                break

        # Find index of common ancestor in to_path and append nodes after it
        ancestor_idx = next(
            (i for i, n in enumerate(to_path) if n.name == common_ancestor.name),
            -1,
        )
        if ancestor_idx >= 0:
            chain.extend(to_path[ancestor_idx + 1 :])

        return chain

    def get_all_chains(self) -> list[list[ChainNode]]:
        """Get all chains from root to leaves."""
        chains = []

        def collect_chains(node: ChainNode, current_chain: list[ChainNode]) -> None:
            """Recursively collect root-to-leaf chains."""
            if not (node is not None):
                raise ValueError("node must be provided")
            if not (node is not None):
                raise ValueError("node must be provided")
            current_chain = current_chain + [node]
            if node.is_leaf():
                chains.append(current_chain)
            else:
                for child in node.children:
                    collect_chains(child, current_chain)

        if self.root:
            collect_chains(self.root, [])

        return chains

    def get_end_effectors(self) -> list[ChainNode]:
        """Get all potential end effectors."""
        return [node for node in self.nodes.values() if node.is_leaf()]

    def get_branch_points(self) -> list[ChainNode]:
        """Get all branch points (nodes with multiple children)."""
        return [node for node in self.nodes.values() if len(node.children) > 1]
