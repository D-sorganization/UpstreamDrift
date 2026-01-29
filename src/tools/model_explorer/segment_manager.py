"""Segment manager for handling URDF segment operations."""

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class SegmentManager:
    """Manager for URDF segments with support for parallel configurations."""

    def __init__(self) -> None:
        """Initialize the segment manager."""
        self.segments: dict[str, dict] = {}
        self.hierarchy: dict[str, list[str]] = {}  # parent -> [children]
        self.parallel_chains: list[dict] = []  # For parallel kinematic chains

    def add_segment(self, segment_data: dict) -> None:
        """Add a segment to the manager.

        Args:
            segment_data: Dictionary containing segment information.

        Raises:
            ValueError: If segment name already exists or is invalid.
        """
        name = segment_data.get("name")
        if not name:
            raise ValueError("Segment must have a name")

        if name in self.segments:
            raise ValueError(f"Segment '{name}' already exists")

        # Validate parent exists (if specified)
        parent = segment_data.get("parent")
        if parent and parent not in self.segments:
            raise ValueError(f"Parent segment '{parent}' does not exist")

        # Add segment
        self.segments[name] = segment_data.copy()

        # Update hierarchy
        if parent:
            if parent not in self.hierarchy:
                self.hierarchy[parent] = []
            self.hierarchy[parent].append(name)

        logger.info(f"Added segment: {name}")

    def remove_segment(self, name: str) -> None:
        """Remove a segment and all its children.

        Args:
            name: Name of the segment to remove.

        Raises:
            ValueError: If segment does not exist.
        """
        if name not in self.segments:
            raise ValueError(f"Segment '{name}' does not exist")

        # Remove all children first
        children = self.get_children(name)
        for child in children:
            self.remove_segment(child)

        # Remove from parent's children list
        parent = self.segments[name].get("parent")
        if parent and parent in self.hierarchy:
            self.hierarchy[parent] = [
                child for child in self.hierarchy[parent] if child != name
            ]

        # Remove segment
        del self.segments[name]

        # Remove from hierarchy if it was a parent
        if name in self.hierarchy:
            del self.hierarchy[name]

        logger.info(f"Removed segment: {name}")

    def modify_segment(self, segment_data: dict) -> None:
        """Modify an existing segment.

        Args:
            segment_data: Dictionary containing updated segment information.

        Raises:
            ValueError: If segment does not exist.
        """
        name = segment_data.get("name")
        if not name:
            raise ValueError("Segment must have a name")

        if name not in self.segments:
            raise ValueError(f"Segment '{name}' does not exist")

        old_parent = self.segments[name].get("parent")
        new_parent = segment_data.get("parent")

        # Update hierarchy if parent changed
        if old_parent != new_parent:
            # Remove from old parent
            if old_parent and old_parent in self.hierarchy:
                self.hierarchy[old_parent] = [
                    child for child in self.hierarchy[old_parent] if child != name
                ]

            # Add to new parent
            if new_parent:
                if new_parent not in self.hierarchy:
                    self.hierarchy[new_parent] = []
                self.hierarchy[new_parent].append(name)

        # Update segment
        self.segments[name] = segment_data.copy()

        logger.info(f"Modified segment: {name}")

    def get_segment(self, name: str) -> dict | None:
        """Get a segment by name.

        Args:
            name: Name of the segment.

        Returns:
            Segment data or None if not found.
        """
        return self.segments.get(name)

    def get_all_segments(self) -> dict[str, dict]:
        """Get all segments.

        Returns:
            Dictionary of all segments.
        """
        return self.segments.copy()

    def get_children(self, name: str) -> list[str]:
        """Get children of a segment.

        Args:
            name: Name of the parent segment.

        Returns:
            List of child segment names.
        """
        return self.hierarchy.get(name, []).copy()

    def get_root_segments(self) -> list[str]:
        """Get segments with no parent (root segments).

        Returns:
            List of root segment names.
        """
        return [name for name, data in self.segments.items() if not data.get("parent")]

    def get_hierarchy_order(self) -> list[str]:
        """Get segments in hierarchical order (parents before children).

        Returns:
            List of segment names in hierarchical order.
        """
        ordered = []
        visited = set()

        def visit_segment(name: str) -> None:
            """Visit a segment and its children."""
            if name in visited:
                return

            visited.add(name)
            ordered.append(name)

            # Visit children
            for child in self.get_children(name):
                visit_segment(child)

        # Start with root segments
        for root in self.get_root_segments():
            visit_segment(root)

        return ordered

    def validate_hierarchy(self) -> list[str]:
        """Validate the segment hierarchy.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check for circular dependencies
        def has_cycle(name: str, visited: set[str], rec_stack: set[str]) -> bool:
            """Check for cycles using DFS."""
            visited.add(name)
            rec_stack.add(name)

            for child in self.get_children(name):
                if child not in visited:
                    if has_cycle(child, visited, rec_stack):
                        return True
                elif child in rec_stack:
                    return True

            rec_stack.remove(name)
            return False

        visited: set[str] = set()
        for name in self.segments:
            if name not in visited:
                if has_cycle(name, visited, set()):
                    errors.append(
                        f"Circular dependency detected involving segment: {name}"
                    )

        # Check for orphaned segments (parent doesn't exist)
        for name, data in self.segments.items():
            parent = data.get("parent")
            if parent and parent not in self.segments:
                errors.append(f"Segment '{name}' has non-existent parent: '{parent}'")

        return errors

    def create_parallel_chain(self, chain_data: dict) -> None:
        """Create a parallel kinematic chain.

        Args:
            chain_data: Dictionary containing parallel chain information.
                Should include: name, segments, constraints
        """
        # Validate chain data
        if not chain_data.get("name"):
            raise ValueError("Parallel chain must have a name")

        segments = chain_data.get("segments", [])
        if len(segments) < 2:
            raise ValueError("Parallel chain must have at least 2 segments")

        # Validate all segments exist
        for segment_name in segments:
            if segment_name not in self.segments:
                raise ValueError(
                    f"Segment '{segment_name}' in parallel chain does not exist"
                )

        self.parallel_chains.append(chain_data.copy())
        logger.info(f"Created parallel chain: {chain_data['name']}")

    def get_parallel_chains(self) -> list[dict]:
        """Get all parallel kinematic chains.

        Returns:
            List of parallel chain data.
        """
        return self.parallel_chains.copy()

    def remove_parallel_chain(self, name: str) -> None:
        """Remove a parallel kinematic chain.

        Args:
            name: Name of the parallel chain to remove.

        Raises:
            ValueError: If chain does not exist.
        """
        original_count = len(self.parallel_chains)
        self.parallel_chains = [
            chain for chain in self.parallel_chains if chain.get("name") != name
        ]

        if len(self.parallel_chains) == original_count:
            raise ValueError(f"Parallel chain '{name}' does not exist")

        logger.info(f"Removed parallel chain: {name}")

    def get_segment_count(self) -> int:
        """Get the total number of segments.

        Returns:
            Number of segments.
        """
        return len(self.segments)

    def clear(self) -> None:
        """Clear all segments and parallel chains."""
        self.segments.clear()
        self.hierarchy.clear()
        self.parallel_chains.clear()
        logger.info("Segment manager cleared")

    def export_for_engine(self, engine: str) -> dict:
        """Export segment data optimized for a specific physics engine.

        Args:
            engine: Target physics engine ('mujoco', 'drake', 'pinocchio').

        Returns:
            Dictionary containing engine-specific data.
        """
        if engine.lower() == "mujoco":
            return self._export_for_mujoco()
        elif engine.lower() == "drake":
            return self._export_for_drake()
        elif engine.lower() == "pinocchio":
            return self._export_for_pinocchio()
        else:
            raise ValueError(f"Unsupported engine: {engine}")

    def _export_for_mujoco(self) -> dict:
        """Export data optimized for MuJoCo.

        Returns:
            MuJoCo-specific data structure.
        """
        # Implement MuJoCo-specific optimizations (future enhancement)
        return {
            "engine": "mujoco",
            "segments": self.segments,
            "parallel_chains": self.parallel_chains,
            "notes": "MuJoCo export - optimizations pending",
        }

    def _export_for_drake(self) -> dict:
        """Export data optimized for Drake.

        Returns:
            Drake-specific data structure.
        """
        # Implement Drake-specific optimizations (future enhancement)
        return {
            "engine": "drake",
            "segments": self.segments,
            "parallel_chains": self.parallel_chains,
            "notes": "Drake export - optimizations pending",
        }

    def _export_for_pinocchio(self) -> dict:
        """Export data optimized for Pinocchio.

        Returns:
            Pinocchio-specific data structure.
        """
        # Implement Pinocchio-specific optimizations (future enhancement)
        return {
            "engine": "pinocchio",
            "segments": self.segments,
            "parallel_chains": self.parallel_chains,
            "notes": "Pinocchio export - optimizations pending",
        }
