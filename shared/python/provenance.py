"""Provenance tracking for scientific reproducibility.

This module provides utilities to track and record the provenance of scientific
analyses, ensuring reproducibility and auditability.

Addresses Assessment C-009: Results lack metadata and provenance information.

DESIGN PRINCIPLES:
------------------
1. **Automatic Capture**: Provenance collected with minimal user intervention
2. **Immutability**: ProvenanceInfo is a frozen dataclass
3. **Standardized Format**: ISO 8601 timestamps, semantic versioning
4. **Git Integration**: Captures code version (SHA) for exact reproducibility
5. **Lightweight**: Minimal performance overhead

USAGE:
------
```python
from shared.python.provenance import ProvenanceInfo, add_provenance_header

# Automatic capture
provenance = ProvenanceInfo.capture()

# Add to CSV export
with open('results.csv', 'w') as f:
    add_provenance_header(f, provenance)
    # ... write data
```
"""

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np


@dataclass(frozen=True)
class ProvenanceInfo:
    """Immutable provenance record for scientific analyses.

    Captures all information needed to reproduce an analysis:
    - When: ISO 8601 UTC timestamp
    - What: Software version and git commit
    - How: Model file hash and analysis parameters
    - Where: Environment information
    """

    # Temporal metadata
    timestamp_utc: str  # ISO 8601 format: "2026-01-05T21:00:00Z"
    timestamp_local: str  # ISO 8601 with timezone: "2026-01-05T13:00:00-08:00"

    # Software version
    software_name: str = "golf-modeling-suite"
    software_version: str = "1.0.0-beta"  # Semantic versioning
    git_commit_sha: str | None = None  # Full 40-char SHA if available
    git_branch: str | None = None
    git_is_dirty: bool = False  # True if uncommitted changes exist

    # Model metadata
    model_file_path: str | None = None
    model_file_hash: str | None = None  # SHA256 of model file

    # Analysis parameters (optional, user-provided)
    parameters: dict[str, Any] = field(default_factory=dict)

    # Environment
    python_version: str | None = None
    numpy_version: str | None = None
    mujoco_version: str | None = None

    @classmethod
    def capture(
        cls,
        model_path: Path | str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "ProvenanceInfo":
        """Automaticall
        y capture provenance information.

                Args:
                    model_path: Optional path to model file (URDF, MJCF, etc.)
                    parameters: Optional analysis parameters to record

                Returns:
                    ProvenanceInfo with automatically captured metadata

                Example:
                    >>> provenance = ProvenanceInfo.capture(
                    ...     model_path="models/humanoid.xml",
                    ...     parameters={"dt": 0.001, "integrator": "RK4"}
                    ... )
        """
        now_utc = datetime.now(timezone.utc)  # noqa: UP017 (mypy compatibility)
        now_local = datetime.now().astimezone()

        # Git information (best effort)
        git_sha, git_branch, git_dirty = cls._get_git_info()

        # Model file hash
        model_hash = None
        model_path_str = None
        if model_path is not None:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                model_hash = cls._hash_file(model_path_obj)
                model_path_str = str(model_path_obj)

        # Environment versions
        import sys

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        numpy_version = np.__version__

        # MuJoCo version (if available)
        mujoco_version = None
        try:
            import mujoco

            mujoco_version = mujoco.__version__
        except ImportError:
            pass

        return cls(
            timestamp_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            timestamp_local=now_local.isoformat(),
            git_commit_sha=git_sha,
            git_branch=git_branch,
            git_is_dirty=git_dirty,
            model_file_path=model_path_str,
            model_file_hash=model_hash,
            parameters=parameters or {},
            python_version=python_version,
            numpy_version=numpy_version,
            mujoco_version=mujoco_version,
        )

    @staticmethod
    def _get_git_info() -> tuple[str | None, str | None, bool]:
        """Get git commit SHA, branch, and dirty status.

        Returns:
            Tuple of (sha, branch, is_dirty)
        """
        try:
            # Get commit SHA
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

            # Get branch name
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()

            # Check if working directory is clean
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            is_dirty = len(status) > 0

            return sha, branch, is_dirty

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Not a git repository or git not installed
            return None, None, False

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA256 hash of file.

        Args:
            path: Path to file

        Returns:
            Hex-encoded SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def to_header_lines(self) -> list[str]:
        """Generate human-readable header lines for file export.

        Returns:
            List of comment lines (with '#' prefix)

        Example:
            >>> provenance = ProvenanceInfo.capture()
            >>> for line in provenance.to_header_lines():
            ...     print(line)
            # Exported by golf-modeling-suite v1.0.0-beta
            # Git commit: abc123... (branch: main)
            # Generated: 2026-01-05T21:00:00Z
            ...
        """
        lines = []

        # Software and version
        version_str = f"{self.software_name} v{self.software_version}"
        if self.git_commit_sha:
            git_info = f"{self.git_commit_sha[:8]}"
            if self.git_branch:
                git_info += f" (branch: {self.git_branch})"
            if self.git_is_dirty:
                git_info += " [UNCOMMITTED CHANGES]"
            version_str += f" (git: {git_info})"

        lines.append(f"# Exported by {version_str}")

        # Timestamp
        lines.append(f"# Generated: {self.timestamp_utc} (UTC)")
        lines.append(f"# Local time: {self.timestamp_local}")

        # Model information
        if self.model_file_path:
            lines.append(f"# Model file: {self.model_file_path}")
            if self.model_file_hash:
                lines.append(f"# Model hash (SHA256): {self.model_file_hash}")

        # Analysis parameters
        if self.parameters:
            lines.append("# Analysis parameters:")
            for key, value in sorted(self.parameters.items()):
                lines.append(f"#   {key}: {value}")

        # Environment
        lines.append("# Environment:")
        lines.append(f"#   Python: {self.python_version}")
        lines.append(f"#   NumPy: {self.numpy_version}")
        if self.mujoco_version:
            lines.append(f"#   MuJoCo: {self.mujoco_version}")

        # Reproducibility warning if git is dirty
        if self.git_is_dirty:
            lines.append("#")
            lines.append("# WARNING: Exported with uncommitted code changes!")
            lines.append(
                "# Results may not be exactly reproducible from git SHA alone."
            )

        return lines


def add_provenance_header_file(file: TextIO, provenance: ProvenanceInfo) -> None:
    """Add provenance header to an open text file.

    Args:
        file: Open text file (CSV, etc.)
        provenance: ProvenanceInfo to write

    Example:
        >>> provenance = ProvenanceInfo.capture()
        >>> with open('results.csv', 'w') as f:
        ...     add_provenance_header(f, provenance)
        ...     f.write("time,position,velocity\\n")
        ...     # ... write data
    """
    for line in provenance.to_header_lines():
        file.write(line + "\n")
    file.write("#\n")  # Blank comment line separator


def add_provenance_to_csv(
    filepath: Path | str,
    provenance: ProvenanceInfo | None = None,
    model_path: Path | str | None = None,
    parameters: dict[str, Any] | None = None,
) -> ProvenanceInfo:
    """Prepend provenance header to existing CSV file.

    Args:
        filepath: Path to CSV file (will be rewritten)
        provenance: Optional pre-captured provenance (if None, auto-capture)
        model_path: Optional model path (used if provenance is None)
        parameters: Optional parameters (used if provenance is None)

    Returns:
        ProvenanceInfo that was added

    Example:
        >>> # After generating results.csv
        >>> add_provenance_to_csv('results.csv', parameters={"dt": 0.001})
    """
    # Capture provenance if not provided
    if provenance is None:
        provenance = ProvenanceInfo.capture(
            model_path=model_path, parameters=parameters
        )

    # Read existing file
    filepath_obj = Path(filepath)
    original_content = filepath_obj.read_text()

    # Write provenance + original content
    with open(filepath_obj, "w") as f:
        add_provenance_header_file(f, provenance)
        f.write(original_content)

    return provenance


# Export public API
__all__ = [
    "ProvenanceInfo",
    "add_provenance_header_file",
    "add_provenance_to_csv",
]
