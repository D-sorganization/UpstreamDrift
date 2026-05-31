"""JaxSim URDF-to-SDF conversion gate helpers.

The JaxSim stack can load SDF directly, but URDF input is delegated through
external sdformat tooling. This module keeps the #6648 gate explicit: find the
tool, convert the canonical URDF, compare inertial payloads, then let the
JaxSim API load the converted SDF.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import xml.etree.ElementTree as ET  # noqa: N817
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SDFORMAT_TOOL_NAMES = ("gz", "ign", "sdf")
INERTIA_FIELDS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")


@dataclass(frozen=True)
class SdformatTool:
    """Resolved sdformat CLI contract."""

    executable: Path
    mode: str

    def command_for(self, urdf_path: Path) -> list[str]:
        """Return the command that prints the converted SDF to stdout."""
        if self.mode in {"gz", "ign"}:
            return [str(self.executable), "sdf", "-p", str(urdf_path)]
        if self.mode == "sdf":
            return [str(self.executable), "-p", str(urdf_path)]
        msg = f"Unsupported sdformat tool mode: {self.mode!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class LinkInertial:
    """Mass and inertia payload for one link."""

    name: str
    mass: float
    inertia: dict[str, float]


@dataclass(frozen=True)
class InertialMismatch:
    """One link/field mismatch between URDF and SDF."""

    link: str
    field: str
    expected: float
    actual: float
    tolerance: float


def find_sdformat_tool(search_path: str | None = None) -> SdformatTool | None:
    """Find a supported sdformat executable on PATH."""
    for name in SDFORMAT_TOOL_NAMES:
        resolved = shutil.which(name, path=search_path)
        if resolved is not None:
            return SdformatTool(executable=Path(resolved), mode=name)
    return None


def require_sdformat_tool(search_path: str | None = None) -> SdformatTool:
    """Find sdformat tooling or raise with the BRICK setup requirement."""
    tool = find_sdformat_tool(search_path=search_path)
    if tool is not None:
        return tool
    searched = search_path if search_path is not None else "PATH"
    msg = (
        "Missing sdformat CLI. Install gz-tools/libsdformat on the Ubuntu "
        f"runner so one of {SDFORMAT_TOOL_NAMES!r} is available on {searched}."
    )
    raise FileNotFoundError(msg)


def convert_urdf_to_sdf(
    urdf_path: Path | str,
    sdf_path: Path | str,
    *,
    tool: SdformatTool | None = None,
    timeout_s: int = 60,
) -> Path:
    """Convert ``urdf_path`` to SDF using sdformat's print command."""
    urdf = Path(urdf_path)
    sdf = Path(sdf_path)
    if not urdf.is_file():
        msg = f"URDF file does not exist: {urdf}"
        raise FileNotFoundError(msg)

    resolved_tool = tool or require_sdformat_tool()
    result = subprocess.run(
        resolved_tool.command_for(urdf),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        msg = (
            f"sdformat conversion failed with exit code {result.returncode}: "
            f"{stderr or '<no stderr>'}"
        )
        raise RuntimeError(msg)

    converted = result.stdout.strip()
    if not converted:
        msg = "sdformat conversion produced no SDF on stdout"
        raise RuntimeError(msg)

    sdf.parent.mkdir(parents=True, exist_ok=True)
    sdf.write_text(converted + "\n", encoding="utf-8")
    return sdf


def read_urdf_inertials(urdf_path: Path | str) -> dict[str, LinkInertial]:
    """Read URDF link inertials keyed by link name."""
    root = ET.parse(urdf_path).getroot()
    inertials: dict[str, LinkInertial] = {}
    for link in root.findall("link"):
        name = _required_attr(link, "name")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(_required_attr(_required_child(inertial, "mass"), "value"))
        inertia_node = _required_child(inertial, "inertia")
        inertia = {
            field: float(_required_attr(inertia_node, field))
            for field in INERTIA_FIELDS
        }
        inertials[name] = LinkInertial(name=name, mass=mass, inertia=inertia)
    return inertials


def read_sdf_inertials(sdf_path: Path | str) -> dict[str, LinkInertial]:
    """Read SDF link inertials keyed by link name."""
    root = ET.parse(sdf_path).getroot()
    inertials: dict[str, LinkInertial] = {}
    for link in root.findall(".//link"):
        name = _required_attr(link, "name")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(_required_text(_required_child(inertial, "mass")))
        inertia_node = _required_child(inertial, "inertia")
        inertia = {
            field: float(_required_text(_required_child(inertia_node, field)))
            for field in INERTIA_FIELDS
        }
        inertials[name] = LinkInertial(name=name, mass=mass, inertia=inertia)
    return inertials


def compare_inertials(
    expected: dict[str, LinkInertial],
    actual: dict[str, LinkInertial],
    *,
    abs_tolerance: float = 1e-9,
    rel_tolerance: float = 1e-9,
) -> list[InertialMismatch]:
    """Return all mass/inertia mismatches outside tolerance."""
    mismatches: list[InertialMismatch] = []
    for link_name, expected_link in sorted(expected.items()):
        actual_link = actual.get(link_name)
        if actual_link is None:
            mismatches.append(
                InertialMismatch(
                    link=link_name,
                    field="<missing link>",
                    expected=1.0,
                    actual=0.0,
                    tolerance=0.0,
                )
            )
            continue

        for field, expected_value, actual_value in _iter_numeric_fields(
            expected_link, actual_link
        ):
            tolerance = max(abs_tolerance, abs(expected_value) * rel_tolerance)
            if not math.isclose(
                expected_value,
                actual_value,
                rel_tol=rel_tolerance,
                abs_tol=abs_tolerance,
            ):
                mismatches.append(
                    InertialMismatch(
                        link=link_name,
                        field=field,
                        expected=expected_value,
                        actual=actual_value,
                        tolerance=tolerance,
                    )
                )
    for link_name in sorted(set(actual) - set(expected)):
        mismatches.append(
            InertialMismatch(
                link=link_name,
                field="<unexpected link>",
                expected=0.0,
                actual=1.0,
                tolerance=0.0,
            )
        )
    return mismatches


def assert_inertials_round_trip(
    urdf_path: Path | str,
    sdf_path: Path | str,
    *,
    abs_tolerance: float = 1e-9,
    rel_tolerance: float = 1e-9,
) -> None:
    """Raise if SDF mass/inertia values drift from the source URDF."""
    mismatches = compare_inertials(
        read_urdf_inertials(urdf_path),
        read_sdf_inertials(sdf_path),
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    if mismatches:
        preview = "; ".join(
            f"{m.link}.{m.field}: expected {m.expected:g}, got {m.actual:g}"
            for m in mismatches[:5]
        )
        msg = f"SDF inertial round-trip mismatch ({len(mismatches)}): {preview}"
        raise AssertionError(msg)


def build_jaxsim_model_from_sdf(sdf_path: Path | str, *, time_step: float = 0.001):
    """Build a JaxSim model from converted SDF."""
    import jaxsim.api as js  # noqa: PLC0415

    return js.model.JaxSimModel.build_from_model_description(
        Path(sdf_path),
        is_urdf=False,
        time_step=time_step,
    )


def _iter_numeric_fields(
    expected: LinkInertial, actual: LinkInertial
) -> Iterable[tuple[str, float, float]]:
    yield "mass", expected.mass, actual.mass
    for field in INERTIA_FIELDS:
        yield field, expected.inertia[field], actual.inertia[field]


def _required_attr(node: ET.Element, name: str) -> str:
    value = node.get(name)
    if value is None:
        msg = f"Missing required attribute {name!r} on <{node.tag}>"
        raise ValueError(msg)
    return value


def _required_child(node: ET.Element, name: str) -> ET.Element:
    child = node.find(name)
    if child is None:
        msg = f"Missing required child <{name}> under <{node.tag}>"
        raise ValueError(msg)
    return child


def _required_text(node: ET.Element) -> str:
    value = node.text
    if value is None or not value.strip():
        msg = f"Missing required text in <{node.tag}>"
        raise ValueError(msg)
    return value.strip()


__all__ = [
    "INERTIA_FIELDS",
    "SDFORMAT_TOOL_NAMES",
    "InertialMismatch",
    "LinkInertial",
    "SdformatTool",
    "assert_inertials_round_trip",
    "build_jaxsim_model_from_sdf",
    "compare_inertials",
    "convert_urdf_to_sdf",
    "find_sdformat_tool",
    "read_sdf_inertials",
    "read_urdf_inertials",
    "require_sdformat_tool",
]
