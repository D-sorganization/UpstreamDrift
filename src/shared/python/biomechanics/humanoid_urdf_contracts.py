"""Humanoid URDF contract validator.

Salvaged from stale PR #2733 (see issue #2802). Provides a focused,
dependency-free validator that enforces structural invariants on a
humanoid URDF: required joints (hips/knees/ankles/shoulders/elbows),
bilateral (left/right) limb symmetry, positive link masses, and
physically valid inertia tensors.

The validator intentionally does NOT modify URDF files. It parses
them with :mod:`xml.etree.ElementTree` and surfaces problems as a
structured list plus an overall ok/not-ok result.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # stdlib retained for Element/SubElement
from dataclasses import dataclass, field
from pathlib import Path

import defusedxml.ElementTree as DefusedET  # noqa: S314  # Security: defusedxml prevents XML attacks

# Bilateral joint stems required by the humanoid contract. Each stem
# must be present as both ``left_<stem>`` and ``right_<stem>``.
REQUIRED_BILATERAL_JOINT_STEMS: tuple[str, ...] = (
    "shoulder",
    "elbow",
    "hip",
    "knee",
    "ankle",
)

# Minimum/maximum plausible link mass (kg). Anything outside this range
# is almost certainly a modeling mistake for a human-scale segment.
MIN_LINK_MASS_KG = 0.01
MAX_LINK_MASS_KG = 200.0

# Maximum bilateral mass asymmetry allowed between corresponding
# ``left_*`` / ``right_*`` links, as a fraction of the larger side.
MAX_BILATERAL_MASS_ASYMMETRY = 0.15


@dataclass(frozen=True)
class ContractViolation:
    """A single humanoid URDF contract violation."""

    category: str
    message: str


@dataclass
class ValidationReport:
    """Result of validating a humanoid URDF against its contract."""

    ok: bool
    violations: list[ContractViolation] = field(default_factory=list)

    def describe(self) -> str:
        """Return a human-readable multi-line description."""
        if self.ok:
            return "humanoid URDF contract: OK"
        lines = ["humanoid URDF contract: FAIL"]
        for v in self.violations:
            lines.append(f"  [{v.category}] {v.message}")
        return "\n".join(lines)


def _parse_root(source: str | Path | ET.Element) -> ET.Element:
    """Return the ``<robot>`` root element for ``source``."""
    if isinstance(source, ET.Element):
        return source
    try:
        path = Path(source)
    except (TypeError, ValueError):
        # Not path-like; treat as raw XML text.
        return DefusedET.fromstring(
            str(source)
        )  # nosec B314 — URDF XML from trusted file paths
    if path.exists():
        # For existing files, surface real parse/I/O failures directly.
        return DefusedET.parse(
            path
        ).getroot()  # nosec B314 — URDF XML from trusted file paths
    # Fall back to treating the argument as raw XML text.
    return DefusedET.fromstring(
        str(source)
    )  # nosec B314 — URDF XML from trusted file paths


def _collect_joint_names(root: ET.Element) -> set[str]:
    return {j.attrib["name"] for j in root.findall("joint") if "name" in j.attrib}


def _iter_link_masses(root: ET.Element) -> dict[str, float]:
    """Map of ``link name -> mass (kg)`` for links that declare inertial."""
    masses: dict[str, float] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name")
        mass_el = link.find("./inertial/mass")
        if name is None or mass_el is None:
            continue
        try:
            masses[name] = float(mass_el.attrib.get("value", "nan"))
        except ValueError:
            masses[name] = float("nan")
    return masses


def _iter_link_inertias(
    root: ET.Element,
) -> dict[str, dict[str, float]]:
    """Map of ``link name -> {ixx, iyy, izz, ixy, ixz, iyz}``."""
    out: dict[str, dict[str, float]] = {}
    for link in root.findall("link"):
        name = link.attrib.get("name")
        inertia_el = link.find("./inertial/inertia")
        if name is None or inertia_el is None:
            continue
        entry: dict[str, float] = {}
        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
            try:
                entry[key] = float(inertia_el.attrib.get(key, "nan"))
            except ValueError:
                entry[key] = float("nan")
        out[name] = entry
    return out


def _check_required_bilateral_joints(
    joint_names: set[str],
) -> list[ContractViolation]:
    violations: list[ContractViolation] = []
    for stem in REQUIRED_BILATERAL_JOINT_STEMS:
        for side in ("left", "right"):
            expected = f"{side}_{stem}"
            if expected not in joint_names:
                violations.append(
                    ContractViolation(
                        category="missing_joint",
                        message=(f"required humanoid joint '{expected}' is absent"),
                    )
                )
    return violations


def _check_masses(masses: dict[str, float]) -> list[ContractViolation]:
    violations: list[ContractViolation] = []
    for link, mass in masses.items():
        if mass != mass:  # NaN check
            violations.append(
                ContractViolation(
                    category="invalid_mass",
                    message=f"link '{link}' has unparseable mass",
                )
            )
            continue
        if mass <= 0.0:
            violations.append(
                ContractViolation(
                    category="invalid_mass",
                    message=(f"link '{link}' has non-positive mass {mass!r} kg"),
                )
            )
        elif mass < MIN_LINK_MASS_KG or mass > MAX_LINK_MASS_KG:
            violations.append(
                ContractViolation(
                    category="implausible_mass",
                    message=(
                        f"link '{link}' mass {mass!r} kg outside "
                        f"[{MIN_LINK_MASS_KG}, {MAX_LINK_MASS_KG}] kg"
                    ),
                )
            )
    return violations


def _check_inertias(
    inertias: dict[str, dict[str, float]],
) -> list[ContractViolation]:
    violations: list[ContractViolation] = []
    for link, tensor in inertias.items():
        for key in ("ixx", "iyy", "izz"):
            val = tensor.get(key, float("nan"))
            if val != val or val <= 0.0:
                violations.append(
                    ContractViolation(
                        category="invalid_inertia",
                        message=(
                            f"link '{link}' principal inertia {key}="
                            f"{val!r} must be positive"
                        ),
                    )
                )
        # Triangle inequalities on principal moments
        ixx = tensor.get("ixx", float("nan"))
        iyy = tensor.get("iyy", float("nan"))
        izz = tensor.get("izz", float("nan"))
        if all(v == v and v > 0.0 for v in (ixx, iyy, izz)) and (
            ixx + iyy < izz or iyy + izz < ixx or ixx + izz < iyy
        ):
            violations.append(
                ContractViolation(
                    category="invalid_inertia",
                    message=(
                        f"link '{link}' principal moments violate triangle inequality"
                    ),
                )
            )
    return violations


def _check_bilateral_symmetry(
    masses: dict[str, float],
) -> list[ContractViolation]:
    violations: list[ContractViolation] = []
    for name, mass in masses.items():
        if not name.startswith("left_"):
            continue
        mirror = "right_" + name[len("left_") :]
        if mirror not in masses:
            violations.append(
                ContractViolation(
                    category="asymmetric_limbs",
                    message=(f"link '{name}' has no mirror '{mirror}'"),
                )
            )
            continue
        right_mass = masses[mirror]
        if mass <= 0 or right_mass <= 0:
            continue  # already reported as invalid_mass
        larger = max(mass, right_mass)
        asym = abs(mass - right_mass) / larger
        if asym > MAX_BILATERAL_MASS_ASYMMETRY:
            violations.append(
                ContractViolation(
                    category="asymmetric_limbs",
                    message=(
                        f"bilateral mass asymmetry {asym:.2%} between "
                        f"'{name}' ({mass} kg) and '{mirror}' "
                        f"({right_mass} kg) exceeds "
                        f"{MAX_BILATERAL_MASS_ASYMMETRY:.0%}"
                    ),
                )
            )

    # Also catch right-side-only links that have no left_ counterpart.
    for name in masses:
        if not name.startswith("right_"):
            continue
        mirror = "left_" + name[len("right_") :]
        if mirror not in masses:
            violations.append(
                ContractViolation(
                    category="asymmetric_limbs",
                    message=(f"link '{name}' has no mirror '{mirror}'"),
                )
            )
    return violations


def validate_humanoid_urdf(
    source: str | Path | ET.Element,
) -> ValidationReport:
    """Validate ``source`` against the humanoid URDF contract.

    Parameters
    ----------
    source:
        Either a filesystem path, a raw URDF XML string, or a parsed
        ``<robot>`` :class:`xml.etree.ElementTree.Element`.
    """
    root = _parse_root(source)
    joint_names = _collect_joint_names(root)
    masses = _iter_link_masses(root)
    inertias = _iter_link_inertias(root)

    violations: list[ContractViolation] = []
    violations.extend(_check_required_bilateral_joints(joint_names))
    violations.extend(_check_masses(masses))
    violations.extend(_check_inertias(inertias))
    violations.extend(_check_bilateral_symmetry(masses))

    return ValidationReport(ok=not violations, violations=violations)


__all__ = [
    "ContractViolation",
    "MAX_BILATERAL_MASS_ASYMMETRY",
    "MAX_LINK_MASS_KG",
    "MIN_LINK_MASS_KG",
    "REQUIRED_BILATERAL_JOINT_STEMS",
    "ValidationReport",
    "validate_humanoid_urdf",
]
