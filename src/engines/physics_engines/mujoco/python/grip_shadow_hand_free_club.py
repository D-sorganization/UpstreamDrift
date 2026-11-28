"""MuJoCo free-club scene hook for Shadow hand grip visuals (#15122).

Drake, Pinocchio, and OpenSim shadow-hand grip visuals stay tracked on issue #15122.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import defusedxml.ElementTree as DefusedET

from src.shared.python.grip_spec import GripSpec
from src.shared.python.model_appearance.club_assembly import (
    GRIP_BUTT_RADIUS_M,
    ClubAssembly,
)
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import grip_hand_meshes
from src.shared.python.model_appearance.mesh_io import stl_bytes


def _write_hand_stl(mesh: Mesh, path: Path) -> None:
    """Write ``mesh`` as an STL file at ``path``."""
    path.write_bytes(stl_bytes(mesh, header="UpstreamDrift shadow grip visual"))


def build_free_club_shadow_hands_mjcf(
    grip_spec: GripSpec | None = None,
    *,
    club: ClubAssembly | None = None,
    mesh_dir: Path | None = None,
) -> tuple[str, Path, tuple[str, ...]]:
    """Return MJCF with a free club cylinder and welded Shadow-hand mesh geoms.

    The hand meshes are the Shadow backend, already placed in club coordinates,
    so each hand body is an identity child of the free club. Postcondition:
    compiled model includes ``lead_hand_vis`` and ``trail_hand_vis`` mesh geoms
    with ``contype=\"0\" conaffinity=\"0\" mass=\"0\"``.
    """
    spec = GripSpec.legacy() if grip_spec is None else grip_spec
    club = ClubAssembly("iron7", 0.95, 0.25, 0.0065) if club is None else club
    pose = spec.to_grip_pose()
    meshes = grip_hand_meshes(club, pose, grip_spec=spec, hand_backend="shadow")
    mesh_root = (
        mesh_dir
        if mesh_dir is not None
        else Path(tempfile.mkdtemp(prefix="ud-shadow-grip-"))
    )
    mesh_root.mkdir(parents=True, exist_ok=True)

    assets: list[tuple[str, Path]] = []
    geom_names: list[str] = []
    body_xml: list[str] = []
    for label in ("lead_hand", "trail_hand"):
        stl_path = mesh_root / f"{label}.stl"
        _write_hand_stl(meshes[label], stl_path)
        mesh_name = f"{label}_shadow_vis"
        assets.append((mesh_name, stl_path))
        gname = f"{label}_vis"
        geom_names.append(gname)
        body_xml.append(
            f'<body name="{label}_body" pos="0 0 0" quat="1 0 0 0">'
            f'<geom name="{gname}" type="mesh" mesh="{mesh_name}" '
            f'contype="0" conaffinity="0" mass="0" group="1" '
            f'rgba="0.82 0.70 0.62 1"/>'
            f"</body>"
        )

    asset_lines = "\n".join(
        f'<mesh name="{name}" file="{path.name}" scale="1 1 1"/>'
        for name, path in assets
    )
    club_len = float(club.length_m)
    grip_lo = float(club.length_m - club.grip_length_m)
    xml = f"""<mujoco model="free_club_shadow_hands">
  <compiler angle="radian" meshdir="{mesh_root.as_posix()}"/>
  <option gravity="0 0 0"/>
  <asset>
    {asset_lines}
  </asset>
  <worldbody>
    <body name="club" pos="0 0 0">
      <freejoint name="club_free"/>
      <geom name="grip" type="cylinder" fromto="0 {grip_lo} 0 0 {club_len} 0" size="{GRIP_BUTT_RADIUS_M}" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      {"".join(body_xml)}
    </body>
  </worldbody>
</mujoco>"""
    DefusedET.fromstring(xml)  # parse check
    return xml, mesh_root, tuple(geom_names)


__all__ = ["build_free_club_shadow_hands_mjcf"]
