"""MuJoCo free-club scene hook for Shadow hand grip visuals (#15122).

Drake, Pinocchio, and OpenSim shadow-hand grip visuals: TODO (track on epic #15122).
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
from src.shared.python.model_appearance.grip_hands import (
    grip_hand_frame,
    grip_hand_meshes,
)
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.mesh_io import stl_bytes


def _write_hand_stl(mesh: Mesh, path: Path) -> None:
    path.write_bytes(stl_bytes(mesh, header="UpstreamDrift shadow grip visual"))


def build_free_club_shadow_hands_mjcf(
    grip_spec: GripSpec | None = None,
    *,
    club: ClubAssembly | None = None,
    mesh_dir: Path | None = None,
) -> tuple[str, Path, tuple[str, ...]]:
    """Return MJCF with a free club cylinder and welded Shadow-hand mesh geoms.

    Postcondition: compiled model includes ``lead_hand_vis`` and ``trail_hand_vis``
    mesh geoms with ``contype=\"0\" conaffinity=\"0\" mass=\"0\"``.
    """
    spec = GripSpec.legacy() if grip_spec is None else grip_spec
    club = (
        ClubAssembly("iron7", 0.95, 0.25, 0.0065)
        if club is None
        else club
    )
    pose = spec.to_grip_pose()
    meshes = grip_hand_meshes(club, pose, grip_spec=spec)
    mesh_root = mesh_dir if mesh_dir is not None else Path(tempfile.mkdtemp(prefix="ud-shadow-grip-"))
    mesh_root.mkdir(parents=True, exist_ok=True)

    assets: list[tuple[str, Path]] = []
    geom_names: list[str] = []
    body_xml: list[str] = []
    for hand, label in ((Hand.LEAD, "lead_hand"), (Hand.TRAIL, "trail_hand")):
        stl_path = mesh_root / f"{label}.stl"
        _write_hand_stl(meshes[label], stl_path)
        frame = grip_hand_frame(club, hand, pose)
        pos = " ".join(f"{v:.17g}" for v in frame[:3, 3])
        quat = _matrix_to_quat(frame[:3, :3])
        mesh_name = f"{label}_shadow_vis"
        assets.append((mesh_name, stl_path))
        gname = f"{label}_vis"
        geom_names.append(gname)
        body_xml.append(
            f'<body name="{label}_body" pos="{pos}" quat="{quat}">'
            f'<geom name="{gname}" type="mesh" mesh="{mesh_name}" '
            f'contype="0" conaffinity="0" mass="0" group="1" '
            f'rgba="0.82 0.70 0.62 1"/>'
            f"</body>"
        )

    asset_lines = "\n".join(
        f'<mesh name="{name}" file="{path.name}" scale="1 1 1"/>' for name, path in assets
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
    </body>
    {''.join(body_xml)}
  </worldbody>
</mujoco>"""
    DefusedET.fromstring(xml)  # parse check
    return xml, mesh_root, tuple(geom_names)


def _matrix_to_quat(rot) -> str:
    import numpy as np
    from scipy.spatial.transform import Rotation as SciRot

    q = SciRot.from_matrix(np.asarray(rot, float)).as_quat()  # x,y,z,w
    return " ".join(f"{v:.17g}" for v in (q[3], q[0], q[1], q[2]))


__all__ = ["build_free_club_shadow_hands_mjcf"]
