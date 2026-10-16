"""Compatibility re-export of the MakeHuman mesh generator.

There is exactly one ``MakeHumanMeshGenerator`` implementation, and it lives in
``_mesh_makehuman`` (issue #12698). This module used to hold a second, divergent
copy of the class -- including a second ``_convert_params_to_makehuman`` -- that
the backlog consolidation in #8322 wired into ``mesh_generator``, which broke the
scripted-mode API (``_build_mh_script``, ``_parse_obj_file``,
``MH_VERTEX_GROUP_MAP``) that the unit tests and callers rely on.

Import via ``mesh_generator`` (the public module).
"""

from __future__ import annotations

from ._mesh_makehuman import MakeHumanMeshGenerator

__all__ = ["MakeHumanMeshGenerator"]
