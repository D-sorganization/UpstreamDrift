"""Compatibility re-export of the SMPL-X mesh generator.

There is exactly one ``SMPLXMeshGenerator`` implementation, and it lives in
``_mesh_smplx`` (issue #12706), which is what the public ``mesh_generator``
facade has always imported. This module used to hold a second, divergent copy
(``model_path``, ``_params_to_betas``) that the facade never used, so the
measured-betas support added to it for #8404 never reached production. That
support, the ``params is None`` guard and the model-directory lookup were ported
to ``_mesh_smplx``. The copy's silent fallback to primitive meshes on failure
was not ported: a failed SMPL-X build now reports ``solver_status="failure"``.

Import via ``mesh_generator`` (the public module).
"""

from __future__ import annotations

from ._mesh_smplx import SMPLXMeshGenerator

__all__ = ["SMPLXMeshGenerator"]
