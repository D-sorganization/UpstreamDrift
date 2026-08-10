"""Bridge from the HMR2 sidecar's ``betas.json`` to the character builder.

Reads the SMPL shape coefficients estimated by the 4D-Humans / HMR 2.0
sidecar (see :mod:`src.tools.hmr2_sidecar.run_hmr2`) and produces a
:class:`~src.shared.python.humanoid_character_builder.core.body_parameters.BodyParameters`
whose ``smplx_betas`` field carries the measured betas. The SMPL-X mesh
generator then uses those betas verbatim instead of its heuristic
anthropometric mapping, closing the loop from monocular video to a
subject-shaped character model.

Only the JSON artifact crosses this boundary — no 4D-Humans code and no
SMPL model files are ever imported here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    GenderModel,
)

_GENDER_MAP = {
    "male": GenderModel.MALE,
    "female": GenderModel.FEMALE,
    "neutral": GenderModel.NEUTRAL,
}


def load_betas_json(path: str | os.PathLike[str]) -> tuple[list[float], str]:
    """Load and validate a sidecar ``betas.json`` artifact.

    Args:
        path: Path to a ``betas.json`` file following the sidecar output
            contract: ``{"betas": [<floats>], "gender": "<gender>"}``.

    Returns:
        ``(betas, gender)`` where ``betas`` is a non-empty list of
        finite floats and ``gender`` is one of ``male`` / ``female`` /
        ``neutral``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON is malformed or violates the contract.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"betas.json not found: {p}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"betas.json {p} is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(f"betas.json {p} must contain a JSON object")

    raw_betas = payload.get("betas")
    if not isinstance(raw_betas, list) or not raw_betas:
        raise ValueError(f"betas.json {p} must carry a non-empty 'betas' list")
    try:
        betas = [float(b) for b in raw_betas]
    except (TypeError, ValueError) as e:
        raise ValueError(f"betas.json {p} 'betas' entries must be numbers: {e}") from e
    if not all(math.isfinite(b) for b in betas):
        raise ValueError(f"betas.json {p} 'betas' entries must be finite")

    gender = str(payload.get("gender", "neutral")).lower()
    if gender not in _GENDER_MAP:
        raise ValueError(
            f"betas.json {p} has unsupported gender {gender!r}; "
            f"expected one of {sorted(_GENDER_MAP)}"
        )
    return betas, gender


def body_parameters_from_betas(
    betas_json_path: str | os.PathLike[str],
    height_m: float = 1.75,
    mass_kg: float = 75.0,
    name: str = "hmr2_subject",
) -> BodyParameters:
    """Build :class:`BodyParameters` from a sidecar ``betas.json``.

    The returned parameters carry the measured SMPL betas in
    ``smplx_betas`` (consumed verbatim by ``SMPLXMeshGenerator``) and
    the sidecar's gender estimate in ``gender_model``. Height and mass
    are not encoded in ``betas.json``, so they remain caller-supplied.

    Postcondition: ``result.smplx_betas`` equals the file's betas and
    ``result.validate()`` reports no errors for in-range height/mass.

    Raises:
        FileNotFoundError: If *betas_json_path* does not exist.
        ValueError: If the artifact violates the sidecar contract.
    """
    betas, gender = load_betas_json(betas_json_path)
    return BodyParameters(
        height_m=height_m,
        mass_kg=mass_kg,
        gender_model=_GENDER_MAP[gender],
        smplx_betas=betas,
        name=name,
        description=f"Shape from HMR2 sidecar artifact {Path(betas_json_path)}",
    )
