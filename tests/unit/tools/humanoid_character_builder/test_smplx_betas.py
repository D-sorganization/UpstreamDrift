"""Tests for measured SMPL betas plumbing (issue #8404).

``BodyParameters.smplx_betas`` (e.g. from the 4D-Humans/HMR2 sidecar's
``betas.json``) must flow verbatim into the SMPL-X mesh generator's
forward pass, while absence of betas preserves the legacy heuristic
anthropometric mapping. External dependencies (smplx, torch, trimesh)
are mocked via ``patch.dict("sys.modules", ...)`` per repo convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)
from src.shared.python.humanoid_character_builder.generators._smplx_generator import (
    SMPLXMeshGenerator,
)

pytestmark = pytest.mark.unit

_MEASURED_BETAS = [0.5, -1.2, 0.0, 0.3, -0.7, 1.1, 0.05, -0.4, 0.9, 0.2]


def _default_params(**overrides: Any) -> BodyParameters:
    kwargs: dict[str, Any] = {"height_m": 1.80, "mass_kg": 80.0}
    kwargs.update(overrides)
    return BodyParameters(**kwargs)


class TestParamsToBetas:
    def test_measured_betas_returned_verbatim(self) -> None:
        gen = SMPLXMeshGenerator()
        params = _default_params(smplx_betas=list(_MEASURED_BETAS))
        assert gen._params_to_betas(params) == _MEASURED_BETAS

    def test_legacy_mapping_unchanged_without_betas(self) -> None:
        """Absence of measured betas keeps the heuristic mapping intact."""
        gen = SMPLXMeshGenerator()
        params = _default_params()
        betas = gen._params_to_betas(params)
        assert len(betas) == 10
        # betas[0] encodes height deviation: (1.80 - 1.7) / 0.2 = 0.5
        assert betas[0] == pytest.approx(0.5)
        # betas[1] encodes weight deviation: (80 - 70) / 20 * 0.5 = 0.25
        assert betas[1] == pytest.approx(0.25)
        # Unused trailing dimensions stay zero
        assert betas[7:] == [0.0, 0.0, 0.0]

    def test_measured_betas_do_not_depend_on_height(self) -> None:
        gen = SMPLXMeshGenerator()
        short = _default_params(height_m=1.50, smplx_betas=list(_MEASURED_BETAS))
        tall = _default_params(height_m=2.00, smplx_betas=list(_MEASURED_BETAS))
        assert gen._params_to_betas(short) == gen._params_to_betas(tall)


class _FakeTrimesh:
    """Minimal trimesh.Trimesh stand-in that records exports."""

    exported: list[str] = []

    def __init__(self, vertices: Any = None, faces: Any = None) -> None:
        self.vertices = vertices
        self.faces = faces

    def export(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
        _FakeTrimesh.exported.append(path)

    def submesh(self, face_masks: Any, append: bool = False) -> _FakeTrimesh:
        return self

    @property
    def convex_hull(self) -> _FakeTrimesh:
        return self


class TestGenerateForwardsBetas:
    """Mocked end-to-end: measured betas reach the smplx forward call."""

    N_VERTS = 40

    def _mock_modules(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        rng = np.random.default_rng(7)
        vertices = rng.standard_normal((1, self.N_VERTS, 3)).astype(np.float64)

        mock_output = MagicMock()
        mock_output.vertices.detach.return_value.cpu.return_value.numpy.return_value = (
            vertices
        )

        mock_model = MagicMock()
        mock_model.return_value = mock_output
        # Every face references valid vertex indices.
        mock_model.faces = rng.integers(0, self.N_VERTS, size=(60, 3))
        # All vertices weighted to joint 0 (pelvis) -> one valid segment.
        weights = np.zeros((self.N_VERTS, 22))
        weights[:, 0] = 1.0
        mock_model.lbs_weights.cpu.return_value.numpy.return_value = weights

        mock_smplx = MagicMock()
        mock_smplx.create.return_value = mock_model

        mock_torch = MagicMock()
        mock_trimesh = MagicMock()
        mock_trimesh.Trimesh = _FakeTrimesh
        return mock_smplx, mock_torch, mock_model, mock_trimesh

    def test_generate_passes_measured_betas_to_model(self, tmp_path: Path) -> None:
        mock_smplx, mock_torch, mock_model, mock_trimesh = self._mock_modules()
        _FakeTrimesh.exported = []
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        gen = SMPLXMeshGenerator(model_path=model_dir)
        params = _default_params(smplx_betas=list(_MEASURED_BETAS))

        with patch.dict(
            "sys.modules",
            {"smplx": mock_smplx, "torch": mock_torch, "trimesh": mock_trimesh},
        ):
            result = gen.generate(params, tmp_path / "out")

        # The betas tensor was built from the measured betas verbatim...
        assert mock_torch.tensor.call_args[0][0] == _MEASURED_BETAS
        # ...and that tensor is what the smplx forward pass received.
        expected_tensor = mock_torch.tensor.return_value.unsqueeze.return_value
        assert mock_model.call_args.kwargs["betas"] is expected_tensor

        assert result.success is True
        assert result.metadata["backend"] == "smplx"
        assert "pelvis" in result.mesh_paths

    def test_generate_without_betas_uses_heuristic_mapping(
        self, tmp_path: Path
    ) -> None:
        """Legacy behavior: no smplx_betas -> anthropometric mapping."""
        mock_smplx, mock_torch, _mock_model, mock_trimesh = self._mock_modules()
        _FakeTrimesh.exported = []
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        gen = SMPLXMeshGenerator(model_path=model_dir)
        params = _default_params()  # no smplx_betas

        with patch.dict(
            "sys.modules",
            {"smplx": mock_smplx, "torch": mock_torch, "trimesh": mock_trimesh},
        ):
            gen.generate(params, tmp_path / "out")

        passed = mock_torch.tensor.call_args[0][0]
        assert passed == gen._params_to_betas(params)
        assert passed != _MEASURED_BETAS
