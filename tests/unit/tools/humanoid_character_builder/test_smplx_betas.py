"""Tests for measured SMPL betas plumbing (issue #8404).

``BodyParameters.smplx_betas`` (e.g. from the 4D-Humans/HMR2 sidecar's
``betas.json``) must flow verbatim into the SMPL-X mesh generator's
forward pass, while absence of betas preserves the legacy heuristic
anthropometric mapping. External dependencies (smplx, torch, trimesh)
are mocked: ``torch`` via ``patch.dict("sys.modules", ...)`` per repo
convention, and ``smplx``/``trimesh`` through the ``mesh_generator`` facade
attributes the generator reads (#4528).

Since #12706 these tests run against the single ``SMPLXMeshGenerator`` in
``_mesh_smplx``, the class the ``mesh_generator`` facade has always exported.
They used to exercise a second copy in ``_smplx_generator`` that production
never reached, so measured betas never got into a real build.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
)
from src.shared.python.humanoid_character_builder.generators import (
    mesh_generator,
)
from src.shared.python.humanoid_character_builder.generators._mesh_smplx import (
    SMPLXMeshGenerator,
)

pytestmark = pytest.mark.unit

_MEASURED_BETAS = [0.5, -1.2, 0.0, 0.3, -0.7, 1.1, 0.05, -0.4, 0.9, 0.2]


def _default_params(**overrides: Any) -> BodyParameters:
    kwargs: dict[str, Any] = {"height_m": 1.80, "mass_kg": 80.0}
    kwargs.update(overrides)
    return BodyParameters(**kwargs)


_to_betas = SMPLXMeshGenerator._convert_params_to_betas


class TestParamsToBetas:
    def test_measured_betas_returned_verbatim(self) -> None:
        params = _default_params(smplx_betas=list(_MEASURED_BETAS))
        np.testing.assert_array_equal(_to_betas(params), _MEASURED_BETAS)

    def test_legacy_mapping_unchanged_without_betas(self) -> None:
        """Absence of measured betas keeps the facade's heuristic mapping.

        The expected values are the ``_mesh_smplx`` mapping that the
        ``mesh_generator`` facade has always used (height about a 1.75 m mean,
        BMI about 22), not the unused ``_smplx_generator`` copy's mapping.
        """
        params = _default_params()
        betas = _to_betas(params)
        assert betas.shape == (SMPLXMeshGenerator.NUM_BETAS,)
        # betas[0] encodes height deviation: (1.80 - 1.75) / 0.2 = 0.25
        assert betas[0] == pytest.approx(0.25)
        # betas[1] encodes BMI deviation: (80 / 1.80**2 - 22) / 5
        assert betas[1] == pytest.approx((80.0 / 1.80**2 - 22.0) / 5.0)
        # The reserved last dimension stays zero
        assert betas[9] == 0.0

    def test_measured_betas_do_not_depend_on_height(self) -> None:
        short = _default_params(height_m=1.50, smplx_betas=list(_MEASURED_BETAS))
        tall = _default_params(height_m=2.00, smplx_betas=list(_MEASURED_BETAS))
        np.testing.assert_array_equal(_to_betas(short), _to_betas(tall))

    def test_none_params_rejected(self) -> None:
        with pytest.raises(ValueError, match="params must be provided"):
            _to_betas(None)  # type: ignore[arg-type]


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


@contextmanager
def _facade_patched(
    mock_smplx: MagicMock, mock_torch: MagicMock, mock_trimesh: MagicMock
) -> Iterator[None]:
    """Stand in smplx/trimesh where the generator reads them, and torch."""
    with (
        patch.object(mesh_generator, "SMPLX_AVAILABLE", True),
        patch.object(mesh_generator, "TRIMESH_AVAILABLE", True),
        patch.object(mesh_generator, "_smplx_module", mock_smplx),
        patch.object(mesh_generator, "_trimesh_module", mock_trimesh),
        patch.dict("sys.modules", {"torch": mock_torch}),
    ):
        yield


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

        with _facade_patched(mock_smplx, mock_torch, mock_trimesh):
            result = gen.generate(params, tmp_path / "out")

        # The betas tensor was built from the measured betas verbatim...
        np.testing.assert_array_equal(
            mock_torch.tensor.call_args[0][0], _MEASURED_BETAS
        )
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

        with _facade_patched(mock_smplx, mock_torch, mock_trimesh):
            gen.generate(params, tmp_path / "out")

        passed = mock_torch.tensor.call_args[0][0]
        np.testing.assert_array_equal(passed, _to_betas(params))
        assert not np.array_equal(passed, _MEASURED_BETAS)

    def test_missing_model_dir_fails_instead_of_passing_none(
        self, tmp_path: Path
    ) -> None:
        """A configured but absent model_dir is a failure, not str(None)."""
        mock_smplx, mock_torch, _mock_model, mock_trimesh = self._mock_modules()
        gen = SMPLXMeshGenerator(model_dir=tmp_path / "absent")

        with _facade_patched(mock_smplx, mock_torch, mock_trimesh):
            result = gen.generate(_default_params(), tmp_path / "out")

        assert result.success is False
        assert result.solver_status == "failure"
        assert "model files not found" in result.error_message
        mock_smplx.create.assert_not_called()

    def test_none_params_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="params must be provided"):
            SMPLXMeshGenerator(model_dir=tmp_path).generate(
                None,  # type: ignore[arg-type]
                tmp_path / "out",
            )
