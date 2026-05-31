from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from tools.offline_validation.nimble_gradient_oracle import (
    GradientOracleUnavailable,
    GradientTolerance,
    NIMBLEPHYSICS_PIN,
    NimbleGradientOracleRequest,
    TorchAutogradNimbleBackend,
    compare_nimble_gradient,
)


class FakeQuadraticBackend:
    def gradient(self, loss_fn, coordinates: np.ndarray) -> np.ndarray:
        del loss_fn
        return 2.0 * coordinates + np.array([0.5, -1.0, 0.25])


class MissingNimbleBackend:
    def gradient(self, loss_fn, coordinates: np.ndarray) -> np.ndarray:
        del loss_fn, coordinates
        raise GradientOracleUnavailable("nimblephysics is not installed")


def _toy_loss(_nimble, x):
    return x


def test_oracle_request_normalizes_vectors_and_metadata() -> None:
    request = NimbleGradientOracleRequest(
        model_name="single-link-toy",
        coordinates=[1, 2, 3],
        candidate_gradient=[2.5, 3.0, 6.25],
        nimble_loss=_toy_loss,
        metadata={"source": "unit"},
    )

    assert request.coordinates.dtype == np.float64
    assert request.candidate_gradient.dtype == np.float64
    assert request.metadata == {"source": "unit"}


def test_oracle_response_passes_for_matching_toy_gradient() -> None:
    coordinates = np.array([1.0, 2.0, -0.5])
    request = NimbleGradientOracleRequest(
        model_name="single-link-toy",
        coordinates=coordinates,
        candidate_gradient=2.0 * coordinates + np.array([0.5, -1.0, 0.25]),
        nimble_loss=_toy_loss,
    )

    response = compare_nimble_gradient(request, backend=FakeQuadraticBackend())

    assert response.status == "passed"
    assert response.reason == "gradient agreement within tolerance"
    assert response.nimble_pin == NIMBLEPHYSICS_PIN
    np.testing.assert_allclose(response.oracle_gradient, request.candidate_gradient)


def test_oracle_response_reports_gradient_disagreement() -> None:
    request = NimbleGradientOracleRequest(
        model_name="single-link-toy",
        coordinates=np.array([1.0, 2.0, -0.5]),
        candidate_gradient=np.array([10.0, 10.0, 10.0]),
        nimble_loss=_toy_loss,
        tolerance=GradientTolerance(rtol=1.0e-8, atol=1.0e-10),
    )

    response = compare_nimble_gradient(request, backend=FakeQuadraticBackend())

    assert response.status == "failed"
    assert response.max_abs_error is not None
    assert response.max_abs_error > 0.0
    assert response.max_rel_error is not None
    assert response.max_rel_error > 0.0


def test_oracle_skips_when_nimble_is_unavailable() -> None:
    request = NimbleGradientOracleRequest(
        model_name="single-link-toy",
        coordinates=np.array([0.0]),
        candidate_gradient=np.array([0.0]),
        nimble_loss=_toy_loss,
    )

    response = compare_nimble_gradient(request, backend=MissingNimbleBackend())

    assert response.status == "skipped"
    assert response.oracle_gradient is None
    assert "nimblephysics" in response.reason


def test_require_available_raises_when_nimble_is_unavailable() -> None:
    request = NimbleGradientOracleRequest(
        model_name="single-link-toy",
        coordinates=np.array([0.0]),
        candidate_gradient=np.array([0.0]),
        nimble_loss=_toy_loss,
    )

    with pytest.raises(GradientOracleUnavailable, match="nimblephysics"):
        compare_nimble_gradient(
            request,
            backend=MissingNimbleBackend(),
            require_available=True,
        )


def test_invalid_request_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        NimbleGradientOracleRequest(
            model_name="bad-shape",
            coordinates=np.array([0.0, 1.0]),
            candidate_gradient=np.array([0.0]),
            nimble_loss=_toy_loss,
        )


def test_runtime_src_does_not_import_nimble() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    offenders: list[str] = []
    for path in (repo_root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in (
            "import nimblephysics",
            "from nimblephysics",
            "import nimble ",
            "from nimble ",
        ):
            if needle in text:
                offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []


def test_pyproject_declares_pinned_nimble_oracle_extra() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]

    assert NIMBLEPHYSICS_PIN in extras["nimble-oracle"]
    assert "upstream-drift[nimble-oracle]" not in extras["all-engines"]


@pytest.mark.requires_nimble
def test_live_torch_autograd_backend_for_scalar_toy_model() -> None:
    pytest.importorskip("nimblephysics")
    pytest.importorskip("torch")
    backend = TorchAutogradNimbleBackend()
    coordinates = np.array([0.25, -0.5, 0.75])

    def toy_loss(_nimble, x):
        return (x * x).sum()

    request = NimbleGradientOracleRequest(
        model_name="torch-scalar-toy",
        coordinates=coordinates,
        candidate_gradient=2.0 * coordinates,
        nimble_loss=toy_loss,
    )

    response = compare_nimble_gradient(request, backend=backend)

    assert response.status == "passed"
