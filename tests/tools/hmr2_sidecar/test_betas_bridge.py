"""Tests for the HMR2 sidecar betas.json -> BodyParameters bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.humanoid_character_builder.core.body_parameters import (
    BodyParameters,
    GenderModel,
)
from src.tools.hmr2_sidecar.betas_bridge import (
    body_parameters_from_betas,
    load_betas_json,
)
from src.tools.hmr2_sidecar.run_hmr2 import NUM_BETAS, _write_stub_artifacts

_BETAS = [0.5, -1.2, 0.0, 0.3, -0.7, 1.1, 0.05, -0.4, 0.9, 0.2]


def _write_betas(
    tmp_path: Path, betas: object = None, gender: object = "neutral"
) -> Path:
    path = tmp_path / "betas.json"
    payload = {"betas": _BETAS if betas is None else betas, "gender": gender}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadBetasJson:
    def test_valid_file(self, tmp_path: Path) -> None:
        betas, gender = load_betas_json(_write_betas(tmp_path))
        assert betas == _BETAS
        assert gender == "neutral"

    def test_gender_case_insensitive(self, tmp_path: Path) -> None:
        _, gender = load_betas_json(_write_betas(tmp_path, gender="Female"))
        assert gender == "female"

    def test_missing_gender_defaults_neutral(self, tmp_path: Path) -> None:
        path = tmp_path / "betas.json"
        path.write_text(json.dumps({"betas": _BETAS}), encoding="utf-8")
        _, gender = load_betas_json(path)
        assert gender == "neutral"

    def test_stub_artifact_roundtrip(self, tmp_path: Path) -> None:
        """The sidecar's own stub betas.json satisfies the bridge contract."""
        _, betas_path, _ = _write_stub_artifacts(tmp_path)
        betas, gender = load_betas_json(betas_path)
        assert betas == [0.0] * NUM_BETAS
        assert gender == "neutral"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_betas_json(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "betas.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_betas_json(path)

    def test_non_object_payload_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "betas.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_betas_json(path)

    def test_missing_betas_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "betas.json"
        path.write_text(json.dumps({"gender": "neutral"}), encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty 'betas' list"):
            load_betas_json(path)

    def test_empty_betas_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty 'betas' list"):
            load_betas_json(_write_betas(tmp_path, betas=[]))

    def test_non_numeric_betas_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must be numbers"):
            load_betas_json(_write_betas(tmp_path, betas=["a", "b"]))

    def test_non_finite_betas_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="finite"):
            load_betas_json(_write_betas(tmp_path, betas=[1.0, float("nan")]))

    def test_unknown_gender_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unsupported gender"):
            load_betas_json(_write_betas(tmp_path, gender="robot"))


class TestBodyParametersFromBetas:
    def test_betas_carried_into_body_parameters(self, tmp_path: Path) -> None:
        params = body_parameters_from_betas(_write_betas(tmp_path))
        assert isinstance(params, BodyParameters)
        assert params.smplx_betas == _BETAS
        assert params.gender_model is GenderModel.NEUTRAL
        assert params.validate() == []

    @pytest.mark.parametrize(
        ("gender", "expected"),
        [
            ("male", GenderModel.MALE),
            ("female", GenderModel.FEMALE),
            ("neutral", GenderModel.NEUTRAL),
        ],
    )
    def test_gender_mapping(
        self, tmp_path: Path, gender: str, expected: GenderModel
    ) -> None:
        params = body_parameters_from_betas(_write_betas(tmp_path, gender=gender))
        assert params.gender_model is expected

    def test_height_and_mass_passthrough(self, tmp_path: Path) -> None:
        params = body_parameters_from_betas(
            _write_betas(tmp_path), height_m=1.62, mass_kg=58.0, name="subject_a"
        )
        assert params.height_m == 1.62
        assert params.mass_kg == 58.0
        assert params.name == "subject_a"

    def test_stub_artifact_end_to_end(self, tmp_path: Path) -> None:
        """Sidecar stub output -> BodyParameters (the full loop closure)."""
        _, betas_path, _ = _write_stub_artifacts(tmp_path)
        params = body_parameters_from_betas(betas_path)
        assert params.smplx_betas == [0.0] * NUM_BETAS

    def test_serialization_roundtrip_keeps_betas(self, tmp_path: Path) -> None:
        params = body_parameters_from_betas(_write_betas(tmp_path))
        restored = BodyParameters.from_dict(params.to_dict())
        assert restored.smplx_betas == _BETAS

    def test_invalid_artifact_propagates(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            body_parameters_from_betas(_write_betas(tmp_path, betas=[]))


class TestBodyParametersBetasContract:
    """Constructor-level DbC for the new smplx_betas field."""

    def test_default_is_none(self) -> None:
        assert BodyParameters().smplx_betas is None

    def test_tuple_coerced_to_float_list(self) -> None:
        params = BodyParameters(smplx_betas=(1, 2, 3))
        assert params.smplx_betas == [1.0, 2.0, 3.0]

    def test_empty_sequence_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            BodyParameters(smplx_betas=[])

    def test_non_sequence_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            BodyParameters(smplx_betas="0.1,0.2")  # type: ignore[arg-type]

    def test_non_numeric_entries_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            BodyParameters(smplx_betas=[0.1, "x"])  # type: ignore[list-item]

    def test_non_finite_entries_rejected(self) -> None:
        with pytest.raises(PreconditionError):
            BodyParameters(smplx_betas=[0.1, float("inf")])
