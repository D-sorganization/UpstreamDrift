"""Config hashing for BunkerShot3D run provenance (issue #8617, finding B18).

Contract: RFC 8785 (JCS) style canonical JSON with ``allow_nan=False``, hashed
with SHA-256. Two digests are emitted:

* ``config_hash`` -- over the whole configuration (the exact run).
* ``physics_hash`` -- over the configuration minus fields that cannot change
  the physics (output paths, log level, thread count, seed, recording rate),
  so "same experiment at a different seed" is a machine-answerable question.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.config import BunkerShotConfig
from bunkershot3d.provenance import (
    PHYSICS_EXCLUDED_FIELDS,
    FieldClass,
    canonical_json,
    classify_field,
    config_hash,
    leaf_field_paths,
    physics_hash,
)

pytestmark = pytest.mark.unit


def _config_dict() -> dict[str, object]:
    """Return a valid BunkerShotConfig payload plus run-control fields."""
    return {
        "bunker_bed": {
            "domain": {"length_x": 0.4, "width_y": 0.3, "depth_z": 0.1},
            "boundary": "fixed",
        },
        "grain_population": {
            "count": 1000,
            "diameter_mean": 0.0004,
            "diameter_sigma_log": 0.2,
            "density": 2650.0,
            "coarse_graining_factor": 1.0,
        },
        "contact_model": {
            "friction_coefficient": 0.5,
            "restitution_coefficient": 0.3,
            "youngs_modulus": 1.0e7,
            "poisson_ratio": 0.25,
        },
        "clubhead": {
            "loft_deg": 56.0,
            "bounce_deg": 10.0,
            "width": 0.08,
            "height": 0.05,
            "mass": 0.3,
        },
        "trajectory": {"file": "traj.csv", "duration": 0.05},
        "output": {"downsample_grains": 1, "rate_hz": 2000.0},
    }


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_has_no_insignificant_whitespace() -> None:
    assert canonical_json({"a": [1, 2], "b": {"c": 3}}) == '{"a":[1,2],"b":{"c":3}}'


def test_canonical_json_serialises_integral_floats_as_integers() -> None:
    """JCS numbers follow ECMAScript ``Number::toString``: 1.0 serialises as 1."""
    assert canonical_json({"x": 1.0}) == '{"x":1}'


def test_canonical_json_normalises_exponent_form() -> None:
    assert canonical_json({"x": 1e-7}) == '{"x":1e-7}'


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError, match="not finite"):
        canonical_json({"x": math.nan})


def test_canonical_json_rejects_infinity() -> None:
    with pytest.raises(ValueError, match="not finite"):
        canonical_json({"x": math.inf})


def test_canonical_json_rejects_unsupported_type() -> None:
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


# ---------------------------------------------------------------------------
# Field classification
# ---------------------------------------------------------------------------


def test_exclusion_list_is_a_frozenset() -> None:
    assert isinstance(PHYSICS_EXCLUDED_FIELDS, frozenset)


def test_every_schema_field_is_classified() -> None:
    """Every leaf of the config schema must have an explicit classification.

    The expectation map is exhaustive on purpose: adding a config field makes
    this test fail until someone decides whether it changes the physics.
    """
    expected: dict[str, FieldClass] = {
        "bunker_bed.domain.length_x": FieldClass.PHYSICS,
        "bunker_bed.domain.width_y": FieldClass.PHYSICS,
        "bunker_bed.domain.depth_z": FieldClass.PHYSICS,
        "bunker_bed.boundary": FieldClass.PHYSICS,
        "grain_population.count": FieldClass.PHYSICS,
        "grain_population.diameter_mean": FieldClass.PHYSICS,
        "grain_population.diameter_sigma_log": FieldClass.PHYSICS,
        "grain_population.density": FieldClass.PHYSICS,
        "grain_population.coarse_graining_factor": FieldClass.PHYSICS,
        "contact_model.friction_coefficient": FieldClass.PHYSICS,
        "contact_model.restitution_coefficient": FieldClass.PHYSICS,
        "contact_model.youngs_modulus": FieldClass.PHYSICS,
        "contact_model.poisson_ratio": FieldClass.PHYSICS,
        "clubhead.loft_deg": FieldClass.PHYSICS,
        "clubhead.bounce_deg": FieldClass.PHYSICS,
        "clubhead.width": FieldClass.PHYSICS,
        "clubhead.height": FieldClass.PHYSICS,
        "clubhead.mass": FieldClass.PHYSICS,
        "trajectory.file": FieldClass.PHYSICS,
        "trajectory.duration": FieldClass.PHYSICS,
        "output.downsample_grains": FieldClass.EXCLUDED,
        "output.rate_hz": FieldClass.EXCLUDED,
    }
    config = BunkerShotConfig(**_config_dict())
    paths = set(leaf_field_paths(config))

    assert paths == set(expected), "config schema drifted; classify the new fields"
    for path, field_class in expected.items():
        assert classify_field(path) is field_class, path


@pytest.mark.parametrize(
    "path",
    ["seed", "solver.seed", "run.log_level", "run.n_threads", "io.output_dir"],
)
def test_run_control_fields_are_excluded_from_physics(path: str) -> None:
    assert classify_field(path) is FieldClass.EXCLUDED


# ---------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------


def test_hashes_are_hex_sha256() -> None:
    digest = config_hash(_config_dict())
    assert len(digest) == 64
    int(digest, 16)  # raises if not hex


def test_config_hash_accepts_pydantic_model_and_mapping_alike() -> None:
    payload = _config_dict()
    model = BunkerShotConfig(**payload)
    assert config_hash(model) == config_hash(payload)


def test_same_config_same_seed_gives_identical_config_hash() -> None:
    left = {**_config_dict(), "seed": 12345}
    right = {**_config_dict(), "seed": 12345}
    assert config_hash(left) == config_hash(right)


def test_different_seed_changes_config_hash_but_not_physics_hash() -> None:
    left = {**_config_dict(), "seed": 1}
    right = {**_config_dict(), "seed": 2}
    assert config_hash(left) != config_hash(right)
    assert physics_hash(left) == physics_hash(right)


def test_physics_change_changes_both_hashes() -> None:
    left = {**_config_dict(), "seed": 1}
    right = {**_config_dict(), "seed": 1}
    right["clubhead"] = {**right["clubhead"], "bounce_deg": 12.0}  # type: ignore[dict-item]
    assert config_hash(left) != config_hash(right)
    assert physics_hash(left) != physics_hash(right)


def test_recording_rate_does_not_change_physics_hash() -> None:
    left = _config_dict()
    right = _config_dict()
    right["output"] = {"downsample_grains": 4, "rate_hz": 500.0}
    assert physics_hash(left) == physics_hash(right)
    assert config_hash(left) != config_hash(right)


def test_hashes_ignore_key_insertion_order() -> None:
    left = _config_dict()
    right = {key: left[key] for key in reversed(list(left))}
    assert config_hash(left) == config_hash(right)
