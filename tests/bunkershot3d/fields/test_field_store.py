"""Persisting a sand field: round trip, provenance, and the relabel refusal.

The load path is the point of this module.  Issue #8710's non-negotiable
is that an illustrative field cannot be relabelled by copying a file, and
a claim like that is only worth anything if something enforces it.  Here
the file is copied, renamed and edited in turn, and the loader is asked
what it thinks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="the result schema is HDF5")

from bunkershot3d.fields.schema import (  # noqa: E402
    FieldIntegrityError,
    FieldLayout,
    RetentionPolicy,
    SandFieldSeries,
    series_digest,
)
from bunkershot3d.fields.store import (  # noqa: E402
    DETERMINISTIC_SEED_NAME,
    field_manifest,
    load_field,
    save_field,
)
from bunkershot3d.io.schema import (  # noqa: E402
    FIELD_DIGEST_ATTR,
    FIELD_GROUP,
    FIELD_METADATA_ATTR,
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTR,
    SUPPORTED_SCHEMA_VERSIONS,
    BunkerShotResultReader,
    BunkerShotResultWriter,
)
from bunkershot3d.provenance.manifest import PROVENANCE_SUFFIX, Validity  # noqa: E402
from bunkershot3d.solvers.envelope import EnvelopeStatus  # noqa: E402
from bunkershot3d.solvers.protocol import FidelityTier  # noqa: E402
from tests.bunkershot3d.fields.test_field_schema import (  # noqa: E402
    grid_series,
    particle_series,
    provenance,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture
def stored(tmp_path: Path) -> tuple[Path, SandFieldSeries]:
    """A saved GRID field and the series it was saved from."""
    series = grid_series(n_frames=4)
    path = save_field(tmp_path / "field.h5", series)
    return path, series


class TestRoundTrip:
    """What goes in comes back, including which tier put it there."""

    def test_the_field_round_trips_its_arrays(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, series = stored
        loaded = load_field(path)
        assert loaded.n_frames == series.n_frames
        assert loaded.n_samples == series.n_samples
        np.testing.assert_array_equal(loaded.time_s, series.time_s)
        np.testing.assert_allclose(
            loaded.velocity_m_s, series.velocity_m_s, rtol=1e-6, atol=1e-9
        )

    def test_the_field_round_trips_its_tier_and_status(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        loaded = load_field(stored[0])
        assert loaded.provenance.fidelity_tier is FidelityTier.F1
        assert loaded.provenance.envelope_status is EnvelopeStatus.BEYOND_VALIDATION

    def test_the_field_round_trips_its_kinematics_and_refusals(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        loaded = load_field(stored[0])
        assert "approach" in loaded.provenance.kinematics
        assert "out_of_plane" in loaded.provenance.refused

    def test_the_field_round_trips_its_retention_record(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        loaded = load_field(stored[0])
        assert loaded.retention.time_stride == 10
        assert loaded.retention.dropped == ("kept every 10th step",)
        assert loaded.retention.policy.store_dtype == "float32"

    def test_the_field_round_trips_its_geometry(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, series = stored
        loaded = load_field(path)
        assert loaded.geometry is not None
        assert series.geometry is not None
        assert loaded.geometry.shape == series.geometry.shape
        np.testing.assert_allclose(
            loaded.sample_positions_m(0), series.sample_positions_m(0)
        )

    def test_a_particle_field_round_trips_too(self, tmp_path: Path) -> None:
        """The container is not F1-shaped: a 3-D grain tier fits it."""
        series = particle_series()
        loaded = load_field(save_field(tmp_path / "grains.h5", series))
        assert loaded.layout is FieldLayout.PARTICLE
        assert loaded.dimension == 3
        assert loaded.shear_rate_1_s is None
        assert loaded.positions_m is not None
        np.testing.assert_allclose(
            loaded.positions_m, series.positions_m, rtol=1e-6, atol=1e-9
        )

    def test_a_float64_policy_round_trips_bit_exactly(self, tmp_path: Path) -> None:
        series = grid_series(
            retention=grid_series().retention,
        )
        exact = SandFieldSeries(
            time_s=series.time_s,
            velocity_m_s=series.velocity_m_s,
            density_kg_m3=series.density_kg_m3,
            shear_rate_1_s=series.shear_rate_1_s,
            positions_m=None,
            layout=series.layout,
            geometry=series.geometry,
            provenance=series.provenance,
            retention=_with_policy(series, RetentionPolicy(store_dtype="float64")),
            occupancy=series.occupancy,
        )
        loaded = load_field(save_field(tmp_path / "exact.h5", exact))
        np.testing.assert_array_equal(loaded.velocity_m_s, exact.velocity_m_s)

    def test_nan_shear_survives_the_round_trip(self, tmp_path: Path) -> None:
        """An empty node's shear rate is nan, and nan is the answer."""
        series = grid_series()
        shear = np.array(series.shear_rate_1_s)
        shear[0, 0] = np.nan
        marked = SandFieldSeries(
            time_s=series.time_s,
            velocity_m_s=series.velocity_m_s,
            density_kg_m3=series.density_kg_m3,
            shear_rate_1_s=shear,
            positions_m=None,
            layout=series.layout,
            geometry=series.geometry,
            provenance=series.provenance,
            retention=series.retention,
            occupancy=series.occupancy,
        )
        loaded = load_field(save_field(tmp_path / "nan.h5", marked))
        assert loaded.shear_rate_1_s is not None
        assert np.isnan(loaded.shear_rate_1_s[0, 0])


class TestTierCannotBeRelabelled:
    """The non-negotiable, tested the way it would actually be broken."""

    def test_copying_the_file_under_a_new_name_changes_nothing(
        self, stored: tuple[Path, SandFieldSeries], tmp_path: Path
    ) -> None:
        path, _ = stored
        renamed = tmp_path / "predictive_F2_validated.h5"
        shutil.copyfile(path, renamed)
        assert load_field(renamed).provenance.fidelity_tier is FidelityTier.F1
        assert (
            load_field(renamed).provenance.envelope_status
            is EnvelopeStatus.BEYOND_VALIDATION
        )

    def test_editing_the_stored_tier_is_refused_on_load(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, _ = stored
        _rewrite_metadata(path, lambda meta: _set_tier(meta, "F2"))
        with pytest.raises(FieldIntegrityError, match="does not match its recorded"):
            load_field(path)

    def test_editing_the_stored_status_is_refused_on_load(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, _ = stored
        _rewrite_metadata(path, lambda meta: _set_status(meta, "within"))
        with pytest.raises(FieldIntegrityError, match="does not match its recorded"):
            load_field(path)

    def test_swapping_the_arrays_is_refused_on_load(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, _ = stored
        with h5py.File(path, "r+") as handle:
            handle[f"{FIELD_GROUP}/velocity"][0, 0, 0] += 1.0
        with pytest.raises(FieldIntegrityError, match="does not match its recorded"):
            load_field(path)

    def test_the_refusal_names_what_the_file_claims(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        """A reader who hits this must be told what was claimed."""
        path, _ = stored
        _rewrite_metadata(path, lambda meta: _set_tier(meta, "F3"))
        with pytest.raises(FieldIntegrityError) as caught:
            load_field(path)
        assert "F3" in str(caught.value)
        assert "beyond_validation" in str(caught.value)

    def test_a_schema_change_is_not_reported_as_tampering(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        """ "Regenerate this" and "somebody edited this" are different news."""
        path, _ = stored
        _rewrite_metadata(path, _bump_schema_version)
        with pytest.raises(FieldIntegrityError) as caught:
            load_field(path)
        assert "Nothing has been tampered with" in str(caught.value)
        assert "regenerating" in str(caught.value)

    def test_a_field_group_without_its_metadata_is_refused(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, _ = stored
        with h5py.File(path, "r+") as handle:
            del handle[FIELD_GROUP].attrs[FIELD_METADATA_ATTR]
        with pytest.raises(ValueError, match="unknowable"):
            load_field(path)

    def test_a_field_group_without_its_digest_is_refused(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, _ = stored
        with h5py.File(path, "r+") as handle:
            del handle[FIELD_GROUP].attrs[FIELD_DIGEST_ATTR]
        with pytest.raises(ValueError, match="unknowable"):
            load_field(path)

    def test_there_is_no_load_anyway_escape_hatch(self) -> None:
        """A caller must not be able to decide a broken field seems fine."""
        import inspect

        signature = inspect.signature(load_field)
        assert list(signature.parameters) == ["path"]


class TestProvenanceRecords:
    """Seeds, tier and settings, so a field traces back to its run."""

    def test_a_sidecar_is_always_written(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        path, _ = stored
        sidecar = path.with_suffix(path.suffix + PROVENANCE_SUFFIX)
        assert sidecar.exists()
        payload = json.loads(sidecar.read_text())
        assert payload["artifact"]["checksum_algorithm"] == "sha256"
        assert payload["run_manifest"]["fidelity_tier"] == "F1"

    def test_the_manifest_records_the_tier_and_the_validity(self) -> None:
        manifest = field_manifest(grid_series())
        assert manifest.fidelity_tier == "F1"
        assert manifest.validity is Validity.OUT_OF_ENVELOPE

    def test_a_refused_field_is_recorded_as_invalid(self) -> None:
        series = grid_series(
            provenance=provenance(envelope_status=EnvelopeStatus.REFUSED)
        )
        assert field_manifest(series).validity is Validity.INVALID

    def test_a_deterministic_tier_records_that_it_drew_no_numbers(self) -> None:
        """The manifest refuses empty seeds; the honest record says why."""
        manifest = field_manifest(grid_series())
        assert manifest.seeds[0].name == DETERMINISTIC_SEED_NAME

    def test_the_manifest_hashes_the_solver_settings(self) -> None:
        one = field_manifest(grid_series())
        other = field_manifest(
            grid_series(provenance=provenance(settings={"cell_size_m": 0.001}))
        )
        assert one.config_hash != other.config_hash

    def test_the_settings_are_enough_to_regenerate(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        settings = load_field(stored[0]).provenance.settings
        assert settings["cell_size_m"] == pytest.approx(0.002)
        assert settings["effective_width_m"] == pytest.approx(0.03)


class TestContainerVersioning:
    """A field is a new payload in the old container, not a new format."""

    def test_the_container_version_is_bumped(self) -> None:
        assert SCHEMA_VERSION == 3
        assert SUPPORTED_SCHEMA_VERSIONS == (1, 2, 3)

    def test_the_version_is_still_the_first_root_attribute(
        self, stored: tuple[Path, SandFieldSeries]
    ) -> None:
        with h5py.File(stored[0], "r") as handle:
            assert next(iter(handle.attrs)) == SCHEMA_VERSION_ATTR
            assert not isinstance(handle.attrs[SCHEMA_VERSION_ATTR], (str, bytes))

    def test_a_result_without_a_field_reads_back_as_none(self, tmp_path: Path) -> None:
        path = tmp_path / "streams_only.h5"
        with BunkerShotResultWriter(path) as writer:
            writer.write_clubhead_state(0.0, np.zeros(3), np.array([1.0, 0, 0, 0]))
        with BunkerShotResultReader(path) as reader:
            assert reader.read_sand_field() is None

    def test_loading_a_fieldless_result_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / "streams_only.h5"
        with BunkerShotResultWriter(path) as writer:
            writer.write_contact_wrench(0.0, np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="records no sand field"):
            load_field(path)

    def test_the_streams_still_round_trip_beside_a_field(self, tmp_path: Path) -> None:
        path = tmp_path / "both.h5"
        series = grid_series()
        with BunkerShotResultWriter(path) as writer:
            writer.write_clubhead_state(0.0, np.ones(3), np.array([1.0, 0, 0, 0]))
            writer.write_sand_field(
                _payload(series), compression="gzip", compression_level=4
            )
        with BunkerShotResultReader(path) as reader:
            times, positions, _ = reader.read_clubhead_states()
            assert times.shape == (1,)
            np.testing.assert_allclose(positions[0], np.ones(3))
            assert reader.read_sand_field() is not None

    def test_a_second_field_write_is_refused(self, tmp_path: Path) -> None:
        series = grid_series()
        with BunkerShotResultWriter(tmp_path / "twice.h5") as writer:
            writer.write_sand_field(_payload(series))
            with pytest.raises(ValueError, match="already written"):
                writer.write_sand_field(_payload(series))

    def test_an_unknown_array_name_is_refused(self, tmp_path: Path) -> None:
        payload = _payload(grid_series())
        payload.arrays["temperature"] = np.zeros((payload.time_s.size, 3))
        with (
            BunkerShotResultWriter(tmp_path / "unknown.h5") as writer,
            pytest.raises(ValueError, match="unknown sand field array"),
        ):
            writer.write_sand_field(payload)

    def test_a_field_without_metadata_cannot_be_written(self, tmp_path: Path) -> None:
        payload = _payload(grid_series())._replace(metadata="  ")
        with (
            BunkerShotResultWriter(tmp_path / "bare.h5") as writer,
            pytest.raises(ValueError, match="not in the filename"),
        ):
            writer.write_sand_field(payload)

    def test_a_field_without_a_digest_cannot_be_written(self, tmp_path: Path) -> None:
        payload = _payload(grid_series())._replace(digest="")
        with (
            BunkerShotResultWriter(tmp_path / "undigested.h5") as writer,
            pytest.raises(ValueError, match="digest"),
        ):
            writer.write_sand_field(payload)

    def test_the_writer_copies_a_live_array(self, tmp_path: Path) -> None:
        """A caller may hand a live solver view; the file must not follow it."""
        series = grid_series()
        live = np.array(series.velocity_m_s)
        payload = _payload(series)
        payload.arrays["velocity"] = live
        path = tmp_path / "aliased.h5"
        with BunkerShotResultWriter(path) as writer:
            writer.write_sand_field(payload)
            live *= 0.0
        with BunkerShotResultReader(path) as reader:
            stored_payload = reader.read_sand_field()
        assert stored_payload is not None
        assert float(np.abs(stored_payload.arrays["velocity"]).max()) > 0.0


class TestCompressionIsDeliberate:
    """Compression is a recorded choice, not an accident of the writer."""

    def test_the_policy_choice_reaches_the_dataset(self, tmp_path: Path) -> None:
        series = grid_series()
        path = save_field(tmp_path / "gz.h5", series)
        with h5py.File(path, "r") as handle:
            assert handle[f"{FIELD_GROUP}/velocity"].compression == "gzip"

    def test_no_compression_is_also_a_choice(self, tmp_path: Path) -> None:
        series = grid_series()
        plain = SandFieldSeries(
            time_s=series.time_s,
            velocity_m_s=series.velocity_m_s,
            density_kg_m3=series.density_kg_m3,
            shear_rate_1_s=series.shear_rate_1_s,
            positions_m=None,
            layout=series.layout,
            geometry=series.geometry,
            provenance=series.provenance,
            retention=_with_policy(series, RetentionPolicy(compression="")),
            occupancy=series.occupancy,
        )
        path = save_field(tmp_path / "plain.h5", plain)
        with h5py.File(path, "r") as handle:
            assert handle[f"{FIELD_GROUP}/velocity"].compression is None

    def test_the_stored_precision_matches_the_declared_one(
        self, tmp_path: Path
    ) -> None:
        path = save_field(tmp_path / "f32.h5", grid_series())
        with h5py.File(path, "r") as handle:
            assert handle[f"{FIELD_GROUP}/velocity"].dtype == np.float32


def _payload(series: SandFieldSeries):  # type: ignore[no-untyped-def]
    """The io-layer payload for a series, for writer-level tests."""
    from bunkershot3d.io.schema import SandFieldPayload
    from bunkershot3d.provenance.hashing import canonical_json

    arrays = {
        "velocity": np.asarray(series.velocity_m_s),
        "density": np.asarray(series.density_kg_m3),
    }
    if series.shear_rate_1_s is not None:
        arrays["shear_rate"] = np.asarray(series.shear_rate_1_s)
    return SandFieldPayload(
        time_s=np.asarray(series.time_s),
        arrays=arrays,
        metadata=canonical_json(series.metadata()),
        digest=series_digest(series),
    )


def _with_policy(series: SandFieldSeries, policy: RetentionPolicy):  # type: ignore[no-untyped-def]
    """The series' retention record with a different policy on it."""
    from bunkershot3d.fields.schema import RetentionRecord

    record = series.retention
    return RetentionRecord(
        policy=policy,
        steps_marched=record.steps_marched,
        time_stride=record.time_stride,
        frames_kept=record.frames_kept,
        time_step_s=record.time_step_s,
        samples_in_domain=record.samples_in_domain,
        samples_kept=record.samples_kept,
        dropped=record.dropped,
    )


def _rewrite_metadata(path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    """Edit the stored metadata JSON in place, leaving the digest alone."""
    with h5py.File(path, "r+") as handle:
        group = handle[FIELD_GROUP]
        payload = json.loads(group.attrs[FIELD_METADATA_ATTR])
        del group.attrs[FIELD_METADATA_ATTR]
        group.attrs[FIELD_METADATA_ATTR] = json.dumps(
            edit(payload), separators=(",", ":"), sort_keys=True
        )


def _set_tier(metadata: dict, tier: str) -> dict:
    """Return ``metadata`` with a different declared fidelity tier."""
    metadata["provenance"]["fidelity_tier"] = tier
    return metadata


def _set_status(metadata: dict, status: str) -> dict:
    """Return ``metadata`` with a different declared envelope status."""
    metadata["provenance"]["envelope_status"] = status
    return metadata


def _bump_schema_version(metadata: dict) -> dict:
    """Return ``metadata`` as if a newer field schema had written it."""
    metadata["field_schema_version"] = int(metadata["field_schema_version"]) + 1
    return metadata
