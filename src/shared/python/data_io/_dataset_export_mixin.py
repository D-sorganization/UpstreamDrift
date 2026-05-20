"""DatasetGenerator export mixin: HDF5, SQLite, and CSV export methods.

Extracted from dataset_generator.py.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from .dataset_generator import SimulationSample, TrainingDataset

logger = get_logger(__name__)


class _DatasetExportMixin:
    """Export methods for DatasetGenerator. Use via DatasetGenerator, not directly."""

    def export_to_hdf5(self, dataset: TrainingDataset, output_path: str | Path) -> Path:
        """Export dataset to HDF5 format.

        Args:
            dataset: Training dataset to export.
            output_path: Output file path (without extension).

        Returns:
            Path to the created HDF5 file.

        Raises:
            ImportError: If h5py is not available.
        """
        if dataset is None:
            raise ValueError("dataset must be provided")
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py required for HDF5 export: pip install h5py"
            ) from None

        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".hdf5")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_path), "w") as f:
            self._write_hdf5_metadata(f, dataset)
            samples_grp = f.create_group("samples")
            for sample in dataset.samples:
                self._write_hdf5_sample(samples_grp, sample)

        logger.info("Exported dataset to HDF5: %s", output_path)
        return output_path

    @staticmethod
    def _write_hdf5_metadata(f: Any, dataset: TrainingDataset) -> None:
        """Write dataset-level metadata to an HDF5 file."""
        if dataset is None:
            raise ValueError("dataset must be provided")
        meta = f.create_group("metadata")
        meta.attrs["model_name"] = dataset.model_name
        meta.attrs["engine_name"] = dataset.engine_name
        meta.attrs["num_samples"] = dataset.num_samples
        meta.attrs["total_frames"] = dataset.total_frames
        meta.attrs["creation_time"] = dataset.creation_time
        meta.attrs["duration"] = dataset.config.duration
        meta.attrs["timestep"] = dataset.config.timestep
        meta.attrs["seed"] = dataset.config.seed

        if dataset.joint_names:
            meta.create_dataset(
                "joint_names",
                data=[n.encode("utf-8") for n in dataset.joint_names],
            )

    @staticmethod
    def _write_hdf5_sample(samples_grp: Any, sample: SimulationSample) -> None:
        """Write a single sample's data to an HDF5 samples group."""
        if sample is None:
            raise ValueError("sample must be provided")
        s_grp = samples_grp.create_group(f"sample_{sample.sample_id:06d}")
        s_grp.create_dataset("times", data=sample.times, compression="gzip")
        s_grp.create_dataset("positions", data=sample.positions, compression="gzip")
        s_grp.create_dataset("velocities", data=sample.velocities, compression="gzip")
        s_grp.create_dataset(
            "accelerations", data=sample.accelerations, compression="gzip"
        )
        s_grp.create_dataset("torques", data=sample.torques, compression="gzip")

        optional_fields = [
            ("mass_matrices", sample.mass_matrices),
            ("bias_forces", sample.bias_forces),
            ("gravity_forces", sample.gravity_forces),
            ("contact_forces", sample.contact_forces),
            ("drift_accelerations", sample.drift_accelerations),
            ("control_accelerations", sample.control_accelerations),
        ]
        for field_name, field_data in optional_fields:
            if field_data is not None:
                s_grp.create_dataset(field_name, data=field_data, compression="gzip")

        if sample.energies:
            e_grp = s_grp.create_group("energies")
            for key, arr in sample.energies.items():
                e_grp.create_dataset(key, data=arr, compression="gzip")

        s_grp.attrs["metadata"] = json.dumps(sample.metadata)

    def export_to_sqlite(
        self, dataset: TrainingDataset, output_path: str | Path
    ) -> Path:
        """Export dataset to SQLite database.

        Args:
            dataset: Training dataset to export.
            output_path: Output database path.

        Returns:
            Path to the created SQLite database.
        """
        if dataset is None:
            raise ValueError("dataset must be provided")
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".db")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(output_path))
        try:
            cursor = conn.cursor()
            self._create_sqlite_tables(cursor)
            self._insert_sqlite_metadata(cursor, dataset)
            for sample in dataset.samples:
                self._insert_sqlite_sample(cursor, sample)
            conn.commit()
        finally:
            conn.close()

        logger.info("Exported dataset to SQLite: %s", output_path)
        return output_path

    @staticmethod
    def _create_sqlite_tables(cursor: sqlite3.Cursor) -> None:
        """Create the SQLite schema tables."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY,
                metadata_json TEXT,
                n_steps INTEGER,
                n_q INTEGER,
                n_v INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                sample_id INTEGER,
                step INTEGER,
                time REAL,
                positions_json TEXT,
                velocities_json TEXT,
                accelerations_json TEXT,
                torques_json TEXT,
                kinetic_energy REAL,
                PRIMARY KEY (sample_id, step),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)

    @staticmethod
    def _insert_sqlite_metadata(
        cursor: sqlite3.Cursor, dataset: TrainingDataset
    ) -> None:
        """Insert dataset-level metadata into the SQLite database."""
        if cursor is None:
            raise ValueError("cursor must be provided")
        meta_items = [
            ("model_name", dataset.model_name),
            ("engine_name", dataset.engine_name),
            ("num_samples", str(dataset.num_samples)),
            ("total_frames", str(dataset.total_frames)),
            ("creation_time", str(dataset.creation_time)),
            ("seed", str(dataset.config.seed)),
            ("duration", str(dataset.config.duration)),
            ("timestep", str(dataset.config.timestep)),
            ("joint_names", json.dumps(dataset.joint_names)),
        ]
        cursor.executemany(
            "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)",
            meta_items,
        )

    @staticmethod
    def _insert_sqlite_sample(cursor: sqlite3.Cursor, sample: SimulationSample) -> None:
        """Insert a single sample and its frames into the SQLite database."""
        if cursor is None:
            raise ValueError("cursor must be provided")
        n_steps = len(sample.times)
        n_q = sample.positions.shape[1] if sample.positions.ndim > 1 else 0
        n_v = sample.velocities.shape[1] if sample.velocities.ndim > 1 else 0

        cursor.execute(
            "INSERT INTO samples (sample_id, metadata_json, n_steps, n_q, n_v) "
            "VALUES (?, ?, ?, ?, ?)",
            (sample.sample_id, json.dumps(sample.metadata), n_steps, n_q, n_v),
        )

        frame_rows = []
        for step in range(n_steps):
            ke = (
                float(sample.energies["kinetic"][step])
                if "kinetic" in sample.energies
                else 0.0
            )
            frame_rows.append(
                (
                    sample.sample_id,
                    step,
                    float(sample.times[step]),
                    json.dumps(sample.positions[step].tolist()),
                    json.dumps(sample.velocities[step].tolist()),
                    json.dumps(sample.accelerations[step].tolist()),
                    json.dumps(sample.torques[step].tolist()),
                    ke,
                )
            )

        cursor.executemany(
            "INSERT INTO frames "
            "(sample_id, step, time, positions_json, velocities_json, "
            "accelerations_json, torques_json, kinetic_energy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            frame_rows,
        )

    def export_to_csv(self, dataset: TrainingDataset, output_dir: str | Path) -> Path:
        """Export dataset to CSV files (one per sample).

        Args:
            dataset: Training dataset to export.
            output_dir: Output directory for CSV files.

        Returns:
            Path to the output directory.
        """
        if dataset is None:
            raise ValueError("dataset must be provided")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample in dataset.samples:
            n_steps = len(sample.times)
            n_q = sample.positions.shape[1]
            n_v = sample.velocities.shape[1]

            # Build header
            headers = ["time"]
            headers.extend([f"q_{i}" for i in range(n_q)])
            headers.extend([f"v_{i}" for i in range(n_v)])
            headers.extend([f"a_{i}" for i in range(n_v)])
            headers.extend([f"tau_{i}" for i in range(n_v)])
            headers.append("kinetic_energy")

            # Build data matrix
            data_cols = [sample.times.reshape(-1, 1)]
            data_cols.append(sample.positions)
            data_cols.append(sample.velocities)
            data_cols.append(sample.accelerations)
            data_cols.append(sample.torques)

            ke = sample.energies.get("kinetic", np.zeros(n_steps))
            data_cols.append(ke.reshape(-1, 1))

            data = np.hstack(data_cols)

            filepath = output_dir / f"sample_{sample.sample_id:06d}.csv"
            np.savetxt(
                str(filepath),
                data,
                delimiter=",",
                header=",".join(headers),
                comments="",
            )

        # Write metadata file
        meta_path = output_dir / "metadata.json"
        meta = {
            "model_name": dataset.model_name,
            "engine_name": dataset.engine_name,
            "num_samples": dataset.num_samples,
            "total_frames": dataset.total_frames,
            "joint_names": dataset.joint_names,
            "config": {
                "duration": dataset.config.duration,
                "timestep": dataset.config.timestep,
                "seed": dataset.config.seed,
                "num_samples": dataset.config.num_samples,
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Exported dataset to CSV: %s", output_dir)
        return output_dir

    @precondition(
        lambda self, dataset, output_path, format="hdf5": dataset is not None,
        "Dataset must not be None",
    )
    @precondition(
        lambda self, dataset, output_path, format="hdf5": (
            output_path is not None and len(str(output_path)) > 0
        ),
        "Output path must be a non-empty string or Path",
    )
    @precondition(
        lambda self, dataset, output_path, format="hdf5": (
            format in ("hdf5", "sqlite", "db", "csv")
        ),
        "Export format must be one of: hdf5, sqlite, db, csv",
    )
    def export(
        self,
        dataset: TrainingDataset,
        output_path: str | Path,
        format: str = "hdf5",
    ) -> Path:
        """Export dataset in the specified format.

        Args:
            dataset: Training dataset to export.
            output_path: Output path (file or directory depending on format).
            format: Export format ('hdf5', 'sqlite', 'csv').

        Returns:
            Path to the exported data.

        Raises:
            ValueError: If format is not supported.
        """
        format = format.lower()
        if format == "hdf5":
            return self.export_to_hdf5(dataset, output_path)
        if format in ("sqlite", "db"):
            return self.export_to_sqlite(dataset, output_path)
        if format == "csv":
            return self.export_to_csv(dataset, output_path)
        raise ValueError(
            f"Unsupported export format: {format}. Supported: hdf5, sqlite, csv"
        )


__all__ = ["_DatasetExportMixin"]
