"""Dataset Generator for Neural Network Training.

Produces structured training datasets (HDF5 + SQLite) from physics engine simulations.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import h5py
import numpy as np

from src.shared.python.dashboard.recorder import GenericPhysicsRecorder
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class DatasetGenerator:
    """Generates ML-ready datasets from physics engine simulations."""

    def __init__(
        self, engine: PhysicsEngine, output_dir: str = "dataset_output"
    ) -> None:
        """Initialize the DatasetGenerator.

        Args:
            engine: PhysicsEngine to use for data generation.
            output_dir: Directory to save generated datasets.
        """
        if engine is None:
            raise ValueError("engine must be provided")
        self.engine = engine
        self.output_dir = output_dir
        self.recorder = GenericPhysicsRecorder(engine)

        # Enable full recording config for dataset generation
        self.recorder.set_analysis_config(
            {
                "ztcf": True,
                "zvcf": True,
                "track_drift": True,
                "track_total_control": True,
            }
        )

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_trajectory(
        self,
        initial_q: np.ndarray,
        initial_v: np.ndarray,
        control_profile: list[np.ndarray],
        dt: float = 0.01,
        run_id: str = "run_0",
    ) -> dict[str, Any]:
        """Generate a single trajectory run.

        Args:
            initial_q: Initial positions.
            initial_v: Initial velocities.
            control_profile: List of control vectors to apply sequentially.
            dt: Time step.
            run_id: Identifier for this run.

        Returns:
            Dictionary of recorded data.
        """
        if initial_q is None or initial_v is None:
            raise ValueError("initial_q and initial_v must be provided")

        self.recorder.reset()
        self.engine.set_state(initial_q, initial_v)

        self.recorder.start()

        for u in control_profile:
            self.engine.set_control(u)
            self.engine.step(dt)
            self.recorder.record_step(u)

        self.recorder.stop()

        data = self.recorder.get_data_dict()
        data["run_id"] = run_id
        return data

    def export_to_hdf5(self, data: dict[str, Any], filename: str) -> None:
        """Export generated data to HDF5 format.

        Args:
            data: Dictionary of recorded data.
            filename: Output filename (within output_dir).
        """
        filepath = os.path.join(self.output_dir, filename)

        with h5py.File(filepath, "w") as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, int | float | str):
                    f.attrs[key] = value
                elif isinstance(value, dict):
                    # Handle nested dicts (like induced_accelerations)
                    group = f.create_group(key)
                    for sub_k, sub_v in value.items():
                        if isinstance(sub_v, np.ndarray):
                            group.create_dataset(str(sub_k), data=sub_v)

        logger.info(f"Dataset exported to HDF5: {filepath}")

    def generate_batch(
        self,
        n_runs: int,
        q_range: tuple[np.ndarray, np.ndarray],
        v_range: tuple[np.ndarray, np.ndarray],
        steps_per_run: int,
        dt: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Generate a batch of varied trajectories.

        Args:
            n_runs: Number of independent trajectories to generate.
            q_range: (min_q, max_q) bounds for random initial positions.
            v_range: (min_v, max_v) bounds for random initial velocities.
            steps_per_run: Number of steps to simulate per run.
            dt: Time step.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        min_q, max_q = q_range
        min_v, max_v = v_range

        db_path = os.path.join(self.output_dir, "dataset_index.sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                hdf5_file TEXT,
                num_steps INTEGER,
                dt REAL,
                seed INTEGER
            )
        """)

        n_u = len(min_v)  # Assuming num_controls == num_velocities

        for i in range(n_runs):
            run_id = f"run_{i:04d}"

            # Sample initial conditions
            q0 = np.random.uniform(min_q, max_q)
            v0 = np.random.uniform(min_v, max_v)

            # Simple random control profile
            controls = [np.random.uniform(-1, 1, n_u) for _ in range(steps_per_run)]

            # Generate and export
            data = self.generate_trajectory(q0, v0, controls, dt, run_id)
            h5_filename = f"{run_id}.h5"
            self.export_to_hdf5(data, h5_filename)

            # Record in SQLite index
            cursor.execute(
                "INSERT OR REPLACE INTO runs (run_id, hdf5_file, num_steps, dt, seed) VALUES (?, ?, ?, ?, ?)",
                (run_id, h5_filename, steps_per_run, dt, seed),
            )

        conn.commit()
        conn.close()
        logger.info(
            f"Batch generation complete. {n_runs} runs saved to {self.output_dir}"
        )
