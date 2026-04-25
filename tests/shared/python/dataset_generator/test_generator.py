import os
import sqlite3

import h5py
import numpy as np

from src.shared.python.dataset_generator.generator import DatasetGenerator
from src.shared.python.engine_core.mock_engine import MockPhysicsEngine


def test_generate_trajectory():
    engine = MockPhysicsEngine(num_joints=2)
    engine.load_from_string("dummy_model")
    generator = DatasetGenerator(engine, output_dir="test_dataset_output")

    q0 = np.zeros(2)
    v0 = np.zeros(2)
    controls = [np.array([1.0, 0.5]), np.array([0.0, -0.5])]

    data = generator.generate_trajectory(q0, v0, controls, dt=0.01)

    assert "joint_positions" in data
    assert "joint_velocities" in data
    assert "joint_accelerations" in data
    assert "joint_torques" in data
    assert len(data["joint_positions"]) == 2
    assert "run_id" in data


def test_export_to_hdf5(tmpdir):
    engine = MockPhysicsEngine(num_joints=2)
    engine.load_from_string("dummy_model")
    output_dir = str(tmpdir.mkdir("dataset"))
    generator = DatasetGenerator(engine, output_dir=output_dir)

    data = {
        "joint_positions": np.array([[0, 0], [1, 1]]),
        "scalar_val": 42.0,
        "nested": {"arr": np.array([1, 2, 3])},
    }

    generator.export_to_hdf5(data, "test.h5")

    filepath = os.path.join(output_dir, "test.h5")
    assert os.path.exists(filepath)

    with h5py.File(filepath, "r") as f:
        assert "joint_positions" in f
        assert np.array_equal(f["joint_positions"][:], data["joint_positions"])
        assert f.attrs["scalar_val"] == 42.0
        assert "nested" in f
        assert "arr" in f["nested"]
        assert np.array_equal(f["nested/arr"][:], np.array([1, 2, 3]))


def test_generate_batch(tmpdir):
    engine = MockPhysicsEngine(num_joints=2)
    engine.load_from_string("dummy_model")
    output_dir = str(tmpdir.mkdir("batch_dataset"))
    generator = DatasetGenerator(engine, output_dir=output_dir)

    min_q = np.array([-1.0, -1.0])
    max_q = np.array([1.0, 1.0])
    min_v = np.array([-5.0, -5.0])
    max_v = np.array([5.0, 5.0])

    generator.generate_batch(
        n_runs=3,
        q_range=(min_q, max_q),
        v_range=(min_v, max_v),
        steps_per_run=5,
        dt=0.01,
        seed=42,
    )

    # Check sqlite index
    db_path = os.path.join(output_dir, "dataset_index.sqlite")
    assert os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM runs")
    count = cursor.fetchone()[0]
    assert count == 3

    cursor.execute("SELECT run_id, hdf5_file FROM runs")
    rows = cursor.fetchall()
    conn.close()

    # Check HDF5 files
    for _run_id, h5_file in rows:
        h5_path = os.path.join(output_dir, h5_file)
        assert os.path.exists(h5_path)
        with h5py.File(h5_path, "r") as f:
            assert "joint_positions" in f
            assert len(f["joint_positions"]) == 5
