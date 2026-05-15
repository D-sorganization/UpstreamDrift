"""
Common result schema (HDF5) for BunkerShot3D results.
"""


from pathlib import Path
import h5py
import numpy as np


class BunkerShotResultWriter:
    """Writer for BunkerShot3D results to HDF5 format."""

    def __init__(self, filepath: Path | str) -> None:
        """Initialize the HDF5 writer."""
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, "w")

        # Initialize datasets
        self.clubhead_group = self.file.create_group("clubhead")
        self.wrench_group = self.file.create_group("wrench")
        self.grains_group = self.file.create_group("grains")

    def write_clubhead_state(
        self, time: float, position: np.ndarray, orientation_quat: np.ndarray
    ) -> None:
        """Write a single clubhead state."""
        grp_name = f"t_{time:.6f}"
        grp = self.clubhead_group.create_group(grp_name)
        grp.attrs["time"] = time
        grp.create_dataset("position", data=position)
        grp.create_dataset("orientation", data=orientation_quat)

    def write_contact_wrench(
        self, time: float, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """Write a single contact wrench."""
        grp_name = f"t_{time:.6f}"
        grp = self.wrench_group.create_group(grp_name)
        grp.attrs["time"] = time
        grp.create_dataset("force", data=force)
        grp.create_dataset("torque", data=torque)

    def write_grain_state(
        self, time: float, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """Write a single grain state (downsampled if needed)."""
        grp_name = f"t_{time:.6f}"
        grp = self.grains_group.create_group(grp_name)
        grp.attrs["time"] = time
        grp.create_dataset("positions", data=positions)
        grp.create_dataset("velocities", data=velocities)

    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()


class BunkerShotResultReader:
    """Reader for BunkerShot3D results from HDF5 format."""

    def __init__(self, filepath: Path | str) -> None:
        """Initialize the HDF5 reader."""
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, "r")

    def read_clubhead_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read all clubhead states."""
        times = []
        positions = []
        quats = []

        grp = self.file["clubhead"]
        for key in sorted(grp.keys()):
            subgrp = grp[key]
            times.append(subgrp.attrs["time"])
            positions.append(subgrp["position"][:])
            quats.append(subgrp["orientation"][:])

        return np.array(times), np.array(positions), np.array(quats)

    def read_contact_wrenches(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read all contact wrenches."""
        times = []
        forces = []
        torques = []

        grp = self.file["wrench"]
        for key in sorted(grp.keys()):
            subgrp = grp[key]
            times.append(subgrp.attrs["time"])
            forces.append(subgrp["force"][:])
            torques.append(subgrp["torque"][:])

        return np.array(times), np.array(forces), np.array(torques)

    def read_grain_states(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Read all grain states. Returns lists for pos/vel because particle count might vary."""
        times = []
        positions = []
        velocities = []

        grp = self.file["grains"]
        for key in sorted(grp.keys()):
            subgrp = grp[key]
            times.append(subgrp.attrs["time"])
            positions.append(subgrp["positions"][:])
            velocities.append(subgrp["velocities"][:])

        return np.array(times), positions, velocities

    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()
