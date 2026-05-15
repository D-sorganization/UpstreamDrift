"""
LIGGGHTS backend driver for BunkerShot3D.
"""

from pathlib import Path
import contextlib
import subprocess
import tempfile
import os
from bunkershot3d.config import BunkerShotConfig


class LiggghtsDriver:
    """Driver for running the bunker shot simulation using LIGGGHTS."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)

    def _generate_input_deck(self, work_dir: Path) -> Path:
        """Generate the LIGGGHTS LIGGGHTS.in script."""
        input_deck_path = work_dir / "in.bunkershot"
        with open(input_deck_path, "w") as f:
            f.write("# LIGGGHTS input script for BunkerShot3D\n")
            f.write("atom_style granular\n")
            f.write("boundary f f f\n")
            f.write("newton off\n")
            f.write("communicate single vel yes\n")
            f.write("units si\n")
            f.write(
                "# TODO: Add full LIGGGHTS commands for geometry, particles, and meshes\n"
            )
            f.write("run 0\n")
        return input_deck_path

    def run(self, output_path: Path | str) -> None:
        """Run the simulation via subprocess and parse dump output into HDF5."""
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)

            # Write STL and trajectory data needed by LIGGGHTS meshes
            # ...

            input_deck = self._generate_input_deck(work_dir)

            # Execute LIGGGHTS
            with contextlib.suppress(subprocess.CalledProcessError):
                subprocess.run(
                    ["liggghts", "-in", str(input_deck)],
                    cwd=str(work_dir),
                    check=True,
                    capture_output=True,
                )

            # TODO: Parse the generated dump files and use BunkerShotResultWriter
