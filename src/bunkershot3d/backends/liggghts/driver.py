"""
LIGGGHTS backend driver for BunkerShot3D — **non-viable, refuses to run**.

Issue #8612 (finding B2). The generated input deck contains a sand box,
gravity, particle insertion and a run command, and **no clubhead of any kind**.
Nothing strikes the sand, so executing it produces a settling sand box wearing
the label of a bunker shot. ADR-0032 records the DEM tier as intractable at
true grain scale (2.1e8 grains, days per shot) and keeps LIGGGHTS in-tree only
as accepted debt; the honest behaviour is therefore to refuse rather than to
return a plausible-looking result file.

Deck generation and dump parsing remain available and tested — they are the
record of what a viable deck would need, and the parser is useful for reading
dumps produced elsewhere — but :meth:`LiggghtsDriver.setup` and
:meth:`LiggghtsDriver.run` raise.

The retained deck writer also fixes two defects found alongside B2: the
log-normal size distribution was emitted as ``radius gaussian`` with sigma read
as metres rather than log-space (B11), and the timestep was hard-coded at
1e-5 s in two places with the run length taken from a fixed 0.5 s that ignored
``trajectory.duration`` (B12).
"""

from __future__ import annotations

import collections.abc
import math
import statistics
from pathlib import Path

import numpy as np

from ...config import BunkerShotConfig
from ...exceptions import BackendNotImplementedError
from ...io.schema import BunkerShotResultWriter
from ...provenance import RunManifest, Validity
from ..run_provenance import (
    LIGGGHTS_GENERATOR,
    dem_run_manifest,
    fixed_seed_record,
)
from ..stability import (
    rayleigh_timestep,
    smallest_grain_radius,
    validate_contact_model,
)

#: Number of equal-probability bins used to discretise the log-normal grain
#: size distribution. LIGGGHTS' ``particletemplate/sphere`` has no log-normal
#: primitive, so the distribution is represented exactly at its quantiles.
SIZE_BINS = 5

#: Seed LIGGGHTS' ``insert/pack`` uses to place particles, as written into the
#: generated deck. Recorded in the run manifest so the packing is replayable.
INSERTION_SEED = 5330

#: Seed of the ``particledistribution/discrete`` fix in the generated deck.
DISTRIBUTION_SEED = 32452843

#: Base seed of the per-bin ``particletemplate/sphere`` fixes; bin ``i`` uses
#: ``TEMPLATE_SEED_BASE + 2 i``.
TEMPLATE_SEED_BASE = 15485863

#: Significant digits used when writing the timestep into the deck. The value
#: is floored, never rounded, so the written deck cannot exceed the limit.
_TIMESTEP_DIGITS = 6

_NON_VIABLE = (
    "the generated input deck contains no clubhead — it is a sand box with "
    "gravity and particle insertion, so running it would simulate settling "
    "sand and label the result a bunker shot. ADR-0032 records LIGGGHTS as a "
    "non-viable tier (2.1e8 grains and days per shot at true grain scale). "
    "Deck generation and dump parsing remain available; execution does not "
    "(issue #8612, finding B2)."
)


def _floor_to_significant(value: float, digits: int = _TIMESTEP_DIGITS) -> float:
    """Round *down* to ``digits`` significant figures."""
    if value <= 0.0:
        raise ValueError(f"value must be positive, got {value}")
    scale = 10.0 ** (math.floor(math.log10(value)) - digits + 1)
    return math.floor(value / scale) * scale


def lognormal_size_bins(
    median_radius: float, sigma_log: float, bins: int = SIZE_BINS
) -> list[tuple[float, float]]:
    """Discretise a log-normal radius distribution into equal-probability bins.

    Returns:
        ``[(radius, number_fraction), ...]``. Bin ``i`` takes the radius at the
        ``(i + 0.5) / bins`` quantile, ``r = median * exp(sigma * z)``, so the
        geometric mean of the radii is the median by construction and sigma is
        interpreted in log-space — not in metres, as ``radius gaussian`` did.
    """
    if median_radius <= 0.0:
        raise ValueError(f"median_radius must be positive, got {median_radius}")
    if sigma_log < 0.0:
        raise ValueError(f"sigma_log must be non-negative, got {sigma_log}")
    if bins < 1:
        raise ValueError(f"bins must be positive, got {bins}")

    normal = statistics.NormalDist()
    fraction = 1.0 / bins
    return [
        (
            median_radius * math.exp(sigma_log * normal.inv_cdf((index + 0.5) / bins)),
            fraction,
        )
        for index in range(bins)
    ]


class LiggghtsDriver:
    """Driver for the LIGGGHTS backend. Deck authoring only; refuses to run."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Refuse: the deck models no clubhead.

        Raises:
            BackendNotImplementedError: Always.
        """
        raise BackendNotImplementedError("liggghts", feature=_NON_VIABLE)

    def run(self, output_path: Path | str) -> None:
        """Refuse: the deck models no clubhead.

        Args:
            output_path: Ignored; no result is ever produced.

        Raises:
            BackendNotImplementedError: Always.
        """
        raise BackendNotImplementedError("liggghts", feature=_NON_VIABLE)

    def run_manifest(self) -> RunManifest:
        """Provenance for a result parsed out of a LIGGGHTS dump (#8608, B18).

        The seeds recorded are the deck's own: LIGGGHTS drives LAMMPS' Park/
        Miller generator from integer literals in the input script, not numpy,
        and the manifest says so rather than implying the PCG64DXSM discipline
        of :mod:`bunkershot3d.provenance.rng`.

        Returns:
            A manifest carrying the configuration hashes, the deck seeds and an
            ``INVALID`` verdict.
        """
        return dem_run_manifest(
            self.config,
            solver="liggghts",
            seeds=(
                fixed_seed_record(
                    "liggghts-insertion", INSERTION_SEED, LIGGGHTS_GENERATOR
                ),
                fixed_seed_record(
                    "liggghts-size-distribution",
                    DISTRIBUTION_SEED,
                    LIGGGHTS_GENERATOR,
                ),
            ),
            validity=Validity.INVALID,
            validity_reason=_NON_VIABLE,
        )

    # ------------------------------------------------------------------
    # Timestep
    # ------------------------------------------------------------------

    def integration_timestep(self) -> float:
        """Rayleigh-stable integration timestep for this configuration (s).

        Single source of truth for the deck writer and the dump parser, which
        previously carried two independent hard-coded copies of ``1.0e-5``
        (#8612, B12). The *smallest* grain in the population governs.
        """
        material = self.config.to_contact_material()
        return _floor_to_significant(
            rayleigh_timestep(
                radius=smallest_grain_radius(self.config),
                density=self.config.to_grain_population().density_kg_m3,
                youngs_modulus=material.youngs_modulus_pa,
                poisson_ratio=material.poisson_ratio,
            )
        )

    # ------------------------------------------------------------------
    # Input-deck generation
    # ------------------------------------------------------------------

    def _size_distribution_block(self) -> str:
        """LIGGGHTS templates encoding the log-normal size distribution."""
        grains = self.config.to_grain_population()
        bins = lognormal_size_bins(
            grains.radius_mean_m,
            grains.diameter_sigma_log,
        )
        density = grains.density_kg_m3

        templates = "\n".join(
            f"fix             pts{index + 1} all particletemplate/sphere "
            f"{TEMPLATE_SEED_BASE + 2 * index} atom_type 1 \\\n"
            f"                    density constant {density:.6g} \\\n"
            f"                    radius constant {radius:.10g}"
            for index, (radius, _fraction) in enumerate(bins)
        )
        pairs = " ".join(
            f"pts{index + 1} {fraction:.10g}"
            for index, (_radius, fraction) in enumerate(bins)
        )
        distribution = (
            f"fix             pdd1 all particledistribution/discrete "
            f"{DISTRIBUTION_SEED} {len(bins)} {pairs}"
        )
        return f"{templates}\n\n{distribution}"

    def _generate_input_deck(self, work_dir: Path) -> Path:
        """Generate a complete LIGGGHTS input script from *BunkerShotConfig*.

        The deck uses Hertz-Mindlin contact, SI units, fixed boundary walls,
        and dumps atom positions + velocities at the configured output rate.

        Raises:
            ContactStiffnessError: The configured stiffness cannot resolve a
                tour-speed impact without gross grain interpenetration.
        """
        validate_contact_model(self.config)

        cfg = self.config
        domain = cfg.bunker_bed.domain
        gp = cfg.grain_population
        cm = cfg.contact_model
        out = cfg.output

        lx, ly, lz = domain.length_x, domain.width_y, domain.depth_z
        r_max = (gp.diameter_mean / 2.0) * math.exp(3.0 * gp.diameter_sigma_log)

        dt = self.integration_timestep()
        dump_every = max(1, round(1.0 / (out.rate_hz * dt)))
        total_steps = max(1, round(cfg.to_trajectory_source().duration_s / dt))

        input_deck_path = work_dir / "in.bunkershot"
        with open(input_deck_path, "w", encoding="utf-8") as f:
            f.write(f"""# LIGGGHTS input script generated by BunkerShot3D
# Issue: #5552; timestep and size distribution per #8612.
# NOTE: this deck has no intruder. It is retained for inspection only; the
# driver refuses to execute it (ADR-0032).

atom_style      granular
atom_modify     map array
boundary        f f f
newton          off
communicate     single vel yes
units           si

# -- Domain -----------------------------------------------------------------
region          domain block 0.0 {lx:.6g} 0.0 {ly:.6g} 0.0 {lz:.6g} units box
create_box      1 domain

# -- Neighbour / communication ----------------------------------------------
neighbor        {r_max * 0.5:.6g} bin
neigh_modify    delay 0

# -- Contact model: Hertz-Mindlin -------------------------------------------
pair_style      gran model hertz tangential history
pair_coeff      * *

# -- Material properties ----------------------------------------------------
fix             m1 all property/global youngsModulus peratomtype {cm.youngs_modulus:.6g}
fix             m2 all property/global poissonsRatio peratomtype {cm.poisson_ratio:.6g}
fix             m3 all property/global coefficientRestitution peratomtypepair 1 1 {cm.restitution_coefficient:.6g}
fix             m4 all property/global coefficientFriction peratomtypepair 1 1 {cm.friction_coefficient:.6g}

# -- Walls (fixed boundary) -------------------------------------------------
fix             wall_bot all wall/gran model hertz tangential history primitive type 1 zplane 0.0
fix             wall_top all wall/gran model hertz tangential history primitive type 1 zplane {lz:.6g}
fix             wall_xlo all wall/gran model hertz tangential history primitive type 1 xplane 0.0
fix             wall_xhi all wall/gran model hertz tangential history primitive type 1 xplane {lx:.6g}
fix             wall_ylo all wall/gran model hertz tangential history primitive type 1 yplane 0.0
fix             wall_yhi all wall/gran model hertz tangential history primitive type 1 yplane {ly:.6g}

# -- Particle insertion: log-normal diameters, discretised at its quantiles --
{self._size_distribution_block()}

fix             ins all insert/pack seed {INSERTION_SEED} distributiontemplate pdd1 \\
                    insert_every once overlapcheck yes all_in yes \\
                    vel constant 0.0 0.0 -0.1 \\
                    region domain ntry_mc 10000 \\
                    particles_in_region {gp.count}

# -- Gravity + integrator ---------------------------------------------------
fix             grav all gravity 9.80665 vector 0.0 0.0 -1.0
fix             integ all nve/sphere

# -- Timestep: 0.2 x Rayleigh for the smallest grain ------------------------
timestep        {dt:.6e}

# -- Thermo output ----------------------------------------------------------
thermo_style    custom step atoms ke vol
thermo          {dump_every}

# -- Dump: positions and velocities -----------------------------------------
dump            dmp all custom {dump_every} dump.bunkershot id type x y z vx vy vz
dump_modify     dmp sort id pad 8

# -- Run --------------------------------------------------------------------
run             {total_steps}
""")

        return input_deck_path

    # ------------------------------------------------------------------
    # Post-processing: parse dump → HDF5
    # ------------------------------------------------------------------

    def _parse_and_write(self, work_dir: Path, output_path: Path) -> None:
        """Parse LIGGGHTS dump files and write results via *BunkerShotResultWriter*.

        LIGGGHTS dump format (custom)::

            ITEM: TIMESTEP
            <step>
            ITEM: NUMBER OF ATOMS
            <n>
            ITEM: BOX BOUNDS ...
            ...
            ITEM: ATOMS id type x y z vx vy vz
            <id> <type> <x> <y> <z> <vx> <vy> <vz>
            ...

        Args:
            work_dir: Directory containing ``dump.bunkershot``.
            output_path: Destination HDF5 file.
        """
        dt = self.integration_timestep()
        output = self.config.output
        downsample = output.downsample_grains

        dump_file = work_dir / "dump.bunkershot"

        writer = BunkerShotResultWriter(output_path, manifest=self.run_manifest())
        try:
            for timestep, positions, velocities in _iter_dump_frames(dump_file):
                time = timestep * dt

                # Apply downsampling
                if downsample > 1:
                    positions = positions[::downsample]
                    velocities = velocities[::downsample]

                writer.write_grain_state(time, positions, velocities)
        finally:
            writer.close()


# ---------------------------------------------------------------------------
# Utility: dump-file parser
# ---------------------------------------------------------------------------


def _iter_dump_frames(
    dump_path: Path,
) -> collections.abc.Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Yield ``(timestep, positions, velocities)`` for each frame in a LIGGGHTS dump.

    Args:
        dump_path: Path to the LIGGGHTS dump file (custom per-atom format).

    Yields:
        Tuple of (timestep: int, positions: ndarray shape (N,3),
        velocities: ndarray shape (N,3)).
    """

    with open(dump_path, encoding="utf-8") as fh:
        while True:
            # -- ITEM: TIMESTEP ----------------------------------------------
            line = fh.readline()
            if not line:
                return
            # Skip blank lines between frames
            while line.strip() == "":
                line = fh.readline()
                if not line:
                    return
            if "TIMESTEP" not in line:
                continue
            timestep = int(fh.readline().strip())

            # -- ITEM: NUMBER OF ATOMS ---------------------------------------
            fh.readline()  # "ITEM: NUMBER OF ATOMS"
            n_atoms = int(fh.readline().strip())

            # -- ITEM: BOX BOUNDS (3 lines of header + 3 lines of data) ------
            fh.readline()  # "ITEM: BOX BOUNDS ..."
            fh.readline()
            fh.readline()
            fh.readline()

            # -- ITEM: ATOMS -------------------------------------------------
            fh.readline()  # "ITEM: ATOMS id type x y z vx vy vz"

            positions = np.empty((n_atoms, 3), dtype=np.float64)
            velocities = np.empty((n_atoms, 3), dtype=np.float64)

            for i in range(n_atoms):
                parts = fh.readline().split()
                # id type x y z vx vy vz
                positions[i, 0] = float(parts[2])
                positions[i, 1] = float(parts[3])
                positions[i, 2] = float(parts[4])
                velocities[i, 0] = float(parts[5])
                velocities[i, 1] = float(parts[6])
                velocities[i, 2] = float(parts[7])

            yield timestep, positions, velocities
