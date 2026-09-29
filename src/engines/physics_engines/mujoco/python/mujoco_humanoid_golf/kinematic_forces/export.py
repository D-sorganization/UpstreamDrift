import csv

from .types import KinematicForceData


def export_kinematic_forces_to_csv(
    force_data_list: list[KinematicForceData],
    filepath: str,
) -> None:
    """Export kinematic force analysis to CSV file.

    Args:
        force_data_list: List of force data
        filepath: Output CSV file path
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "time",
            "coriolis_power",
            "centrifugal_power",
            "rotational_ke",
            "translational_ke",
        ]

        # Add joint-wise Coriolis forces
        nv = len(force_data_list[0].coriolis_forces)
        for i in range(nv):
            header.extend(
                [f"coriolis_force_{i}", f"gravity_force_{i}", f"centrifugal_force_{i}"],
            )

        # Add club head forces
        header.extend(
            [
                "club_coriolis_x",
                "club_coriolis_y",
                "club_coriolis_z",
                "club_centrifugal_x",
                "club_centrifugal_y",
                "club_centrifugal_z",
            ],
        )

        writer.writerow(header)

        # Data rows
        for data in force_data_list:
            row = [
                data.time,
                data.coriolis_power,
                data.centrifugal_power,
                data.rotational_kinetic_energy,
                data.translational_kinetic_energy,
            ]

            for i in range(nv):
                row.extend(
                    [
                        data.coriolis_forces[i],
                        data.gravity_forces[i],
                        (
                            data.centrifugal_forces[i]
                            if data.centrifugal_forces is not None
                            else 0.0
                        ),
                    ],
                )

            if data.club_head_coriolis_force is not None:
                row.extend(data.club_head_coriolis_force.tolist())
            else:
                row.extend([0, 0, 0])

            if data.club_head_centrifugal_force is not None:
                row.extend(data.club_head_centrifugal_force.tolist())
            else:
                row.extend([0, 0, 0])

            writer.writerow(row)
