#!/usr/bin/env python3
"""Pipe database with standard sizes and material properties.

Contains comprehensive pipe specifications following ASME/ANSI standards
and material roughness values from established references.

References:
    - ASME B36.10M-2015: Welded and Seamless Wrought Steel Pipe
    - ASME B36.19M-2004: Stainless Steel Pipe
    - Crane Technical Paper No. 410
    - Moody, L.F. (1944): "Friction factors for pipe flow"
    - Colebrook, C.F. (1939): "Turbulent flow in pipes"
"""

from ..models.pressure_drop_data_models import PipeSpecification

# ============================================================================
# MATERIAL ROUGHNESS VALUES
# ============================================================================

MATERIAL_ROUGHNESS = {
    # Material: (roughness_mm, roughness_ft, description)
    # Sources: Moody (1944), Colebrook (1939), Crane TP-410
    # Steel and Iron
    "Commercial Steel": (0.045, 0.00015, "New commercial steel pipe"),
    "New Steel": (0.025, 0.00008, "New steel pipe, smooth"),
    "Drawn Tubing": (0.0015, 0.000005, "Drawn tubing, very smooth"),
    "Cast Iron": (0.26, 0.00085, "Uncoated cast iron"),
    "Galvanized Iron": (0.15, 0.0005, "Galvanized iron"),
    "Wrought Iron": (0.045, 0.00015, "Wrought iron"),
    "Rusted Steel": (0.4, 0.0013, "Rusted/corroded steel"),
    "Severely Rusted Steel": (3.0, 0.01, "Severely rusted/corroded steel"),
    # Stainless Steel
    "Stainless Steel": (0.015, 0.00005, "Polished stainless steel"),
    "Stainless Steel 304": (0.015, 0.00005, "SS304, electropolished"),
    "Stainless Steel 316": (0.015, 0.00005, "SS316, electropolished"),
    # Non-ferrous metals
    "Copper": (0.0015, 0.000005, "Copper tubing"),
    "Brass": (0.0015, 0.000005, "Brass tubing"),
    "Aluminum": (0.015, 0.00005, "Aluminum pipe"),
    # Plastic and composites
    "PVC": (0.0015, 0.000005, "PVC pipe, smooth"),
    "Fiberglass": (0.005, 0.000016, "Fiberglass reinforced plastic"),
    "HDPE": (0.001, 0.000003, "High-density polyethylene"),
    # Concrete and masonry
    "Concrete": (0.3, 0.001, "Concrete, good finish"),
    "Concrete Rough": (3.0, 0.01, "Concrete, rough finish"),
    # Glass and ceramic
    "Glass": (0.0003, 0.000001, "Glass pipe, very smooth"),
    "Ceramic": (0.0015, 0.000005, "Ceramic lined"),
    # Special applications
    "Bituminous Lined": (0.12, 0.0004, "Bituminous lined steel"),
    "Cement Lined": (0.12, 0.0004, "Cement lined steel"),
}


def get_roughness(material: str, unit: str = "m") -> float:
    """Get roughness value for a material.

    Args:
        material: Material name
        unit: 'm', 'mm', or 'ft'

    Returns:
        Roughness value in specified unit

    Raises:
        ValueError: If material or unit not found
    """
    if material not in MATERIAL_ROUGHNESS:
        raise ValueError(f"Material '{material}' not found in database")

    roughness_mm, roughness_ft, _ = MATERIAL_ROUGHNESS[material]

    if unit == "m":
        return roughness_mm / 1000.0
    if unit == "mm":
        return roughness_mm
    if unit == "ft":
        return roughness_ft
    raise ValueError(f"Unit '{unit}' not recognized. Use 'm', 'mm', or 'ft'")


# ============================================================================
# STANDARD PIPE SIZES (ASME B36.10M)
# ============================================================================

# Format: (NPS, Schedule): (OD_mm, Wall_mm, ID_mm)
STEEL_PIPE_DIMENSIONS = {
    # NPS 1/2"
    ("1/2", "5S"): (21.3, 1.65, 17.98),
    ("1/2", "10S"): (21.3, 2.11, 17.08),
    ("1/2", "40"): (21.3, 2.77, 15.76),
    ("1/2", "STD"): (21.3, 2.77, 15.76),
    ("1/2", "80"): (21.3, 3.73, 13.84),
    ("1/2", "XS"): (21.3, 3.73, 13.84),
    ("1/2", "160"): (21.3, 4.78, 11.74),
    ("1/2", "XXS"): (21.3, 7.47, 6.36),
    # NPS 3/4"
    ("3/4", "5S"): (26.7, 1.65, 23.40),
    ("3/4", "10S"): (26.7, 2.11, 22.48),
    ("3/4", "40"): (26.7, 2.87, 20.96),
    ("3/4", "STD"): (26.7, 2.87, 20.96),
    ("3/4", "80"): (26.7, 3.91, 18.88),
    ("3/4", "XS"): (26.7, 3.91, 18.88),
    ("3/4", "160"): (26.7, 5.56, 15.58),
    ("3/4", "XXS"): (26.7, 7.82, 11.06),
    # NPS 1"
    ("1", "5S"): (33.4, 1.65, 30.10),
    ("1", "10S"): (33.4, 2.77, 27.86),
    ("1", "40"): (33.4, 3.38, 26.64),
    ("1", "STD"): (33.4, 3.38, 26.64),
    ("1", "80"): (33.4, 4.55, 24.30),
    ("1", "XS"): (33.4, 4.55, 24.30),
    ("1", "160"): (33.4, 6.35, 20.70),
    ("1", "XXS"): (33.4, 9.09, 15.22),
    # NPS 1.5"
    ("1.5", "5S"): (48.3, 1.65, 45.00),
    ("1.5", "10S"): (48.3, 2.77, 42.76),
    ("1.5", "40"): (48.3, 3.68, 40.94),
    ("1.5", "STD"): (48.3, 3.68, 40.94),
    ("1.5", "80"): (48.3, 5.08, 38.14),
    ("1.5", "XS"): (48.3, 5.08, 38.14),
    ("1.5", "160"): (48.3, 7.14, 34.02),
    ("1.5", "XXS"): (48.3, 10.15, 28.00),
    # NPS 2"
    ("2", "5S"): (60.3, 1.65, 57.00),
    ("2", "10S"): (60.3, 2.77, 54.76),
    ("2", "40"): (60.3, 3.91, 52.48),
    ("2", "STD"): (60.3, 3.91, 52.48),
    ("2", "80"): (60.3, 5.54, 49.22),
    ("2", "XS"): (60.3, 5.54, 49.22),
    ("2", "160"): (60.3, 8.74, 42.82),
    ("2", "XXS"): (60.3, 11.07, 38.16),
    # NPS 3"
    ("3", "5S"): (88.9, 2.11, 84.68),
    ("3", "10S"): (88.9, 3.05, 82.80),
    ("3", "40"): (88.9, 5.49, 77.92),
    ("3", "STD"): (88.9, 5.49, 77.92),
    ("3", "80"): (88.9, 7.62, 73.66),
    ("3", "XS"): (88.9, 7.62, 73.66),
    ("3", "160"): (88.9, 11.13, 66.64),
    ("3", "XXS"): (88.9, 15.24, 58.42),
    # NPS 4"
    ("4", "5S"): (114.3, 2.11, 110.08),
    ("4", "10S"): (114.3, 3.05, 108.20),
    ("4", "40"): (114.3, 6.02, 102.26),
    ("4", "STD"): (114.3, 6.02, 102.26),
    ("4", "80"): (114.3, 8.56, 97.18),
    ("4", "XS"): (114.3, 8.56, 97.18),
    ("4", "120"): (114.3, 11.13, 92.04),
    ("4", "160"): (114.3, 13.49, 87.32),
    ("4", "XXS"): (114.3, 17.12, 80.06),
    # NPS 6"
    ("6", "5S"): (168.3, 2.77, 162.76),
    ("6", "10S"): (168.3, 3.40, 161.50),
    ("6", "40"): (168.3, 7.11, 154.08),
    ("6", "STD"): (168.3, 7.11, 154.08),
    ("6", "80"): (168.3, 10.97, 146.36),
    ("6", "XS"): (168.3, 10.97, 146.36),
    ("6", "120"): (168.3, 14.27, 139.76),
    ("6", "160"): (168.3, 18.26, 131.78),
    ("6", "XXS"): (168.3, 21.95, 124.40),
    # NPS 8"
    ("8", "5S"): (219.1, 2.77, 213.56),
    ("8", "10S"): (219.1, 3.76, 211.58),
    ("8", "20"): (219.1, 6.35, 206.40),
    ("8", "30"): (219.1, 7.04, 205.02),
    ("8", "40"): (219.1, 8.18, 202.74),
    ("8", "STD"): (219.1, 8.18, 202.74),
    ("8", "60"): (219.1, 10.31, 198.48),
    ("8", "80"): (219.1, 12.70, 193.70),
    ("8", "XS"): (219.1, 12.70, 193.70),
    ("8", "100"): (219.1, 15.09, 188.92),
    ("8", "120"): (219.1, 18.26, 182.58),
    ("8", "140"): (219.1, 20.62, 177.86),
    ("8", "160"): (219.1, 22.23, 174.64),
    ("8", "XXS"): (219.1, 23.01, 173.08),
    # NPS 10"
    ("10", "5S"): (273.0, 3.40, 266.20),
    ("10", "10S"): (273.0, 4.19, 264.62),
    ("10", "20"): (273.0, 6.35, 260.30),
    ("10", "30"): (273.0, 7.80, 257.40),
    ("10", "40"): (273.0, 9.27, 254.46),
    ("10", "STD"): (273.0, 9.27, 254.46),
    ("10", "60"): (273.0, 12.70, 247.60),
    ("10", "XS"): (273.0, 12.70, 247.60),
    ("10", "80"): (273.0, 15.09, 242.82),
    ("10", "100"): (273.0, 18.26, 236.48),
    ("10", "120"): (273.0, 21.44, 230.12),
    ("10", "140"): (273.0, 25.40, 222.20),
    ("10", "160"): (273.0, 28.58, 215.84),
    ("10", "XXS"): (273.0, 25.40, 222.20),
    # NPS 12"
    ("12", "5S"): (323.8, 3.96, 315.88),
    ("12", "10S"): (323.8, 4.57, 314.66),
    ("12", "20"): (323.8, 6.35, 311.10),
    ("12", "STD"): (323.8, 9.53, 304.74),
    ("12", "30"): (323.8, 8.38, 307.04),
    ("12", "40"): (323.8, 10.31, 303.18),
    ("12", "XS"): (323.8, 12.70, 298.40),
    ("12", "60"): (323.8, 14.27, 295.26),
    ("12", "80"): (323.8, 17.48, 288.84),
    ("12", "100"): (323.8, 21.44, 280.92),
    ("12", "120"): (323.8, 25.40, 273.00),
    ("12", "140"): (323.8, 28.58, 266.64),
    ("12", "160"): (323.8, 33.32, 257.16),
    ("12", "XXS"): (323.8, 25.40, 273.00),
    # NPS 14"
    ("14", "5S"): (355.6, 3.96, 347.68),
    ("14", "10S"): (355.6, 4.78, 346.04),
    ("14", "10"): (355.6, 6.35, 342.90),
    ("14", "20"): (355.6, 7.92, 339.76),
    ("14", "STD"): (355.6, 9.53, 336.54),
    ("14", "30"): (355.6, 9.53, 336.54),
    ("14", "40"): (355.6, 11.13, 333.34),
    ("14", "XS"): (355.6, 12.70, 330.20),
    ("14", "60"): (355.6, 15.09, 325.42),
    ("14", "80"): (355.6, 19.05, 317.50),
    ("14", "100"): (355.6, 23.01, 309.58),
    ("14", "120"): (355.6, 26.19, 303.22),
    ("14", "140"): (355.6, 29.36, 296.88),
    ("14", "160"): (355.6, 31.75, 292.10),
    # NPS 16"
    ("16", "5S"): (406.4, 4.19, 398.02),
    ("16", "10S"): (406.4, 4.78, 396.84),
    ("16", "10"): (406.4, 6.35, 393.70),
    ("16", "20"): (406.4, 7.92, 390.56),
    ("16", "STD"): (406.4, 9.53, 387.34),
    ("16", "30"): (406.4, 9.53, 387.34),
    ("16", "40"): (406.4, 12.70, 381.00),
    ("16", "XS"): (406.4, 12.70, 381.00),
    ("16", "60"): (406.4, 16.66, 373.08),
    ("16", "80"): (406.4, 21.44, 363.52),
    ("16", "100"): (406.4, 26.19, 354.02),
    ("16", "120"): (406.4, 30.96, 344.48),
    ("16", "140"): (406.4, 36.53, 333.34),
    ("16", "160"): (406.4, 40.49, 325.42),
    # NPS 18"
    ("18", "5S"): (457.0, 4.19, 448.62),
    ("18", "10S"): (457.0, 4.78, 447.44),
    ("18", "10"): (457.0, 6.35, 444.30),
    ("18", "20"): (457.0, 7.92, 441.16),
    ("18", "STD"): (457.0, 9.53, 437.94),
    ("18", "30"): (457.0, 11.13, 434.74),
    ("18", "40"): (457.0, 14.27, 428.46),
    ("18", "XS"): (457.0, 14.27, 428.46),
    ("18", "60"): (457.0, 19.05, 418.90),
    ("18", "80"): (457.0, 23.83, 409.34),
    ("18", "100"): (457.0, 29.36, 398.28),
    ("18", "120"): (457.0, 34.93, 387.14),
    ("18", "140"): (457.0, 39.67, 377.66),
    ("18", "160"): (457.0, 45.24, 366.52),
    # NPS 20"
    ("20", "5S"): (508.0, 4.78, 498.44),
    ("20", "10S"): (508.0, 5.54, 496.92),
    ("20", "10"): (508.0, 6.35, 495.30),
    ("20", "20"): (508.0, 9.53, 488.94),
    ("20", "STD"): (508.0, 9.53, 488.94),
    ("20", "30"): (508.0, 12.70, 482.60),
    ("20", "XS"): (508.0, 15.09, 477.82),
    ("20", "40"): (508.0, 15.09, 477.82),
    ("20", "60"): (508.0, 20.62, 466.76),
    ("20", "80"): (508.0, 26.19, 455.62),
    ("20", "100"): (508.0, 32.54, 442.92),
    ("20", "120"): (508.0, 38.10, 431.80),
    ("20", "140"): (508.0, 44.45, 419.10),
    ("20", "160"): (508.0, 50.01, 407.98),
    # NPS 24"
    ("24", "5S"): (610.0, 5.54, 598.92),
    ("24", "10S"): (610.0, 6.35, 597.30),
    ("24", "10"): (610.0, 6.35, 597.30),
    ("24", "20"): (610.0, 9.53, 590.94),
    ("24", "STD"): (610.0, 9.53, 590.94),
    ("24", "XS"): (610.0, 17.48, 575.04),
    ("24", "30"): (610.0, 14.27, 581.46),
    ("24", "40"): (610.0, 17.48, 575.04),
    ("24", "60"): (610.0, 24.61, 560.78),
    ("24", "80"): (610.0, 31.75, 546.50),
    ("24", "100"): (610.0, 38.89, 532.22),
    ("24", "120"): (610.0, 46.02, 517.96),
    ("24", "140"): (610.0, 52.37, 505.26),
    ("24", "160"): (610.0, 59.51, 490.98),
}


def get_pipe_spec(
    nominal_size: str, schedule: str, material: str = "Commercial Steel"
) -> PipeSpecification:
    """Get pipe specification from database.

    Args:
        nominal_size: Nominal pipe size (e.g., "2", "4", "6")
        schedule: Pipe schedule (e.g., "40", "80", "160", "STD", "XS")
        material: Pipe material (default: "Commercial Steel")

    Returns:
        PipeSpecification object

    Raises:
        ValueError: If pipe size/schedule combination not found

    Example:
        >>> spec = get_pipe_spec("4", "40")
        >>> print(spec.inner_diameter)  # mm
        102.26
    """
    key = (nominal_size, schedule)
    if key not in STEEL_PIPE_DIMENSIONS:
        raise ValueError(
            f'Pipe size {nominal_size}" Schedule {schedule} not found in database'
        )

    od, wall, id_val = STEEL_PIPE_DIMENSIONS[key]

    return PipeSpecification(
        nominal_size=nominal_size,
        schedule=schedule,
        outer_diameter=od,
        wall_thickness=wall,
        inner_diameter=id_val,
        material=material,
    )


def list_available_sizes() -> list[str]:
    """List all available nominal pipe sizes."""
    sizes = sorted(
        {nps for nps, _ in STEEL_PIPE_DIMENSIONS},
        key=lambda x: float(x.replace("/", ".")) if "/" in x else float(x),
    )
    return sizes


def list_schedules_for_size(nominal_size: str) -> list[str]:
    """List all available schedules for a given nominal size."""
    schedules = [sch for nps, sch in STEEL_PIPE_DIMENSIONS if nps == nominal_size]
    return sorted(schedules, key=lambda x: "000" if x in ["STD", "XS", "XXS"] else x)


def create_custom_pipe(
    inner_diameter_mm: float, material: str = "Commercial Steel"
) -> PipeSpecification:
    """Create custom pipe specification.

    Args:
        inner_diameter_mm: Inner diameter in millimeters
        material: Pipe material

    Returns:
        PipeSpecification for custom pipe
    """
    return PipeSpecification(
        nominal_size="Custom",
        schedule="Custom",
        outer_diameter=inner_diameter_mm,  # Assume thin wall for custom
        wall_thickness=0.0,
        inner_diameter=inner_diameter_mm,
        material=material,
    )
