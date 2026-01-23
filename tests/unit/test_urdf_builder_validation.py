"""Tests for URDF builder physical parameter validation."""

import pytest
from tools.urdf_generator.urdf_builder import URDFBuilder


class TestPhysicalValidation:
    """Test physical parameter validation in URDF builder."""

    def test_positive_mass_required(self):
        """Mass must be positive."""
        builder = URDFBuilder()

        # Negative mass should raise
        with pytest.raises(ValueError, match="Mass must be positive"):
            builder.add_segment(
                {
                    "name": "test_segment",
                    "geometry": {"shape": "box", "dimensions": {}},
                    "physics": {"mass": -1.0},
                }
            )

        # Zero mass should raise
        with pytest.raises(ValueError, match="Mass must be positive"):
            builder.add_segment(
                {
                    "name": "test_segment",
                    "geometry": {"shape": "box", "dimensions": {}},
                    "physics": {"mass": 0.0},
                }
            )

    def test_positive_inertia_diagonal_required(self):
        """Inertia diagonal elements must be positive."""
        builder = URDFBuilder()

        # Negative Ixx
        with pytest.raises(
            ValueError, match="Inertia diagonal elements must be positive"
        ):
            builder.add_segment(
                {
                    "name": "test_segment",
                    "geometry": {"shape": "box", "dimensions": {}},
                    "physics": {
                        "mass": 1.0,
                        "inertia": {"ixx": -0.1, "iyy": 0.1, "izz": 0.1},
                    },
                }
            )

        # Zero Iyy
        with pytest.raises(
            ValueError, match="Inertia diagonal elements must be positive"
        ):
            builder.add_segment(
                {
                    "name": "test_segment",
                    "geometry": {"shape": "box", "dimensions": {}},
                    "physics": {
                        "mass": 1.0,
                        "inertia": {"ixx": 0.1, "iyy": 0.0, "izz": 0.1},
                    },
                }
            )

    def test_positive_definite_inertia_required(self):
        """Inertia matrix must be positive-definite."""
        builder = URDFBuilder()

        # This matrix is symmetric but not positive-definite
        # (negative eigenvalue)
        with pytest.raises(ValueError, match="must be positive-definite"):
            builder.add_segment(
                {
                    "name": "test_segment",
                    "geometry": {"shape": "box", "dimensions": {}},
                    "physics": {
                        "mass": 1.0,
                        "inertia": {
                            "ixx": 1.0,
                            "iyy": 1.0,
                            "izz": 1.0,
                            "ixy": 2.0,  # Off-diagonal too large
                            "ixz": 0.0,
                            "iyz": 0.0,
                        },
                    },
                }
            )

    def test_valid_inertia_passes(self):
        """Valid inertia matrix should pass validation."""
        builder = URDFBuilder()

        # Simple diagonal matrix (always PD if positive)
        builder.add_segment(
            {
                "name": "test_segment",
                "geometry": {"shape": "box", "dimensions": {}},
                "physics": {
                    "mass": 1.0,
                    "inertia": {"ixx": 0.1, "iyy": 0.1, "izz": 0.1},
                },
            }
        )

        assert builder.get_segment_count() == 1

    def test_realistic_inertia_passes(self):
        """Realistic inertia from a rod should pass."""
        builder = URDFBuilder()

        # Rod along X-axis: length=1m, radius=0.05m, mass=1kg
        # Ixx = (1/2) * m * r^2 = 0.00125
        # Iyy = Izz = (1/12) * m * L^2 + (1/4) * m * r^2 = 0.0840
        builder.add_segment(
            {
                "name": "rod",
                "geometry": {
                    "shape": "cylinder",
                    "dimensions": {"length": 1.0, "width": 0.1},
                },
                "physics": {
                    "mass": 1.0,
                    "inertia": {"ixx": 0.00125, "iyy": 0.0840, "izz": 0.0840},
                },
            }
        )

        assert builder.get_segment_count() == 1
