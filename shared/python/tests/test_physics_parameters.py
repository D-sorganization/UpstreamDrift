import json

import pytest

from shared.python.physics_parameters import (
    ParameterCategory,
    PhysicsParameter,
    PhysicsParameterRegistry,
    get_registry,
)


class TestPhysicsParameters:
    @pytest.fixture
    def registry(self):
        """Create a new registry for each test."""
        return PhysicsParameterRegistry()

    def test_parameter_validation(self):
        """Test parameter validation logic."""
        param = PhysicsParameter(
            name="TEST_PARAM",
            value=10.0,
            unit="m",
            category=ParameterCategory.SIMULATION,
            description="Test parameter",
            source="Test",
            min_value=0.0,
            max_value=20.0,
            is_constant=False,
        )

        # Valid value
        valid, msg = param.validate(5.0)
        assert valid
        assert msg == ""

        # Invalid type
        valid, msg = param.validate("string")
        assert not valid
        assert "numeric" in msg

        # Out of range (min)
        valid, msg = param.validate(-1.0)
        assert not valid
        assert "must be >=" in msg

        # Out of range (max)
        valid, msg = param.validate(21.0)
        assert not valid
        assert "must be <=" in msg

        # Constant
        param.is_constant = True
        valid, msg = param.validate(5.0)
        assert not valid
        assert "constant" in msg

    def test_registry_initialization(self, registry):
        """Test that default parameters are loaded."""
        assert len(registry.parameters) > 0

        # Check specific parameters
        assert registry.get("GRAVITY") is not None
        assert registry.get("BALL_MASS") is not None
        assert registry.get("CLUB_LENGTH") is not None

    def test_get_set_parameter(self, registry):
        """Test getting and setting parameters."""
        # Get
        param = registry.get("CLUB_LENGTH")
        assert param is not None
        initial_value = param.value

        # Set valid
        new_value = initial_value + 0.05
        success, msg = registry.set("CLUB_LENGTH", new_value)
        assert success
        assert registry.get("CLUB_LENGTH").value == new_value

        # Set invalid (out of bounds)
        success, msg = registry.set("CLUB_LENGTH", 100.0)  # Too long
        assert not success
        assert registry.get("CLUB_LENGTH").value == new_value  # Should not change

        # Set non-existent
        success, msg = registry.set("NON_EXISTENT", 1.0)
        assert not success
        assert "not found" in msg

    def test_get_by_category(self, registry):
        """Test filtering by category."""
        ball_params = registry.get_by_category(ParameterCategory.BALL)
        assert len(ball_params) > 0
        for p in ball_params:
            assert p.category == ParameterCategory.BALL

        env_params = registry.get_by_category(ParameterCategory.ENVIRONMENT)
        assert len(env_params) > 0

    def test_export_import_json(self, registry, tmp_path):
        """Test JSON export and import."""
        # Export
        json_path = tmp_path / "params.json"
        registry.export_to_json(json_path)
        assert json_path.exists()

        # Modify file to verify import
        with open(json_path) as f:
            data = json.load(f)

        # Change a value
        data["CLUB_LENGTH"]["value"] = 1.0  # 1 meter

        with open(json_path, "w") as f:
            json.dump(data, f)

        # Import
        count = registry.import_from_json(json_path)
        assert count > 0
        assert registry.get("CLUB_LENGTH").value == 1.0

    def test_get_summary(self, registry):
        """Test summary string generation."""
        summary = registry.get_summary()
        assert "Physics Parameter Registry" in summary
        assert "GRAVITY" in summary
        assert "BALL_MASS" in summary

    def test_global_registry(self):
        """Test singleton accessor."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
