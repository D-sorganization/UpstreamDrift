import pytest

from src.shared.python.physics.physics_parameters import (
    ParameterCategory,
    PhysicsParameter,
    PhysicsParameterRegistry,
    get_registry,
)


@pytest.fixture
def registry():
    """Create a fresh registry for testing."""
    return PhysicsParameterRegistry()


def test_registry_initialization(registry):
    """Test registry loads defaults."""
    assert len(registry.parameters) > 0
    assert registry.get("BALL_MASS") is not None
    assert registry.get("GRAVITY") is not None


@pytest.fixture
def bounded_param():
    """Create a bounded parameter for validation tests."""
    return PhysicsParameter(
        name="TEST",
        value=10.0,
        unit="m",
        category=ParameterCategory.SIMULATION,
        description="Test",
        source="Test",
        min_value=0.0,
        max_value=20.0,
    )


@pytest.mark.parametrize(
    "test_value,expect_valid,msg_contains",
    [
        (15.0, True, None),
        ("string", False, "numeric"),
        (-1.0, False, "must be >="),
        (25.0, False, "must be <="),
    ],
    ids=["valid_in_range", "invalid_type", "below_min", "above_max"],
)
def test_parameter_validation(bounded_param, test_value, expect_valid, msg_contains):
    """Test parameter validation logic."""
    valid, msg = bounded_param.validate(test_value)
    assert valid == expect_valid
    if msg_contains is not None:
        assert msg_contains in msg


def test_constant_parameter():
    """Test constant parameter enforcement."""
    param = PhysicsParameter(
        name="CONST",
        value=1.0,
        unit="m",
        category=ParameterCategory.ENVIRONMENT,
        description="Constant",
        source="Test",
        is_constant=True,
    )

    valid, msg = param.validate(2.0)
    assert not valid
    assert "constant" in msg


def test_registry_set(registry):
    """Test setting parameters in registry."""
    # Set valid
    success, msg = registry.set("CLUB_MASS", 0.4)
    assert success
    assert registry.get("CLUB_MASS").value == 0.4

    # Set invalid (out of bounds)
    success, msg = registry.set("CLUB_MASS", 100.0)
    assert not success
    assert registry.get("CLUB_MASS").value == 0.4  # Unchanged

    # Set non-existent
    success, msg = registry.set("NON_EXISTENT", 1.0)
    assert not success


def test_get_by_category(registry):
    """Test retrieving parameters by category."""
    ball_params = registry.get_by_category(ParameterCategory.BALL)
    assert len(ball_params) > 0
    for param in ball_params:
        assert param.category == ParameterCategory.BALL


def test_export_import_json(registry, tmp_path):
    """Test exporting and importing parameters."""
    json_path = tmp_path / "params.json"

    # Modify a value
    registry.set("CLUB_MASS", 0.25)

    # Export
    registry.export_to_json(json_path)
    assert json_path.exists()

    # Create new registry and import
    new_registry = PhysicsParameterRegistry()
    # verify default is different from what we saved
    # default CLUB_MASS is 0.310
    param = new_registry.get("CLUB_MASS")
    assert param is not None and param.value == 0.310

    count = new_registry.import_from_json(json_path)
    assert count > 0
    param = new_registry.get("CLUB_MASS")
    assert param is not None and param.value == 0.25


def test_get_summary(registry):
    """Test summary generation."""
    summary = registry.get_summary()
    assert "Physics Parameter Registry" in summary
    assert "BALL_MASS" in summary
    assert "GRAVITY" in summary


def test_global_registry():
    """Test global registry singleton."""
    reg1 = get_registry()
    reg2 = get_registry()
    assert reg1 is reg2
