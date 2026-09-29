from pathlib import Path
import pytest
from pydantic import ValidationError
from bunkershot3d.config import BunkerShotConfig


def test_config_parsing(tmp_path: Path) -> None:
    yaml_content = """
    bunker_bed:
      domain:
        length_x: 2.0
        width_y: 1.0
        depth_z: 0.5
      boundary: "fixed"
    grain_population:
      count: 10000
      diameter_mean: 0.002
      diameter_sigma_log: 0.1
      density: 2650.0
      coarse_graining_factor: 1.0
    contact_model:
      friction_coefficient: 0.5
      restitution_coefficient: 0.3
      youngs_modulus: 1e7
      poisson_ratio: 0.25
    clubhead:
      loft_deg: 56.0
      bounce_deg: 10.0
      width: 0.1
      height: 0.05
      mass: 0.3
    trajectory:
      file: "swing_data.csv"
    output:
      downsample_grains: 10
      rate_hz: 500.0
    """

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    config = BunkerShotConfig.from_yaml(config_file)
    assert config.bunker_bed.domain.length_x == 2.0
    assert config.contact_model.friction_coefficient == 0.5
    assert config.clubhead.loft_deg == 56.0


def test_config_validation() -> None:
    # Ensure Pydantic catches negative dimensions
    with pytest.raises(ValidationError):
        from bunkershot3d.config import DomainConfig

        DomainConfig(length_x=-1.0, width_y=1.0, depth_z=0.5)
