"""Unit tests for the models API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.models import _parse_urdf, router

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the models router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_list_models(client: TestClient) -> None:
    """Test listing models."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) > 0
    # simple_pendulum should be in the list
    model_names = [m["name"] for m in data["models"]]
    assert "simple_pendulum" in model_names


def test_get_model_urdf(client: TestClient) -> None:
    """Test getting parsed URDF data."""
    response = client.get("/models/simple_pendulum/urdf")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "simple_pendulum"
    assert "links" in data
    assert "joints" in data
    assert "root_link" in data
    assert "urdf_raw" in data


def test_get_model_urdf_not_found(client: TestClient) -> None:
    """Test getting parsed URDF data for non-existent model."""
    response = client.get("/models/unknown_model/urdf")
    assert response.status_code == 404


def test_get_model_urdf_basename_fallback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disambiguated 'dir/name' entry resolves by exact basename."""
    import src.api.routes.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda: [
            {"name": "sub/widget", "format": "urdf", "path": "missing/widget.urdf"},
        ],
    )
    # Exact basename "widget" matches the single "sub/widget" entry. The file
    # does not exist, so resolution gets past the name lookup and 404s on the
    # missing file (deterministic basename match, not a substring guess).
    response = client.get("/models/widget/urdf")
    assert response.status_code == 404
    assert "file not found" in response.json()["detail"].lower()


def test_get_model_urdf_ambiguous_basename(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two entries sharing a basename yield a 404 listing the candidates."""
    import src.api.routes.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda: [
            {"name": "a/widget", "format": "urdf", "path": "a/widget.urdf"},
            {"name": "b/widget", "format": "urdf", "path": "b/widget.urdf"},
        ],
    )
    response = client.get("/models/widget/urdf")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "ambiguous" in detail.lower()
    assert "a/widget" in detail and "b/widget" in detail


def test_get_model_urdf_no_substring_match(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A substring of a model name no longer resolves (deterministic match)."""
    import src.api.routes.models as models_mod

    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda: [
            {"name": "simple_pendulum", "format": "urdf", "path": "x/sp.urdf"},
        ],
    )
    # "pendulum" used to match via the old substring fallback; now it 404s.
    response = client.get("/models/pendulum/urdf")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_parse_urdf_rejects_non_numeric_float_attribute() -> None:
    """URDF scalar parsing fails closed on non-numeric float attributes."""
    urdf = """<robot name="bad_radius">
      <link name="base">
        <visual>
          <geometry><cylinder radius="wide" length="0.3"/></geometry>
        </visual>
      </link>
    </robot>"""

    with pytest.raises(ValueError, match="cylinder radius"):
        _parse_urdf(urdf)


@pytest.mark.parametrize(
    ("urdf", "message"),
    [
        (
            """<robot name="empty_xyz">
              <link name="base">
                <visual>
                  <origin xyz="" rpy="0 0 0"/>
                  <geometry><box size="1 1 1"/></geometry>
                </visual>
              </link>
            </robot>""",
            "visual origin xyz",
        ),
        (
            """<robot name="short_rpy">
              <link name="base">
                <visual>
                  <origin xyz="0 0 0" rpy="0 0"/>
                  <geometry><box size="1 1 1"/></geometry>
                </visual>
              </link>
            </robot>""",
            "visual origin rpy",
        ),
        (
            """<robot name="empty_axis">
              <link name="base"/>
              <link name="tip"/>
              <joint name="hinge" type="revolute">
                <parent link="base"/>
                <child link="tip"/>
                <axis xyz=""/>
              </joint>
            </robot>""",
            "joint axis xyz",
        ),
        (
            """<robot name="short_rgba">
              <link name="base">
                <visual>
                  <geometry><box size="1 1 1"/></geometry>
                  <material name="bad"><color rgba="1 0 0"/></material>
                </visual>
              </link>
            </robot>""",
            "color rgba",
        ),
    ],
)
def test_parse_urdf_rejects_empty_or_short_vectors(urdf: str, message: str) -> None:
    """URDF vector attributes must have their exact component counts."""
    with pytest.raises(ValueError, match=message):
        _parse_urdf(urdf)
