from pathlib import Path
from bunkershot3d.geometry.clubhead import ClubheadGenerator


def test_clubhead_generator(tmp_path: Path) -> None:
    """Test generating a simple parametric wedge geometry."""
    generator = ClubheadGenerator(
        loft_deg=60.0, bounce_deg=10.0, width=0.05, height=0.04
    )

    # Generate vertices and faces
    vertices, faces = generator.generate_mesh()

    # A simple wedge should have at least 6 vertices and 8 faces for a basic block representation
    assert len(vertices) >= 6
    assert len(faces) >= 8

    # Ensure loft angle is represented in the vertices geometry
    # (very simplified check: just verify vertices are generated)
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3

    # Test STL export
    stl_path = tmp_path / "wedge.stl"
    generator.export_stl(stl_path)
    assert stl_path.exists()
    assert stl_path.stat().st_size > 0
