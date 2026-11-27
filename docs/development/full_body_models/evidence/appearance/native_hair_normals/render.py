"""Render Native OBJ Hair Shading Before and After (Refs #12678).

Run from the repository root with Python's -m option and this module's dotted
path. Requires meshcat, Playwright Chromium and Pillow; outputs under
output/playwright/native_hair_normals. Baseline strips the newly added normal
records and face-normal references, reproducing the previous OBJ writer.
"""

from pathlib import Path

import meshcat
import meshcat.geometry as g
import numpy as np
from PIL import Image

from src.tools.native_viewer_export.backends._head import head_mesh_files
from src.tools.native_viewer_export.backends._meshcat_page import MeshcatPage


def main() -> None:
    """Capture Identical Head Geometry and Materials With Both Shading Paths."""
    root = Path("output/playwright/native_hair_normals")
    files = head_mesh_files({"bodies": [{"name": "Head"}]}, root / "meshes")
    viewer = meshcat.Visualizer()
    viewer["/Grid"].set_property("visible", False)
    viewer["/Axes"].set_property("visible", False)
    viewer["/Cameras/default"].set_transform(np.eye(4))
    try:
        for stage in ("before", "after"):
            for part in files:
                text = part.path.read_text()
                if stage == "before":
                    text = "\n".join(
                        "f " + " ".join(v.split("//")[0] for v in line.split()[1:])
                        if line.startswith("f ")
                        else line
                        for line in text.splitlines()
                        if not line.startswith(("vn ", "s "))
                    )
                colour = sum(
                    int(v * 255) << shift
                    for v, shift in zip(part.rgba[:3], (16, 8, 0), strict=True)
                )
                viewer[part.name].set_object(
                    g.MeshGeometry(text, "obj"), g.MeshLambertMaterial(color=colour)
                )
            for width, height in ((1280, 844), (390, 844)):
                with MeshcatPage(viewer.url(), width, height) as page:
                    for view, position in (
                        ("front", [0.65, 0.15, 0]),
                        ("three_quarter", [0.55, 0.18, 0.35]),
                    ):
                        viewer["/Cameras/default/rotated/<object>"].set_property(
                            "position", position
                        )
                        page.look_at((0, 0.14, 0))
                        page.settle(500)
                        Image.fromarray(page.screenshot()).save(
                            root / f"{stage}_{view}_{width}.png"
                        )
    finally:
        viewer.window.server_proc.kill()


if __name__ == "__main__":
    main()
