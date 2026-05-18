import xml.etree.ElementTree as ET

from scripts.remove_icon_backdrops import process_svgs

SVG_NS = "http://www.w3.org/2000/svg"


def _rects(path):
    root = ET.parse(path).getroot()
    return [
        element for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "rect"
    ]


def _shadow_wrappers(path):
    root = ET.parse(path).getroot()
    return [
        element
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "g"
        and element.get("filter") == "url(#drop-shadow)"
    ]


def test_process_svgs_removes_full_canvas_24px_backdrop(tmp_path):
    svg_path = tmp_path / "opensim.svg"
    svg_path.write_text(
        f"""<svg xmlns="{SVG_NS}" viewBox="0 0 24 24" width="24" height="24">
  <g>
    <rect width="24" height="24" rx="4" fill="#1a1a2e" />
    <circle cx="12" cy="12" r="4" fill="#008080" />
  </g>
</svg>""",
        encoding="utf-8",
    )

    process_svgs(str(tmp_path))

    rects = _rects(svg_path)
    assert rects == []
    assert len(_shadow_wrappers(svg_path)) == 1


def test_process_svgs_preserves_non_backdrop_shape_rects(tmp_path):
    svg_path = tmp_path / "matlab_logo.svg"
    svg_path.write_text(
        f"""<svg xmlns="{SVG_NS}" viewBox="0 0 64 64" fill="none">
  <rect x="16" y="16" width="32" height="32" rx="4" fill="#f57c00" />
  <text x="32" y="35">M</text>
</svg>""",
        encoding="utf-8",
    )

    process_svgs(str(tmp_path))

    rects = _rects(svg_path)
    assert len(rects) == 1
    assert rects[0].get("width") == "32"
    assert rects[0].get("height") == "32"
    assert len(_shadow_wrappers(svg_path)) == 1


def test_process_svgs_is_idempotent_for_existing_shadow_wrapper(tmp_path):
    svg_path = tmp_path / "project_map.svg"
    svg_path.write_text(
        f"""<svg xmlns="{SVG_NS}" viewBox="0 0 24 24" width="24" height="24">
  <defs>
    <filter id="drop-shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="4" stdDeviation="4" />
    </filter>
  </defs>
  <g filter="url(#drop-shadow)">
    <rect width="24" height="24" fill="#1a1a2e" />
    <circle cx="12" cy="12" r="3" fill="#4A90D9" />
  </g>
</svg>""",
        encoding="utf-8",
    )

    process_svgs(str(tmp_path))
    process_svgs(str(tmp_path))

    rects = _rects(svg_path)
    assert rects == []
    assert len(_shadow_wrappers(svg_path)) == 1
