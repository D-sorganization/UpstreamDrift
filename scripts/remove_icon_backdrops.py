import glob
import os
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
SHADOW_FILTER_ID = "drop-shadow"


def _local_name(tag):
    return tag.rsplit("}", 1)[-1]


def _parse_length(value):
    if value is None:
        return None
    value = value.strip()
    if value.endswith("%"):
        return 1.0 if value == "100%" else None
    for suffix in ("px", "pt"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    try:
        return float(value)
    except ValueError:
        return None


def _canvas_dimensions(root):
    width = _parse_length(root.get("width"))
    height = _parse_length(root.get("height"))
    view_box = root.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            try:
                width = float(parts[2])
                height = float(parts[3])
            except ValueError:
                pass
    return width, height


def _matches_origin(value):
    parsed = _parse_length(value)
    return parsed is None or parsed == 0


def _matches_canvas_length(value, canvas_length):
    parsed = _parse_length(value)
    if parsed is None:
        return False
    if parsed == 1.0 and str(value).strip().endswith("%"):
        return True
    return canvas_length is not None and abs(parsed - canvas_length) < 0.001


def _is_full_canvas_rect(rect, root):
    canvas_width, canvas_height = _canvas_dimensions(root)
    return (
        _matches_origin(rect.get("x"))
        and _matches_origin(rect.get("y"))
        and _matches_canvas_length(rect.get("width"), canvas_width)
        and _matches_canvas_length(rect.get("height"), canvas_height)
    )


def _has_shadow_filter(defs):
    return any(
        _local_name(child.tag) == "filter" and child.get("id") == SHADOW_FILTER_ID
        for child in list(defs)
    )


def _is_shadow_wrapped(elements):
    return (
        len(elements) == 1
        and _local_name(elements[0].tag) == "g"
        and elements[0].get("filter") == f"url(#{SHADOW_FILTER_ID})"
    )


def process_svgs(directory):
    ET.register_namespace("", SVG_NS)

    for filepath in glob.glob(os.path.join(directory, "*.svg")):
        filename = os.path.basename(filepath)
        if filename == "drake.svg":
            continue

        print(f"Processing {filename}...")
        tree = ET.parse(filepath)
        root = tree.getroot()

        # 1. Remove background rects
        for parent in root.iter():
            if _local_name(parent.tag) == "defs":
                continue
            for child in list(parent):
                if _local_name(child.tag) == "rect" and _is_full_canvas_rect(
                    child, root
                ):
                    parent.remove(child)
                    print(f"  Removed background rect from {filename}")

        # 2. Add defs for shadow if not present
        defs = None
        for child in list(root):
            if _local_name(child.tag) == "defs":
                defs = child
                break
        if defs is None:
            defs = ET.Element(f"{{{SVG_NS}}}defs")
            root.insert(0, defs)

        if not _has_shadow_filter(defs):
            shadow_clone = ET.fromstring("""
            <filter xmlns="http://www.w3.org/2000/svg" id="drop-shadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="2" dy="4" stdDeviation="4" flood-color="#000000" flood-opacity="0.5"/>
            </filter>
            """)
            defs.append(shadow_clone)
            print(f"  Added drop shadow filter to {filename}")

        # 3. Apply the shadow to the main graphic group
        elements_to_wrap = []
        for child in list(root):
            if _local_name(child.tag) not in {"defs", "metadata", "title", "desc"}:
                elements_to_wrap.append(child)

        if elements_to_wrap and not _is_shadow_wrapped(elements_to_wrap):
            for el in elements_to_wrap:
                root.remove(el)
            wrapper_g = ET.Element(f"{{{SVG_NS}}}g")
            wrapper_g.set("filter", f"url(#{SHADOW_FILTER_ID})")
            for el in elements_to_wrap:
                wrapper_g.append(el)
            root.append(wrapper_g)
            print(f"  Applied shadow wrapper to {filename}")

        tree.write(filepath, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    logos_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "assets", "logos")
    )
    process_svgs(logos_dir)
