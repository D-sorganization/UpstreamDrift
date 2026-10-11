"""Operational PDF/SVG inspection helper for shot-pattern publication review."""

from __future__ import annotations

import hashlib
import logging
import math
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pypdf
from defusedxml import ElementTree as ET
from defusedxml import minidom as SafeDOM
from defusedxml.common import DefusedXmlException, EntitiesForbidden

_DIMENSION_REGEX = re.compile(
    r"^([+-]?(?:(?:\d*\.\d+)|\d+)(?:[eE][+-]?\d+)?)\s*(pt|px|in|mm|cm)?$"
)
_URL_REF_REGEX = re.compile(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", re.IGNORECASE)
_EXTERNAL_STYLE_REGEX = re.compile(r"@import\s+", re.IGNORECASE)


def _compute_sha256(path: Path) -> str:
    """Return SHA-256 hex digest for an existing file.

    Precondition: path must be an existing regular file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Target is not an existing file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash_before_read(path: Path, expected_sha256: str) -> str:
    """Verify file digest matches expected SHA-256 before reading."""
    actual = _compute_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Pre-read SHA256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def _rehash_after_read(path: Path, expected_sha256: str) -> str:
    """Verify file digest remains identical after reading."""
    actual = _compute_sha256(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"Post-read SHA256 mutation detected for {path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def _get_source_sha256() -> str:
    """Return SHA-256 digest of this module's source file."""
    source_path = Path(__file__).resolve()
    if source_path.is_file():
        return hashlib.sha256(source_path.read_bytes()).hexdigest()
    return ""


def _get_runtime_versions() -> dict[str, str]:
    """Return runtime and key library versions."""
    return {
        "python": platform.python_version(),
        "python_full": sys.version,
        "pypdf": pypdf.__version__,
    }


def _validate_box_coords(box: Any, box_name: str) -> dict[str, float]:
    """Validate that rectangle box coordinates are finite and positive."""
    try:
        x0 = float(box.left)
        y0 = float(box.bottom)
        x1 = float(box.right)
        y1 = float(box.top)
        width = float(box.width)
        height = float(box.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {box_name} coordinates in PDF: {exc}") from exc

    for name, val in (
        ("x0", x0),
        ("y0", y0),
        ("x1", x1),
        ("y1", y1),
        ("width", width),
        ("height", height),
    ):
        if not math.isfinite(val):
            raise ValueError(f"Non-finite value in {box_name} ({name}={val})")

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Non-positive or non-finite dimensions in {box_name}: "
            f"width={width}, height={height}"
        )

    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "width": width,
        "height": height,
    }


def _check_pdf_stream_nonempty(page: pypdf.PageObject) -> None:
    """Verify that page contains a non-empty content stream."""
    contents = page.get_contents()
    if contents is None:
        raise ValueError("PDF page stream is empty")

    if isinstance(contents, list):
        total_data = b""
        for item in contents:
            item_obj = item.get_object() if hasattr(item, "get_object") else item
            if hasattr(item_obj, "get_data"):
                total_data += item_obj.get_data()
        if not total_data.strip():
            raise ValueError("PDF page stream is empty")
    elif hasattr(contents, "get_data"):
        data = contents.get_data()
        if not data.strip():
            raise ValueError("PDF page stream is empty")


def _extract_font_resources(page: pypdf.PageObject) -> list[dict[str, str | None]]:
    """Inventory fonts declared in page resources without judging correctness."""
    resources = page.get("/Resources")
    if resources is None:
        return []

    res_dict = resources.get_object() if hasattr(resources, "get_object") else resources
    if not isinstance(res_dict, dict) or "/Font" not in res_dict:
        return []

    fonts_obj = res_dict["/Font"]
    font_dict = (
        fonts_obj.get_object() if hasattr(fonts_obj, "get_object") else fonts_obj
    )
    if not isinstance(font_dict, dict):
        return []

    inventory: list[dict[str, str | None]] = []
    for key, val in font_dict.items():
        font_data = val.get_object() if hasattr(val, "get_object") else val
        base_font: str | None = None
        subtype: str | None = None
        if isinstance(font_data, dict):
            raw_base = font_data.get("/BaseFont")
            if raw_base is not None:
                base_font = str(raw_base)
            raw_subtype = font_data.get("/Subtype")
            if raw_subtype is not None:
                subtype = str(raw_subtype)
        inventory.append(
            {
                "resource_name": str(key),
                "base_font": base_font,
                "subtype": subtype,
            }
        )

    inventory.sort(key=lambda item: item["resource_name"] or "")
    return inventory


class _ListLogHandler(logging.Handler):
    """Logging handler that collects log messages into an in-memory list."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _read_pdf_page(
    path: Path,
) -> tuple[pypdf.PageObject, pypdf.PdfReader, list[str]]:
    """Open PDF strictly; the caller captures warnings through all lazy reads."""
    try:
        reader = pypdf.PdfReader(str(path), strict=True)
    except Exception as exc:
        raise ValueError(f"Malformed PDF in {path.name}: {exc}") from exc
    if reader.is_encrypted:
        raise ValueError(f"Encrypted PDF rejected: {path.name}")
    page_count = len(reader.pages)
    if page_count != 1:
        raise ValueError(
            f"Expected exactly 1 page in PDF, found {page_count}: {path.name}"
        )
    return reader.pages[0], reader, []


def inspect_pdf(path: Path | str, expected_sha256: str) -> dict[str, Any]:
    """Inspect single-page PDF vector asset and return diagnostics."""
    file_path = Path(path)
    initial_sha = _rehash_before_read(file_path, expected_sha256)

    log_handler = _ListLogHandler()
    logger = logging.getLogger("pypdf")
    logger.addHandler(log_handler)
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            page, _reader, pypdf_warnings = _read_pdf_page(file_path)
            media_box = _validate_box_coords(page.mediabox, "MediaBox")
            crop_box = _validate_box_coords(page.cropbox, "CropBox")
            _check_pdf_stream_nonempty(page)
            extracted_text = page.extract_text()
            font_inventory = _extract_font_resources(page)
            embedded_image_count = len(page.images)
            pypdf_warnings.extend(str(w.message) for w in recorded)
        pypdf_warnings.extend(log_handler.messages)
        pypdf_warnings = list(dict.fromkeys(pypdf_warnings))
    finally:
        logger.removeHandler(log_handler)

    _rehash_after_read(file_path, initial_sha)

    return {
        "format": "pdf",
        "path": str(file_path.resolve()),
        "file_sha256": initial_sha,
        "expected_sha256": expected_sha256,
        "byte_size": file_path.stat().st_size,
        "visual_approval": False,
        "claim": (
            "parse diagnostics only; does not prove visual acceptance, "
            "glyph visibility, or layout"
        ),
        "dimensions": {
            "page_count": 1,
            "media_box": media_box,
            "crop_box": crop_box,
        },
        "text": extracted_text,
        "font_resources": font_inventory,
        "embedded_image_count": embedded_image_count,
        "warnings": pypdf_warnings,
        "runtime_versions": _get_runtime_versions(),
        "source_sha256": _get_source_sha256(),
    }


def _parse_viewport_dimension(
    raw_dim: str | None, attr_name: str
) -> dict[str, Any] | None:
    """Parse SVG viewport dimension into positive finite value and unit."""
    if raw_dim is None:
        return None

    match = _DIMENSION_REGEX.match(raw_dim.strip())
    if not match:
        raise ValueError(
            f"Invalid SVG viewport {attr_name} attribute format: {raw_dim!r}"
        )

    val = float(match.group(1))
    unit = match.group(2) or ""

    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(
            f"SVG viewport {attr_name} must be positive and finite: {raw_dim!r}"
        )

    return {"value": val, "unit": unit}


def _parse_viewbox(raw_viewbox: str | None) -> dict[str, float] | None:
    """Parse SVG viewBox attribute into min_x, min_y, width, height."""
    if raw_viewbox is None:
        return None

    parts = [p for p in re.split(r"[\s,]+", raw_viewbox.strip()) if p]
    if len(parts) != 4:
        raise ValueError(
            f"SVG viewBox must contain 4 coordinates, found {len(parts)}: {raw_viewbox!r}"
        )

    try:
        coords = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Non-numeric coordinate in viewBox: {raw_viewbox!r}") from exc

    for c in coords:
        if not math.isfinite(c):
            raise ValueError(f"Non-finite coordinate in viewBox: {raw_viewbox!r}")

    min_x, min_y, width, height = coords
    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"viewBox width and height must be positive, got width={width}, height={height}"
        )

    return {
        "min_x": min_x,
        "min_y": min_y,
        "width": width,
        "height": height,
    }


def _validate_and_parse_svg_xml(
    raw_bytes: bytes,
) -> tuple[ET.Element, list[str]]:
    """Parse SVG through hardened DOM and ElementTree with inert-DTD policy."""
    comments: list[str] = []
    try:
        document = SafeDOM.parseString(
            raw_bytes, forbid_dtd=False, forbid_entities=True, forbid_external=True
        )
    except EntitiesForbidden as exc:
        raise ValueError("XML entity declaration rejected") from exc
    except DefusedXmlException as exc:
        raise ValueError(f"Malformed SVG XML: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Malformed SVG XML: {exc}") from exc

    try:
        doctype = document.doctype
        if doctype is not None:
            if doctype.internalSubset is not None:
                raise ValueError("XML DTD internal subset rejected")
            name = doctype.name
            system_id = doctype.systemId
            public_id = doctype.publicId
            if name != "svg":
                raise ValueError(f"Disallowed DOCTYPE root: {name}")
            if public_id and public_id != "-//W3C//DTD SVG 1.1//EN":
                raise ValueError(f"Disallowed DTD public ID: {public_id}")
            if system_id and system_id not in (
                "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd",
                "",
            ):
                raise ValueError(f"Disallowed DTD system ID: {system_id}")

        def visit(node: Any) -> None:
            # DOM nodeType 8 is Comment; 7 is ProcessingInstruction.
            if node.nodeType == 8:
                comments.append(node.data.strip())
            elif node.nodeType == 7 and node.target.lower() == "xml-stylesheet":
                raise ValueError("External stylesheet processing instruction rejected")
            for child in node.childNodes:
                visit(child)

        visit(document)
    finally:
        document.unlink()

    try:
        root = ET.fromstring(raw_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Malformed SVG XML: {exc}") from exc
    except DefusedXmlException as exc:
        raise ValueError(f"Malformed SVG XML: {exc}") from exc

    root_tag = root.tag
    if root_tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError(f"Root element is not <svg>, found: {root_tag}")

    return root, comments


def _check_url_reference(target: str, valid_ids: set[str], context: str) -> None:
    """Validate a url(...) reference target."""
    clean_target = target.strip()
    if clean_target.startswith("#"):
        local_id = clean_target[1:]
        if local_id not in valid_ids:
            raise ValueError(f"Unknown local reference {clean_target} in {context}")
    else:
        raise ValueError(
            f"External resource reference rejected in {context}: {clean_target}"
        )


def _validate_svg_element_resources(elem: ET.Element, valid_ids: set[str]) -> int:
    """Check element attributes and tags for external resources or broken local IDs."""
    tag = elem.tag
    image_count = 0
    clean_tag = tag.split("}")[-1] if "}" in tag else tag

    if clean_tag == "script":
        raise ValueError("Script element rejected in SVG")

    if clean_tag == "link" and (
        "href" in elem.attrib or "{http://www.w3.org/1999/xlink}href" in elem.attrib
    ):
        raise ValueError(f"External link resource rejected: {elem.attrib}")

    if clean_tag == "style" and elem.text:
        if _EXTERNAL_STYLE_REGEX.search(elem.text):
            raise ValueError(
                f"External style or font resource rejected: {elem.text.strip()}"
            )
        for match in _URL_REF_REGEX.finditer(elem.text):
            _check_url_reference(match.group(1), valid_ids, "<style>")

    href = elem.attrib.get("href") or elem.attrib.get(
        "{http://www.w3.org/1999/xlink}href"
    )
    if href is not None:
        clean_href = href.strip()
        if clean_href.startswith("#"):
            local_id = clean_href[1:]
            if local_id not in valid_ids:
                raise ValueError(
                    f"Unknown local reference {clean_href} in <{clean_tag}>"
                )
        elif clean_href.startswith("data:image/png;base64,") and clean_tag == "image":
            image_count += 1
        else:
            raise ValueError(
                f"External resource reference rejected in <{clean_tag}>: {clean_href}"
            )

    for attr_name in ("clip-path", "mask", "filter", "fill", "stroke", "style"):
        attr_val = elem.attrib.get(attr_name)
        if attr_val:
            for match in _URL_REF_REGEX.finditer(attr_val):
                _check_url_reference(
                    match.group(1), valid_ids, f"<{clean_tag} {attr_name}>"
                )

    return image_count


def _validate_svg_resources_and_references(
    root: ET.Element,
) -> tuple[list[str], int]:
    """Inventory IDs and ensure all references are safe and local."""
    valid_ids: set[str] = set()
    for elem in root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            if elem_id != elem_id.strip() or elem_id in valid_ids:
                raise ValueError(f"Invalid or duplicate SVG ID: {elem_id!r}")
            valid_ids.add(elem_id)

    total_embedded_images = 0
    for elem in root.iter():
        total_embedded_images += _validate_svg_element_resources(elem, valid_ids)

    sorted_ids = sorted(valid_ids)
    return sorted_ids, total_embedded_images


def _extract_svg_text(root: ET.Element) -> list[str]:
    """Extract text-element content, without asserting rendered glyph visibility."""
    texts: list[str] = []

    def visit(elem: ET.Element) -> None:
        if elem.tag == "{http://www.w3.org/2000/svg}metadata":
            return
        if elem.tag == "{http://www.w3.org/2000/svg}text":
            content = "".join(elem.itertext()).strip()
            if content:
                texts.append(content)
            return
        for child in elem:
            visit(child)

    visit(root)
    return texts


def inspect_svg(path: Path | str, expected_sha256: str) -> dict[str, Any]:
    """Inspect SVG vector asset and return diagnostics."""
    file_path = Path(path)
    initial_sha = _rehash_before_read(file_path, expected_sha256)

    raw_bytes = file_path.read_bytes()
    root, comments = _validate_and_parse_svg_xml(raw_bytes)

    width_info = _parse_viewport_dimension(root.attrib.get("width"), "width")
    height_info = _parse_viewport_dimension(root.attrib.get("height"), "height")
    view_box = _parse_viewbox(root.attrib.get("viewBox"))

    if (width_info is None or height_info is None) and view_box is None:
        raise ValueError(
            "Unsupported SVG qualification: complete viewport or viewBox required"
        )

    viewport_dict: dict[str, Any] = {}
    if width_info and height_info:
        viewport_dict = {
            "width": width_info["value"],
            "width_unit": width_info["unit"],
            "height": height_info["value"],
            "height_unit": height_info["unit"],
        }

    ids, embedded_image_count = _validate_svg_resources_and_references(root)
    text_content = _extract_svg_text(root)

    _rehash_after_read(file_path, initial_sha)

    return {
        "format": "svg",
        "path": str(file_path.resolve()),
        "file_sha256": initial_sha,
        "expected_sha256": expected_sha256,
        "byte_size": file_path.stat().st_size,
        "visual_approval": False,
        "claim": (
            "parse diagnostics only; does not prove visual acceptance, "
            "glyph visibility, or layout"
        ),
        "viewport": viewport_dict,
        "view_box": view_box,
        "text": text_content,
        "comments": comments,
        "ids": ids,
        "embedded_image_count": embedded_image_count,
        "runtime_versions": _get_runtime_versions(),
        "source_sha256": _get_source_sha256(),
    }


def inspect_vector(path: Path | str, expected_sha256: str) -> dict[str, Any]:
    """Dispatch inspection by vector file suffix (.pdf or .svg)."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return inspect_pdf(file_path, expected_sha256)
    if suffix == ".svg":
        return inspect_svg(file_path, expected_sha256)
    raise ValueError(f"Unsupported vector format: {file_path.suffix}")
