"""Render research Markdown through Pandoc AST into a styled, linked PDF.

Usage: python -m scripts.research.shot_pattern_analysis.build_pdf SOURCE.md OUTPUT.pdf
The source remains authoritative; this builder never edits Markdown or figures.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any


def normalize_math_for_mathtext(source: str) -> str:
    """Map common concise LaTeX spellings to equivalent MathText grammar."""
    expression = source.replace("\n", " ").replace(r"\frac12", r"\frac{1}{2}")
    return re.sub(
        r"\\(mathbf|boldsymbol)\s*(\\[A-Za-z]+|[A-Za-z])",
        lambda match: rf"\{match.group(1)}{{{match.group(2)}}}",
        expression,
    )


def _pandoc_ast(source: Path, pandoc_binary: Path | None) -> dict:
    executable = (
        str(pandoc_binary) if pandoc_binary is not None else shutil.which("pandoc")
    )
    if not executable or (pandoc_binary is not None and not pandoc_binary.is_file()):
        raise FileNotFoundError(
            "pandoc executable is required to parse research Markdown"
        )
    result = subprocess.run(
        [executable, "--from=gfm+tex_math_dollars", "--to=json", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"pandoc failed: {result.stderr.strip()}")
    document = json.loads(result.stdout)
    if not isinstance(document.get("blocks"), list):
        raise ValueError("pandoc did not return a block-level document")
    return document


def _register_fonts() -> None:
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    directory = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    names = {
        "DejaVu": "DejaVuSans.ttf",
        "DejaVu-Bold": "DejaVuSans-Bold.ttf",
        "DejaVu-Oblique": "DejaVuSans-Oblique.ttf",
        "DejaVu-BoldOblique": "DejaVuSans-BoldOblique.ttf",
        "DejaVu-Mono": "DejaVuSansMono.ttf",
    }
    for name, filename in names.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"required Unicode PDF font missing: {path}")
            pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "DejaVu",
        normal="DejaVu",
        bold="DejaVu-Bold",
        italic="DejaVu-Oblique",
        boldItalic="DejaVu-BoldOblique",
    )


def _render_math(
    expression: str, directory: Path, index: int, *, display: bool
) -> tuple[Path, float, float]:
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    from PIL import Image as PillowImage

    normalized = normalize_math_for_mathtext(expression)
    path = directory / f"math-{index}.png"
    dpi = 180
    try:
        math_to_image(
            f"${normalized}$",
            str(path),
            prop=FontProperties(size=11.5 if display else 9.5),
            dpi=dpi,
            format="png",
        )
    except Exception as exc:
        raise ValueError(f"MathText cannot render equation: {expression!r}") from exc
    with PillowImage.open(path) as image:
        width_px, height_px = image.size
    return path, width_px * 72 / dpi, height_px * 72 / dpi


class _FlowBuilder:
    def __init__(
        self, source: Path, temporary: Path, styles: dict[str, Any], width: float
    ):
        self.source = source
        self.temporary = temporary
        self.styles = styles
        self.width = width
        self.math_index = 0

    def _math(self, value: str, *, display: bool) -> tuple[Path, float, float]:
        self.math_index += 1
        return _render_math(value, self.temporary, self.math_index, display=display)

    def inlines(self, nodes: list[dict]) -> str:
        pieces = []
        for node in nodes:
            kind = node["t"]
            data: Any = node.get("c")
            if kind == "Str":
                pieces.append(html.escape(str(data)))
            elif kind in ("Space", "SoftBreak"):
                pieces.append(" ")
            elif kind == "LineBreak":
                pieces.append("<br/>")
            elif kind in ("Emph", "Strong", "Strikeout"):
                tag = {"Emph": "i", "Strong": "b", "Strikeout": "strike"}[kind]
                pieces.append(f"<{tag}>{self.inlines(data)}</{tag}>")
            elif kind == "Code":
                pieces.append(
                    f'<font name="DejaVu-Mono" size="8">{html.escape(data[1])}</font>'
                )
            elif kind == "Link":
                target = html.escape(data[2][0], quote=True)
                pieces.append(
                    f'<link href="{target}" color="#087889">{self.inlines(data[1])}</link>'
                )
            elif kind == "Math":
                path, width, height = self._math(data[1], display=False)
                if width > self.width * 0.82:
                    scale = self.width * 0.82 / width
                    width, height = width * scale, height * scale
                pieces.append(
                    f'<img src="{html.escape(str(path), quote=True)}" '
                    f'width="{width:.2f}" height="{height:.2f}" valign="middle"/>'
                )
            elif kind == "Image":
                pieces.append(self.inlines(data[1]))
            elif kind in ("Span", "Quoted"):
                pieces.append(self.inlines(data[-1]))
            elif kind == "Superscript":
                pieces.append(f"<super>{self.inlines(data)}</super>")
            elif kind == "Subscript":
                pieces.append(f"<sub>{self.inlines(data)}</sub>")
            elif kind == "RawInline":
                pieces.append(html.escape(str(data[1])))
            else:
                raise ValueError(f"unsupported Pandoc inline node: {kind}")
        return "".join(pieces)

    def _paragraph(self, nodes: list[dict], style: str = "Body") -> Any:
        from reportlab.platypus import Paragraph

        return Paragraph(self.inlines(nodes), self.styles[style])

    def _image(self, node: dict) -> list[Any]:
        from PIL import Image as PillowImage
        from reportlab.platypus import Image, Paragraph

        _, alternate, target = node["c"]
        path = (self.source.parent / target[0]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Markdown image is missing: {target[0]}")
        with PillowImage.open(path) as bitmap:
            pixels_w, pixels_h = bitmap.size
        scale = min(self.width / pixels_w, 510 / pixels_h, 1.0)
        picture = Image(str(path), width=pixels_w * scale, height=pixels_h * scale)
        picture.hAlign = "CENTER"
        caption = Paragraph(self.inlines(alternate), self.styles["Caption"])
        return [picture, caption]

    def _table(self, data: list) -> Any:
        from reportlab.lib import colors
        from reportlab.platypus import LongTable, Paragraph, TableStyle

        def cells(row: list) -> list[Any]:
            result = []
            for cell in row[1]:
                blocks = cell[4]
                markup = "<br/>".join(
                    self.inlines(block["c"])
                    for block in blocks
                    if block["t"] in ("Plain", "Para")
                )
                result.append(Paragraph(markup or " ", self.styles["TableCell"]))
            return result

        header = [cells(row) for row in data[3][1]]
        body = [cells(row) for section in data[4] for row in section[3]]
        footer = [cells(row) for row in data[5][1]]
        rows = header + body + footer
        if not rows:
            raise ValueError("Markdown table has no rows")
        columns = len(rows[0])
        if any(len(row) != columns for row in rows):
            raise ValueError("Markdown table rows have inconsistent column counts")
        widths = [self.width / columns] * columns
        if columns > 2:
            widths[0] = self.width * 0.22
            widths[1:] = [(self.width - widths[0]) / (columns - 1)] * (columns - 1)
        table = LongTable(rows, colWidths=widths, repeatRows=len(header), hAlign="LEFT")
        table.spaceBefore = 6
        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, len(header) - 1),
                        colors.HexColor("#E5EDF1"),
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, len(header)),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F7FAFB")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "LINEBELOW",
                        (0, len(header) - 1),
                        (-1, len(header) - 1),
                        0.65,
                        colors.HexColor("#7CA7AD"),
                    ),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8E3E8")),
                    ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#B7CDD3")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    # Keep a small tail together so an otherwise full page
                    # does not strand one result row on a new page.
                    *(
                        [("NOSPLIT", (0, len(rows) - 4), (-1, -1))]
                        if len(rows) - len(header) >= 4
                        else []
                    ),
                ]
            )
        )
        return table

    def blocks(self, nodes: list[dict]) -> list[Any]:
        from reportlab.lib import colors
        from reportlab.platypus import (
            HRFlowable,
            Image,
            KeepTogether,
            Paragraph,
            Spacer,
        )

        flow = []
        for index, node in enumerate(nodes):
            kind = node["t"]
            data: Any = node.get("c")
            next_node = nodes[index + 1] if index + 1 < len(nodes) else None
            if kind == "Header":
                level, _, inlines = data
                style = "H1" if level == 1 else "H2" if level == 2 else "H3"
                heading = self._paragraph(inlines, style)
                heading._outline_level = min(level - 1, 2)
                heading._outline_text = re.sub(r"<[^>]+>", "", self.inlines(inlines))
                flow.append(heading)
            elif kind in ("Para", "Plain"):
                if len(data) == 1 and data[0]["t"] == "Image":
                    flow.extend(self._image(data[0]))
                elif (
                    len(data) == 1
                    and data[0]["t"] == "Math"
                    and data[0]["c"][0]["t"] == "DisplayMath"
                ):
                    path, width, height = self._math(data[0]["c"][1], display=True)
                    scale = min(1.0, self.width / width)
                    equation = Image(
                        str(path), width=width * scale, height=height * scale
                    )
                    equation.hAlign = "CENTER"
                    flow.extend((Spacer(1, 5), equation, Spacer(1, 8)))
                elif (
                    len(data) == 1
                    and data[0]["t"] == "Strong"
                    and next_node is not None
                    and next_node["t"]
                    in ("Para", "Plain", "OrderedList", "BulletList", "BlockQuote")
                ):
                    heading = self._paragraph(data)
                    heading.keepWithNext = True
                    flow.append(heading)
                else:
                    flow.append(self._paragraph(data))
            elif kind == "Table":
                # Declare spaceBefore on LongTable directly so preceding
                # headings keep with the table without an intervening spacer.
                flow.extend((self._table(data), Spacer(1, 9)))

            elif kind in ("OrderedList", "BulletList"):
                items = data[1] if kind == "OrderedList" else data
                for index, item in enumerate(items, start=1):
                    prefix = f"{index}." if kind == "OrderedList" else "•"
                    pieces = [
                        Paragraph(
                            f"<b>{html.escape(prefix)}</b>  {self.inlines(block['c'])}",
                            self.styles["List"],
                        )
                        for block in item
                        if block["t"] in ("Para", "Plain")
                    ]
                    if pieces:
                        flow.append(KeepTogether(pieces))
            elif kind == "BlockQuote":
                flow.extend(
                    self._paragraph(block["c"], "Quote")
                    for block in data
                    if block["t"] in ("Para", "Plain")
                )
            elif kind == "CodeBlock":
                from reportlab.platypus import Preformatted

                flow.append(
                    Preformatted(data[1], self.styles["CodeBlock"], maxLineLength=95)
                )
            elif kind == "HorizontalRule":
                flow.append(
                    HRFlowable(
                        width="100%", thickness=0.5, color=colors.HexColor("#A7BFC7")
                    )
                )
            else:
                raise ValueError(f"unsupported Pandoc block node: {kind}")
        return flow


def _styles() -> dict[str, Any]:
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    ink = colors.HexColor("#223342")
    accent = colors.HexColor("#087889")
    return {
        "Body": ParagraphStyle(
            "Body",
            fontName="DejaVu",
            fontSize=9.25,
            leading=14.0,
            textColor=ink,
            spaceAfter=8,
        ),
        "H1": ParagraphStyle(
            "H1",
            fontName="DejaVu-Bold",
            fontSize=17,
            leading=22,
            textColor=ink,
            spaceAfter=11,
            keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2",
            fontName="DejaVu-Bold",
            fontSize=12.5,
            leading=17,
            textColor=accent,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "H3": ParagraphStyle(
            "H3",
            fontName="DejaVu-Bold",
            fontSize=10.5,
            leading=14,
            textColor=ink,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "List": ParagraphStyle(
            "List",
            fontName="DejaVu",
            fontSize=9.15,
            leading=14,
            leftIndent=17,
            firstLineIndent=-12,
            textColor=ink,
            spaceAfter=5,
        ),
        "Quote": ParagraphStyle(
            "Quote",
            fontName="DejaVu-Oblique",
            fontSize=9,
            leading=13.5,
            leftIndent=16,
            rightIndent=12,
            textColor=ink,
            spaceAfter=8,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            fontName="DejaVu",
            fontSize=8,
            leading=11,
            alignment=1,
            textColor=colors.HexColor("#536776"),
            spaceBefore=4,
            spaceAfter=10,
        ),
        "TableCell": ParagraphStyle(
            "TableCell", fontName="DejaVu", fontSize=7.55, leading=10.5, textColor=ink
        ),
        "CodeBlock": ParagraphStyle(
            "CodeBlock",
            fontName="DejaVu-Mono",
            fontSize=7.7,
            leading=10.3,
            backColor=colors.HexColor("#F2F5F6"),
            leftIndent=8,
            rightIndent=8,
            spaceBefore=5,
            spaceAfter=9,
        ),
    }


def render_markdown_to_pdf(
    source: Path,
    output: Path,
    *,
    pandoc_binary: Path | None = None,
) -> Path:
    """Build a PDF atomically from one authoritative Markdown research file."""
    from pypdf import PdfReader
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"research Markdown is missing: {source}")
    if source.suffix.lower() != ".md" or output.suffix.lower() != ".pdf":
        raise ValueError("source must be .md and output must be .pdf")
    document = _pandoc_ast(source, pandoc_binary)
    _register_fonts()
    page_w, page_h = A4
    margin_x, margin_top, margin_bottom = 54, 62, 51
    width = page_w - 2 * margin_x
    title = next(
        (
            " ".join(part["c"] for part in block["c"][2] if part["t"] == "Str")
            for block in document["blocks"]
            if block["t"] == "Header" and block["c"][0] == 1
        ),
        source.stem.replace("_", " ").title(),
    )

    class ResearchDoc(BaseDocTemplate):
        def afterFlowable(self, flowable: Any) -> None:
            if hasattr(flowable, "_outline_level"):
                key = f"heading-{self.seq.nextf('heading')}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(
                    flowable._outline_text,
                    key,
                    level=flowable._outline_level,
                    closed=False,
                )

    def furniture(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#A7C8CF"))
        canvas.setLineWidth(0.6)
        canvas.line(margin_x, page_h - 44, page_w - margin_x, page_h - 44)
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor("#56707B"))
        canvas.drawString(margin_x, page_h - 35, title[:85])
        canvas.line(margin_x, 39, page_w - margin_x, 39)
        canvas.drawString(margin_x, 27, "UpstreamDrift  |  Research Reference")
        canvas.drawRightString(page_w - margin_x, 27, f"Page {doc.page}")
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="shot-pattern-pdf-") as directory:
        working = Path(directory)
        builder = _FlowBuilder(source, working, _styles(), width)
        flow = builder.blocks(document["blocks"])
        if not flow:
            raise ValueError("research Markdown contains no renderable content")
        with NamedTemporaryFile(
            suffix=".pdf", dir=output.parent, delete=False
        ) as handle:
            staged = Path(handle.name)
        try:
            doc = ResearchDoc(
                str(staged),
                pagesize=A4,
                leftMargin=margin_x,
                rightMargin=margin_x,
                topMargin=margin_top,
                bottomMargin=margin_bottom,
                title=title,
                author="UpstreamDrift Research",
                invariant=1,
            )
            doc.addPageTemplates(
                PageTemplate(
                    id="Research",
                    frames=[
                        Frame(
                            margin_x,
                            margin_bottom,
                            width,
                            page_h - margin_top - margin_bottom,
                            leftPadding=0,
                            rightPadding=0,
                            topPadding=0,
                            bottomPadding=0,
                        )
                    ],
                    onPage=furniture,
                )
            )
            doc.build(flow)
            if len(PdfReader(staged).pages) < 1:
                raise RuntimeError("PDF build produced no pages")
            os.replace(staged, output)
        finally:
            staged.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pandoc", type=Path)
    args = parser.parse_args()
    result = render_markdown_to_pdf(args.source, args.output, pandoc_binary=args.pandoc)
    print(result)


if __name__ == "__main__":
    main()
