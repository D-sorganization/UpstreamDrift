"""Shared CSS for equation popups.

Extracted from equations_popup.py.
"""

# ---------------------------------------------------------------------------
# Shared CSS for all equation popups
# ---------------------------------------------------------------------------

_CSS = """
body {
    background: #1a1a28;
    color: #c0c0d8;
    font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif;
    font-size: 14px;
    line-height: 1.7;
    padding: 20px 28px;
    max-width: 800px;
}
h1 { color: #6fa8dc; font-size: 22px; border-bottom: 2px solid #3a5a8c; padding-bottom: 8px; margin-top: 28px; }
h2 { color: #6fa8dc; font-size: 18px; margin-top: 24px; border-bottom: 1px solid #303050; padding-bottom: 4px; }
h3 { color: #7db8ec; font-size: 15px; margin-top: 18px; }
.eq {
    background: #12121e;
    border: 1px solid #303050;
    border-radius: 6px;
    padding: 14px 20px;
    margin: 12px 0;
    font-family: 'Cambria Math', 'STIX Two Math', 'Latin Modern Math', Georgia, serif;
    font-size: 16px;
    color: #a0e0a0;
    overflow-x: auto;
    line-height: 2.0;
}
.eq-inline {
    font-family: 'Cambria Math', 'STIX Two Math', Georgia, serif;
    color: #a0e0a0;
    font-size: 15px;
}
table.params {
    border-collapse: collapse;
    margin: 10px 0;
    width: 100%;
}
table.params td {
    padding: 6px 12px;
    border-bottom: 1px solid #303050;
    vertical-align: top;
}
table.params td:first-child {
    font-family: 'Cambria Math', Georgia, serif;
    color: #a0e0a0;
    white-space: nowrap;
    width: 120px;
}
.note {
    background: #1e1e32;
    border-left: 3px solid #6fa8dc;
    padding: 10px 16px;
    margin: 12px 0;
    font-style: italic;
}
.matrix {
    font-family: 'Cambria Math', Georgia, serif;
    font-size: 15px;
    white-space: pre;
    line-height: 1.8;
}
ul { padding-left: 22px; }
li { margin-bottom: 6px; }
"""

__all__ = ["_CSS"]
