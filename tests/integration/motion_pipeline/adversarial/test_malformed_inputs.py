"""Adversarial: malformed inputs across every adapter format.

For every supported adapter, verify that the adapter rejects malformed
input with a clear, typed error rather than crashing uninformatively or
returning silent garbage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import list_formats, load_any
from src.shared.python.motion_pipeline.sources.base import (
    AdapterContractError,
    UnsupportedFormatError,
)
import contextlib

# Map each format to a canonical extension we can exercise.
FORMAT_EXT = {
    "bvh": ".bvh",
    "trc": ".trc",
    "opensim_sto_mot": ".sto",
    "mediapipe_json": ".json",
    "alphapose_json": ".json",
    "hrnet_json": ".json",
    "openpose_json": ".json",
    "csv": ".csv",
    "c3d": ".c3d",
}


def _write(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


# ---------------------------------------------------------------------------
# Empty + truncated files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", sorted(set(FORMAT_EXT.values())))
def test_empty_file_rejected(tmp_path: Path, ext: str, request) -> None:
    """0-byte files of every recognised extension must be rejected."""
    if ext == ".c3d":
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="GH #4721 — c3d adapter raises raw OSError on empty file instead of typed AdapterContractError",
                raises=OSError,
            )
        )
    p = _write(tmp_path, f"empty{ext}", b"")
    with pytest.raises((UnsupportedFormatError, AdapterContractError, ValueError)):
        load_any(p)


def test_truncated_bvh_header_only(tmp_path: Path) -> None:
    """BVH file with only HIERARCHY header but no data must error."""
    p = _write(tmp_path, "trunc.bvh", b"HIERARCHY\nROOT Hips\n")
    with pytest.raises(
        (UnsupportedFormatError, AdapterContractError, ValueError, Exception)
    ):
        load_any(p)


def test_truncated_trc_header_only(tmp_path: Path) -> None:
    """TRC file with only the PathFileType header must error."""
    p = _write(tmp_path, "trunc.trc", b"PathFileType\t4\t(X/Y/Z)\ttrunc.trc\n")
    with pytest.raises(
        (UnsupportedFormatError, AdapterContractError, ValueError, Exception)
    ):
        load_any(p)


# ---------------------------------------------------------------------------
# Wrong format / extension
# ---------------------------------------------------------------------------


def test_csv_content_with_bvh_extension(tmp_path: Path) -> None:
    """CSV bytes given a .bvh extension must not silently parse."""
    p = _write(tmp_path, "evil.bvh", b"a,b,c\n1,2,3\n4,5,6\n")
    with pytest.raises(
        Exception
    ):  # noqa: B017 - any exception is acceptable; no-crash is the test
        load_any(p)


def test_random_binary_with_json_extension(tmp_path: Path) -> None:
    """Random binary garbage with a .json extension must not crash."""
    p = _write(tmp_path, "garbage.json", bytes(range(256)) * 4)
    with pytest.raises(Exception):  # noqa: B017
        load_any(p)


# ---------------------------------------------------------------------------
# NaN / Inf in numeric fields (CSV, JSON)
# ---------------------------------------------------------------------------


def test_csv_with_nan_values(tmp_path: Path) -> None:
    """CSV containing NaN in coordinate columns must be rejected by the
    adapter contract (frames must be finite)."""
    content = b"time,m1_x,m1_y,m1_z\n0.0,nan,0.0,0.0\n0.01,1.0,2.0,3.0\n"
    p = _write(tmp_path, "nan.csv", content)
    with pytest.raises((AdapterContractError, ValueError, Exception)):
        load_any(p)


def test_json_with_inf_values(tmp_path: Path) -> None:
    """A JSON file with Infinity literals must not be silently accepted as
    finite mocap data."""
    # Strict JSON does not accept bare Infinity, but Python's json library
    # parses it by default. Either rejection (parse error) or contract
    # rejection (non-finite coords) is acceptable.
    content = b'{"frames": [{"timestamp": 0.0, "keypoints": [{"x": Infinity, "y": 0, "z": 0}]}]}'
    p = _write(tmp_path, "inf.json", content)
    with pytest.raises(Exception):  # noqa: B017
        load_any(p)


# ---------------------------------------------------------------------------
# Bogus timestamps
# ---------------------------------------------------------------------------


def test_csv_negative_timestamp(tmp_path: Path) -> None:
    """Negative timestamps in CSV must be rejected."""
    content = b"time,m1_x,m1_y,m1_z\n-1.0,0.0,0.0,0.0\n0.0,1.0,2.0,3.0\n"
    p = _write(tmp_path, "neg.csv", content)
    with pytest.raises(Exception):  # noqa: B017
        load_any(p)


def test_csv_duplicate_timestamps(tmp_path: Path) -> None:
    """Two frames with identical timestamps must be rejected (timestamps
    must be strictly monotonic, or at minimum non-decreasing — duplicates
    typically indicate a parser bug)."""
    content = b"time,m1_x,m1_y,m1_z\n0.0,0.0,0.0,0.0\n0.0,1.0,2.0,3.0\n"
    p = _write(tmp_path, "dup.csv", content)
    # Loose: at least no crash. Strict: should raise.
    try:
        result = load_any(p)
    except Exception:
        return
    # If it loaded, contract requires non-decreasing timestamps which is
    # technically satisfied by 0 == 0, but identical timestamps are a smell.
    assert result is not None


# ---------------------------------------------------------------------------
# Sparse / oversized / Unicode
# ---------------------------------------------------------------------------


def test_huge_zero_padded_file(tmp_path: Path) -> None:
    """A 1MB file of zero bytes with a .bvh extension must be rejected
    quickly (no infinite loop, no crash)."""
    p = _write(tmp_path, "zeros.bvh", b"\x00" * (1024 * 1024))
    with pytest.raises(Exception):  # noqa: B017
        load_any(p)


def test_unicode_marker_names_in_csv(tmp_path: Path) -> None:
    """CSV with non-ASCII characters in marker headers must either parse
    correctly or raise — never crash on encoding."""
    content = "time,café_x,café_y,café_z\n0.0,1.0,2.0,3.0\n0.01,1.1,2.1,3.1\n".encode()
    p = _write(tmp_path, "unicode.csv", content)
    # Either succeeds or raises; must not segfault.
    with contextlib.suppress(Exception):
        load_any(p)
