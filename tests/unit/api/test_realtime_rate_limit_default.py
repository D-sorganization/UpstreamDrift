"""Tests for realtime publish rate limit default (#6972).

The default of 60/minute makes realtime streams above 1 Hz unusable when
WSPubSub.publish() posts every message via HTTP.  The default is raised to
6000/minute (100/s) so typical realtime analysis streams work out of the box.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.mark.unit
def test_realtime_publish_default_allows_high_frequency() -> None:
    """API_LIMIT_REALTIME_PUBLISH default must be >= 6000/minute for realtime streams.

    The decorator arg `get_limit("API_LIMIT_REALTIME_PUBLISH", "<default>")` is
    evaluated at import time.  We check the source of routes/realtime.py to ensure
    the hard-coded default is at least 6000/minute (100 msg/s) so 30–100 Hz realtime
    analysis feeds don't hit the rate-limiter under normal operation.
    """
    realtime_src = Path("src/api/routes/realtime.py").read_text()

    # Extract the default value from the get_limit call, e.g. "60/minute"
    match = re.search(
        r'get_limit\s*\(\s*["\']API_LIMIT_REALTIME_PUBLISH["\'],\s*["\'](\d+)/minute["\']\s*\)',
        realtime_src,
    )
    assert match, (
        "Could not find get_limit('API_LIMIT_REALTIME_PUBLISH', ...) in "
        "src/api/routes/realtime.py — has the decorator changed?"
    )

    default_count = int(match.group(1))
    assert default_count >= 6000, (
        f"Default publish rate limit is {default_count}/minute; "
        "must be >= 6000/minute to support 100 Hz realtime streams without throttling"
    )
