"""Backward compatibility shim - module moved to core.datetime_utils."""

import sys as _sys

from .core import datetime_utils as _real_module  # noqa: E402
from .core.datetime_utils import (  # noqa: F401
    UTC,
    TimestampFormat,
    add_days,
    end_of_day,
    format_datetime,
    format_duration,
    format_timedelta,
    is_expired,
    now,
    now_local,
    now_utc,
    parse_date,
    parse_timestamp,
    start_of_day,
    time_ago,
    timestamp,
    timestamp_display,
    timestamp_filename,
    timestamp_iso,
)

_sys.modules[__name__] = _real_module
