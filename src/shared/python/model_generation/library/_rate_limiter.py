"""Rate-limit handling for GitHub API requests.

This module provides utilities for handling GitHub API rate limits:
- Checking rate-limit headers in responses
- Implementing exponential backoff for 429 responses
- Logging rate-limit status for monitoring
"""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# Rate-limit constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_MAX_BACKOFF = 32.0  # seconds
RATE_LIMIT_HEADER = "X-RateLimit-Remaining"
RATE_LIMIT_RESET_HEADER = "X-RateLimit-Reset"
RATE_LIMIT_LIMIT_HEADER = "X-RateLimit-Limit"


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and retries exhausted."""


def extract_rate_limit_info(response: Any) -> dict[str, int | None]:
    """Extract rate-limit information from response headers.

    Args:
        response: urllib response object with headers

    Returns:
        Dictionary with rate-limit info (remaining, limit, reset_epoch)
    """
    headers = response.headers if hasattr(response, "headers") else {}
    return {
        "remaining": _get_int_header(headers, RATE_LIMIT_HEADER),
        "limit": _get_int_header(headers, RATE_LIMIT_LIMIT_HEADER),
        "reset_epoch": _get_int_header(headers, RATE_LIMIT_RESET_HEADER),
    }


def _get_int_header(headers: Any, key: str) -> int | None:
    """Extract integer value from headers."""
    try:
        value = headers.get(key)
        return int(value) if value else None
    except (ValueError, TypeError, AttributeError):
        return None


def log_rate_limit_status(
    url: str,
    rate_limit_info: dict[str, int | None],
    status_code: int | None = None,
) -> None:
    """Log current rate-limit status.

    Args:
        url: Request URL
        rate_limit_info: Rate-limit info dict from extract_rate_limit_info
        status_code: HTTP status code (for context)
    """
    remaining = rate_limit_info.get("remaining")
    limit = rate_limit_info.get("limit")
    reset_epoch = rate_limit_info.get("reset_epoch")

    if remaining is not None and limit is not None:
        logger.info(
            f"Rate-limit status: {remaining}/{limit} remaining (URL: {url[:50]}...)"
        )
    elif status_code == 429:
        logger.warning(f"Rate-limit exceeded (429) for {url[:50]}...")

    if reset_epoch is not None:
        reset_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reset_epoch))
        logger.info(f"Rate-limit resets at {reset_time}")


def make_request_with_backoff(
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
    headers: dict[str, str] | None = None,
) -> urllib.request.Response:
    """Make a request with exponential backoff for rate-limit errors.

    Args:
        url: Request URL
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff duration in seconds
        max_backoff: Maximum backoff duration in seconds
        headers: Optional headers dict

    Returns:
        Response object

    Raises:
        RateLimitError: If rate limit exceeded after retries
        urllib.error.URLError: For other HTTP errors
    """
    if headers is None:
        headers = {}

    backoff = initial_backoff
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            for key, value in headers.items():
                req.add_header(key, value)

            response = urllib.request.urlopen(req, timeout=10)  # nosec B310

            # Log rate-limit status on success
            rate_limit_info = extract_rate_limit_info(response)
            if rate_limit_info.get("remaining") is not None:
                log_rate_limit_status(url, rate_limit_info, 200)

            return response

        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limited - exponential backoff
                rate_limit_info = extract_rate_limit_info(e)
                log_rate_limit_status(url, rate_limit_info, 429)

                if attempt < max_retries - 1:
                    wait_time = min(backoff, max_backoff)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Backing off for {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    backoff *= 2  # Exponential backoff
                    last_error = e
                    continue
                raise RateLimitError(
                    f"Rate limit exceeded after {max_retries} attempts"
                ) from e
            # Other HTTP errors - don't retry
            raise

        except (urllib.error.URLError, OSError) as e:
            # Network errors - retry with backoff
            if attempt < max_retries - 1:
                wait_time = min(backoff, max_backoff)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                backoff *= 2
                last_error = e
                continue
            raise

    # Should not reach here, but raise last error if we do
    if last_error:
        raise last_error
    raise RateLimitError("Request failed: no retries remaining")
