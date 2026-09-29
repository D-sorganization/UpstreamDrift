"""An *optional* route module with a broken import.

When ``video`` is declared optional, route discovery is allowed to skip it on
ImportError (its feature extra is absent in slim images) rather than failing
the whole server startup.
"""

import this_optional_dep_is_absent_7128  # noqa: F401

from fastapi import APIRouter

router = APIRouter()
