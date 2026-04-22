"""Upload size validation middleware."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from src.api.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB
from src.api.middleware.security_headers import add_security_headers_to_response

UPLOAD_STREAM_CHUNK_SIZE_BYTES = 64 * 1024


def _upload_size_limit_detail(max_bytes: int | None) -> str:
    """Return the standard limit-exceeded message for uploads."""
    if max_bytes is None or max_bytes == MAX_UPLOAD_SIZE_BYTES:
        return f"Request too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB"
    return f"Request too large. Maximum size is {max_bytes} bytes"


async def iter_upload_file_chunks(
    file: UploadFile,
    *,
    max_bytes: int | None = None,
    chunk_size: int = UPLOAD_STREAM_CHUNK_SIZE_BYTES,
) -> AsyncIterator[bytes]:
    """Yield upload chunks while enforcing a hard byte ceiling."""
    if file is None:
        raise ValueError("file must be provided")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    limit = MAX_UPLOAD_SIZE_BYTES if max_bytes is None else max_bytes
    if limit <= 0:
        raise ValueError("max_bytes must be positive")

    bytes_read = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        bytes_read += len(chunk)
        if bytes_read > limit:
            raise HTTPException(
                status_code=413,
                detail=_upload_size_limit_detail(max_bytes),
            )
        yield chunk


async def read_upload_file_bytes(
    file: UploadFile,
    *,
    max_bytes: int | None = None,
    chunk_size: int = UPLOAD_STREAM_CHUNK_SIZE_BYTES,
) -> bytes:
    """Read an upload into memory using bounded streaming reads."""
    data = bytearray()
    async for chunk in iter_upload_file_chunks(
        file, max_bytes=max_bytes, chunk_size=chunk_size
    ):
        data.extend(chunk)
    return bytes(data)


async def write_upload_file_to_path(
    file: UploadFile,
    destination: Path,
    *,
    max_bytes: int | None = None,
    chunk_size: int = UPLOAD_STREAM_CHUNK_SIZE_BYTES,
) -> int:
    """Stream an upload to disk while enforcing the upload byte ceiling."""
    bytes_written = 0
    try:
        with destination.open("wb") as output_file:
            async for chunk in iter_upload_file_chunks(
                file, max_bytes=max_bytes, chunk_size=chunk_size
            ):
                output_file.write(chunk)
                bytes_written += len(chunk)
    except Exception:
        with contextlib.suppress(FileNotFoundError, OSError):
            destination.unlink()
        raise

    return bytes_written


async def validate_upload_size(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Reject requests exceeding upload size limits."""
    if request is None:
        raise ValueError("request must be provided")
    content_length = request.headers.get("content-length")

    if content_length:
        try:
            content_length_int = int(content_length)
        except ValueError:
            response = JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )
            return add_security_headers_to_response(response, request)
        if content_length_int > MAX_UPLOAD_SIZE_BYTES:
            response = JSONResponse(
                status_code=413,
                content={
                    "detail": f"Request too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB"
                },
            )
            return add_security_headers_to_response(response, request)

    return await call_next(request)
