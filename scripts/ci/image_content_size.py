#!/usr/bin/env python3
"""Print a local image's compressed content size in MB (#14146).

The docker size gates used ``docker image inspect --format '{{.Size}}'``, but
with the containerd image store that field changed meaning between Docker
Engine releases: 29.1.3 reports the compressed content size the budgets were
calibrated against, while 29.9.2 reports the unpacked snapshot size (3-4x
larger). The gate then passed or failed depending on which runner took the job.

This script reads the per-platform ``Size.Content`` that the Engine API returns
for ``images/<ref>/json?manifests=1`` (API 1.48+): the sum of the compressed
layer and config blobs, which does not depend on the engine version. It fails
closed rather than fall back to ``.Size``, so an engine change can never
silently move what the budgets measure.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import http.client
import json
import os
import socket
import sys
from typing import Any
from urllib.parse import quote

API_VERSION = "v1.48"  # first Engine API with per-manifest sizes
DEFAULT_SOCKET = "/var/run/docker.sock"
BYTES_PER_MB = 1024 * 1024


class MeasurementError(RuntimeError):
    """The daemon did not report a version-independent content size."""


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._socket_path)


def _socket_path() -> str:
    host = os.environ.get("DOCKER_HOST", "")
    if host and not host.startswith("unix://"):
        raise MeasurementError(f"only unix:// DOCKER_HOST is supported, got {host}")
    return host.removeprefix("unix://") or DEFAULT_SOCKET


def _get_json(path: str) -> Any:
    connection = _UnixHTTPConnection(_socket_path())
    try:
        connection.request("GET", f"/{API_VERSION}{path}")
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status != 200:
        raise MeasurementError(f"GET {path} -> HTTP {response.status}: {body!r}")
    return json.loads(body)


def content_size_bytes(inspect: Mapping[str, Any], architecture: str) -> int:
    """Return the compressed content size of the image for ``architecture``.

    Args:
        inspect: The ``images/<ref>/json?manifests=1`` payload.
        architecture: The daemon's architecture (e.g. ``amd64``).

    Raises:
        MeasurementError: No per-manifest sizes, or no available image
            manifest for ``architecture``.
    """
    manifests = inspect.get("Manifests")
    if not manifests:
        raise MeasurementError(
            "inspect payload has no Manifests; the daemon needs the containerd "
            "image store and Engine API 1.48+"
        )
    for manifest in manifests:
        platform = manifest.get("Descriptor", {}).get("platform", {})
        if (
            manifest.get("Kind") == "image"
            and manifest.get("Available")
            and platform.get("architecture") == architecture
        ):
            return int(manifest["Size"]["Content"])
    raise MeasurementError(f"no available image manifest for {architecture}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="local image reference, e.g. app:slim")
    args = parser.parse_args(argv)
    try:
        architecture = _get_json("/version")["Arch"]
        inspect = _get_json(f"/images/{quote(args.image, safe='')}/json?manifests=1")
        size = content_size_bytes(inspect, architecture)
    except (MeasurementError, OSError) as exc:
        sys.stderr.write(f"::error::cannot measure {args.image}: {exc}\n")
        return 1
    sys.stdout.write(f"{size // BYTES_PER_MB}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
