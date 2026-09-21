"""Process lifecycle and socket discovery for interactive 3D viewers (MV-05 #10481).

Manages subprocess execution, socket polling, existing listener discovery,
process ownership tracking, and safe teardown for MeshCat WebGL and
Gepetto CORBA viewer processes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlparse

from src.shared.python.contracts import postcondition, precondition

logger = logging.getLogger(__name__)

# Default CORBA nameservice port used by gepetto-gui
GEPETTO_CORBA_PORT: int = 12321

# Default base port for MeshCat WebGL server
MESHCAT_DEFAULT_PORT: int = 7000


class PortCollisionError(RuntimeError):
    """Raised when an attempt is made to start a server on a port that is already in use."""


@dataclass(frozen=True)
class ViewerEndpoint:
    """Network endpoint descriptor for a running viewer server."""

    name: str
    host: str
    port: int
    pid: int | None = None
    is_owned: bool = False
    url: str | None = None


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is actively listening for incoming connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex((host, port)) == 0


def wait_for_port(
    host: str,
    port: int,
    timeout_s: float = 5.0,
    interval_s: float = 0.05,
) -> bool:
    """Poll a host and port until it accepts connections or timeout expires."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if is_port_in_use(host, port):
            return True
        time.sleep(interval_s)
    return is_port_in_use(host, port)


class ViewerProcessManager:
    """Manages lifecycles of viewer server processes with ownership tracking.

    Enforces that unowned processes (e.g., existing gepetto-gui instances started
    by a user or another session) are discovered and attached to rather than
    colliding or being terminated upon cleanup.
    """

    def __init__(self) -> None:
        self._owned_processes: dict[int, subprocess.Popen[str]] = {}

    def is_owned(self, pid: int | None) -> bool:
        """Return True if the specified PID was started and is managed by this instance."""
        return pid is not None and pid in self._owned_processes

    @precondition(lambda self, name, cmd, **_: bool(name) and len(cmd) > 0)
    def start_server(
        self,
        name: str,
        cmd: Sequence[str],
        host: str = "127.0.0.1",
        port: int = GEPETTO_CORBA_PORT,
        timeout_s: float = 5.0,
        interval_s: float = 0.05,
        reuse_existing: bool = False,
        env: dict[str, str] | None = None,
    ) -> ViewerEndpoint:
        """Start a viewer background server or attach to an existing listener.

        Args:
            name: Identifier for the server (e.g. 'gepetto', 'meshcat').
            cmd: Command line arguments to spawn the process.
            host: Bind host / address.
            port: Network port to monitor.
            timeout_s: Maximum seconds to wait for port readiness.
            interval_s: Polling interval in seconds.
            reuse_existing: If True and port is already open, attach without error.
            env: Optional environment dictionary.

        Returns:
            ViewerEndpoint describing the connection.

        Raises:
            PortCollisionError: If port is busy and reuse_existing is False.
            RuntimeError: If the spawned process exits prematurely.
            TimeoutError: If the server port does not become ready within timeout.
        """
        if is_port_in_use(host, port):
            if reuse_existing:
                logger.info(
                    "Attaching to existing %s listener on %s:%d (unowned)",
                    name,
                    host,
                    port,
                )
                return ViewerEndpoint(
                    name=name,
                    host=host,
                    port=port,
                    pid=None,
                    is_owned=False,
                )
            raise PortCollisionError(
                f"Port {port} on {host} is already in use by another process."
            )

        logger.info("Spawning %s server: %s", name, " ".join(cmd))
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if proc.pid is not None:
            self._owned_processes[proc.pid] = proc

        if not wait_for_port(host, port, timeout_s=timeout_s, interval_s=interval_s):
            self._handle_startup_timeout(name, host, port, proc, timeout_s)

        logger.info(
            "Server %s is ready on %s:%d (PID %s)",
            name,
            host,
            port,
            proc.pid,
        )
        return ViewerEndpoint(
            name=name,
            host=host,
            port=port,
            pid=proc.pid,
            is_owned=True,
        )

    def _handle_startup_timeout(
        self,
        name: str,
        host: str,
        port: int,
        proc: subprocess.Popen[str],
        timeout_s: float,
    ) -> None:
        """Handle process exit or timeout when waiting for a server port."""
        poll_code = proc.poll()
        if poll_code is not None:
            out, err = "", ""
            try:
                out, err = proc.communicate(timeout=0.5)
            except Exception:
                pass
            if proc.pid in self._owned_processes:
                del self._owned_processes[proc.pid]
            err_detail = err.strip() or out.strip() or f"exit code {poll_code}"
            raise RuntimeError(
                f"Server {name} exited unexpectedly (code {poll_code}): {err_detail}"
            )

        self.stop_server(
            ViewerEndpoint(
                name=name,
                host=host,
                port=port,
                pid=proc.pid,
                is_owned=True,
            )
        )
        raise TimeoutError(
            f"Timed out waiting for {name} server to open port {port} on {host} after {timeout_s}s"
        )

    def stop_server(
        self,
        endpoint: ViewerEndpoint,
        timeout_s: float = 3.0,
    ) -> bool:
        """Safely terminate an owned viewer process.

        If endpoint is unowned (e.g. was discovered pre-existing), termination is
        strictly refused and False is returned.

        Args:
            endpoint: The viewer endpoint to terminate.
            timeout_s: Seconds to wait for SIGTERM before SIGKILL.

        Returns:
            True if an owned process was terminated; False otherwise.
        """
        if not endpoint.is_owned or endpoint.pid is None:
            logger.info(
                "Refusing to terminate unowned viewer endpoint %s (PID %s)",
                endpoint.name,
                endpoint.pid,
            )
            return False

        proc = self._owned_processes.get(endpoint.pid)
        if proc is None:
            logger.warning(
                "PID %s marked owned on endpoint %s but missing from process map",
                endpoint.pid,
                endpoint.name,
            )
            return False

        logger.info(
            "Terminating owned viewer process %s (PID %s)", endpoint.name, proc.pid
        )
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Viewer process %s (PID %s) did not exit after %0.1fs; killing",
                endpoint.name,
                proc.pid,
                timeout_s,
            )
            proc.kill()
            proc.wait(timeout=2.0)
        finally:
            self._owned_processes.pop(endpoint.pid, None)

        return True

    def stop_all(self, timeout_s: float = 3.0) -> None:
        """Terminate all owned viewer processes."""
        for pid in list(self._owned_processes.keys()):
            proc = self._owned_processes.get(pid)
            if proc is not None:
                endpoint = ViewerEndpoint(
                    name=f"pid_{pid}",
                    host="127.0.0.1",
                    port=0,
                    pid=pid,
                    is_owned=True,
                )
                self.stop_server(endpoint, timeout_s=timeout_s)

    @staticmethod
    def resolve_meshcat_url(url_or_port: str | int) -> str:
        """Parse and validate MeshCat URL or port into canonical HTTP URL."""
        if isinstance(url_or_port, int):
            return f"http://127.0.0.1:{url_or_port}/static/"

        parsed = urlparse(str(url_or_port))
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return str(url_or_port)

        raise ValueError(f"Invalid MeshCat URL: {url_or_port}")
