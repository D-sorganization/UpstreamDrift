"""Registry that surfaces CLI providers to the chat header dropdown.

Mirrors the HTTP-API ``ProviderRegistry`` pattern that Tools #2880
introduced. Consumer code (the chat header populator) calls
``AllProviders.list()`` to get a unified view across CLI and HTTP
providers — it never reaches into either registry directly. This
keeps the consumer compliant with Law of Demeter.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.ai.cli_providers.contracts import CliProviderDescriptor
from src.shared.python.ai.cli_providers.discovery import discover_cli_providers

_list = list  # Alias to avoid name clashes with class-level 'list' methods in mypy


class CliProviderRegistry:
    """Maintains the set of CLI providers discovered on this host.

    Discovery runs lazily on first access and the result is cached.
    Call ``refresh()`` to re-run discovery after the user installs a
    new agent.
    """

    def __init__(self) -> None:
        self._descriptors: _list[CliProviderDescriptor] | None = None

    def list(self) -> _list[CliProviderDescriptor]:
        """Return all known CLI provider descriptors."""
        if self._descriptors is None:
            self._descriptors = discover_cli_providers()
        return list(self._descriptors)

    def refresh(self) -> _list[CliProviderDescriptor]:
        """Re-run discovery and return the updated list."""
        self._descriptors = discover_cli_providers()
        return list(self._descriptors)

    def get(self, provider_id: str) -> CliProviderDescriptor | None:
        """Return the descriptor for ``provider_id`` or None.

        Args:
            provider_id: Identifier such as ``"claude-cli"``.

        Raises:
            ValueError: If ``provider_id`` is empty.
        """
        if not provider_id:
            raise ValueError("provider_id must be a non-empty string")
        for descriptor in self.list():
            if descriptor.id == provider_id:
                return descriptor
        return None


@dataclass(frozen=True)
class ProviderEntry:
    """A unified entry for one provider in the chat header dropdown.

    Attributes:
        id: Stable identifier (collision-free across HTTP and CLI).
        name: Display name shown to the user.
        category: ``"http"`` for HTTP-API providers, ``"cli"`` for
            CLI-agent providers. Used by the populator to insert the
            separator between the two sections.
        descriptor: For CLI providers, the underlying
            ``CliProviderDescriptor``. None for HTTP entries.
    """

    id: str
    name: str
    category: str
    descriptor: CliProviderDescriptor | None = None


class AllProviders:
    """Unified read-only view of HTTP + CLI providers for the dropdown.

    The chat header populator depends only on this class — it has no
    knowledge of how HTTP providers or CLI providers are discovered,
    cached, or instantiated. This preserves Law of Demeter and lets
    each registry evolve independently.
    """

    HTTP_CATEGORY = "http"
    CLI_CATEGORY = "cli"

    def __init__(
        self,
        cli_registry: CliProviderRegistry | None = None,
        http_providers: _list[tuple[str, str]] | None = None,
    ) -> None:
        """Build a unified view over HTTP and CLI registries.

        Args:
            cli_registry: Optional injected CLI registry (for tests).
            http_providers: Optional injected HTTP provider list as
                ``(id, name)`` tuples (for tests). When ``None``, the
                consumer is expected to pass the live list via
                ``set_http_providers()``.
        """
        self._cli = cli_registry or CliProviderRegistry()
        self._http: _list[tuple[str, str]] = list(http_providers or ())

    def set_http_providers(self, providers: _list[tuple[str, str]]) -> None:
        """Replace the HTTP provider list shown in the dropdown.

        Args:
            providers: ``(id, name)`` tuples. The caller is responsible
                for filtering by availability.
        """
        self._http = list(providers)

    def list(self) -> _list[ProviderEntry]:
        """Return the unified list of providers for the dropdown.

        Order: HTTP entries first (in input order), then CLI entries
        (in discovery order). A consumer that wants a visual separator
        between the categories can detect the transition by watching
        ``category``.

        Namespace collisions are avoided by prefixing CLI ids with
        ``cli:`` when an HTTP provider with the same id exists.
        """
        http_ids = {provider_id for provider_id, _ in self._http}
        entries: list[ProviderEntry] = [
            ProviderEntry(id=pid, name=pname, category=self.HTTP_CATEGORY)
            for pid, pname in self._http
        ]
        for descriptor in self._cli.list():
            entry_id = descriptor.id
            if entry_id in http_ids:
                entry_id = f"cli:{entry_id}"
            entries.append(
                ProviderEntry(
                    id=entry_id,
                    name=descriptor.name,
                    category=self.CLI_CATEGORY,
                    descriptor=descriptor,
                )
            )
        return entries

    def cli_entries(self) -> _list[ProviderEntry]:
        """Return only the CLI portion of the unified list."""
        return [e for e in self.list() if e.category == self.CLI_CATEGORY]
