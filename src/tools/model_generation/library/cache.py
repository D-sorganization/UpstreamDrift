"""
Model caching system for offline access and performance.

Provides persistent caching of downloaded models with
integrity verification and cleanup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in the model cache."""

    model_id: str
    source_url: str | None
    local_path: Path
    checksum: str | None = None
    cached_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    is_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "source_url": self.source_url,
            "local_path": str(self.local_path),
            "checksum": self.checksum,
            "cached_at": self.cached_at,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            source_url=data.get("source_url"),
            local_path=Path(data["local_path"]),
            checksum=data.get("checksum"),
            cached_at=data.get("cached_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            size_bytes=data.get("size_bytes", 0),
            is_complete=data.get("is_complete", True),
        )


@dataclass
class CacheConfig:
    """Configuration for model cache."""

    # Cache location
    cache_dir: Path = field(
        default_factory=lambda: Path.home() / ".model_generation" / "cache"
    )

    # Size limits
    max_size_mb: int = 1000  # 1GB default
    max_entries: int = 100

    # Cleanup settings
    cleanup_threshold: float = 0.9  # Start cleanup at 90% capacity
    min_age_days: int = 7  # Don't delete files newer than this

    # Verification
    verify_checksums: bool = True


class ModelCache:
    """
    Persistent cache for downloaded models.

    Features:
    - LRU-based eviction
    - Checksum verification
    - Automatic cleanup
    - Offline access tracking
    """

    INDEX_FILE = "cache_index.json"

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize cache.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._entries: dict[str, CacheEntry] = {}

        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_path = self.config.cache_dir / self.INDEX_FILE
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                for entry_data in data.get("entries", []):
                    entry = CacheEntry.from_dict(entry_data)
                    if entry.local_path.exists():
                        self._entries[entry.model_id] = entry
                logger.debug(f"Loaded {len(self._entries)} cached models")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")

    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_path = self.config.cache_dir / self.INDEX_FILE
        try:
            data = {
                "entries": [e.to_dict() for e in self._entries.values()],
                "version": "1.0",
            }
            index_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")

    def get(self, model_id: str) -> CacheEntry | None:
        """
        Get a cached model.

        Args:
            model_id: Model identifier

        Returns:
            CacheEntry if cached, None otherwise
        """
        entry = self._entries.get(model_id)
        if entry and entry.local_path.exists():
            entry.last_accessed = time.time()
            self._save_index()
            return entry
        return None

    def put(
        self,
        model_id: str,
        local_path: Path,
        source_url: str | None = None,
        compute_checksum: bool = True,
    ) -> CacheEntry:
        """
        Add a model to the cache.

        Args:
            model_id: Model identifier
            local_path: Path to cached files
            source_url: Original source URL
            compute_checksum: If True, compute file checksum

        Returns:
            Created CacheEntry
        """
        # Check if cleanup needed
        self._maybe_cleanup()

        # Compute checksum if requested
        checksum = None
        if compute_checksum and local_path.is_file():
            checksum = self._compute_checksum(local_path)

        # Calculate size
        size = self._get_size(local_path)

        entry = CacheEntry(
            model_id=model_id,
            source_url=source_url,
            local_path=local_path,
            checksum=checksum,
            size_bytes=size,
        )

        self._entries[model_id] = entry
        self._save_index()

        return entry

    def remove(self, model_id: str, delete_files: bool = True) -> bool:
        """
        Remove a model from the cache.

        Args:
            model_id: Model to remove
            delete_files: If True, delete cached files

        Returns:
            True if removed
        """
        entry = self._entries.get(model_id)
        if not entry:
            return False

        if delete_files:
            try:
                if entry.local_path.is_dir():
                    shutil.rmtree(entry.local_path)
                elif entry.local_path.is_file():
                    entry.local_path.unlink()
                    # Also try to remove parent if empty
                    parent = entry.local_path.parent
                    if parent != self.config.cache_dir and not any(parent.iterdir()):
                        parent.rmdir()
            except Exception as e:
                logger.warning(f"Failed to delete cached files: {e}")

        del self._entries[model_id]
        self._save_index()
        return True

    def contains(self, model_id: str) -> bool:
        """Check if model is cached."""
        entry = self._entries.get(model_id)
        return entry is not None and entry.local_path.exists()

    def verify(self, model_id: str) -> bool:
        """
        Verify cached model integrity.

        Args:
            model_id: Model to verify

        Returns:
            True if valid, False if corrupted or missing
        """
        entry = self._entries.get(model_id)
        if not entry or not entry.local_path.exists():
            return False

        if not entry.checksum or not self.config.verify_checksums:
            return True

        current_checksum = self._compute_checksum(entry.local_path)
        return current_checksum == entry.checksum

    def get_cache_path(self, model_id: str) -> Path:
        """Get the cache path for a model (may not exist yet)."""
        safe_id = model_id.replace("/", "_").replace("\\", "_")
        return self.config.cache_dir / safe_id

    def get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(e.size_bytes for e in self._entries.values())

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = self.get_total_size()
        return {
            "entry_count": len(self._entries),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.config.max_size_mb,
            "usage_percent": (total_size / (self.config.max_size_mb * 1024 * 1024))
            * 100,
            "cache_dir": str(self.config.cache_dir),
        }

    def _maybe_cleanup(self) -> None:
        """Run cleanup if cache is getting full."""
        total_size = self.get_total_size()
        max_size = self.config.max_size_mb * 1024 * 1024
        threshold = max_size * self.config.cleanup_threshold

        if total_size > threshold or len(self._entries) > self.config.max_entries:
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove old entries to free space."""
        logger.info("Running cache cleanup...")

        # Sort by last accessed (oldest first)
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.last_accessed)

        min_age = time.time() - (self.config.min_age_days * 24 * 60 * 60)
        max_size = self.config.max_size_mb * 1024 * 1024
        target_size = max_size * 0.7  # Clean to 70%

        removed = 0
        for entry in sorted_entries:
            if self.get_total_size() <= target_size:
                break

            # Don't remove recent files
            if entry.cached_at > min_age:
                continue

            self.remove(entry.model_id)
            removed += 1

        logger.info(f"Cleanup removed {removed} entries")

    def clear(self) -> None:
        """Clear entire cache."""
        for model_id in list(self._entries.keys()):
            self.remove(model_id)
        logger.info("Cache cleared")

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_size(self, path: Path) -> int:
        """Get total size of path (file or directory)."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return 0

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._entries)

    def __contains__(self, model_id: str) -> bool:
        """Check if model is cached."""
        return self.contains(model_id)
