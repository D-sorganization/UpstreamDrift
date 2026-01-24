"""Recording library and database management for golf swing analysis.

Provides:
- SQLite database for metadata
- Recording organization and search
- Tagging and filtering
- Import/export library

PERFORMANCE FIX: Uses connection pooling to avoid repeated connect/disconnect overhead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import shutil
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


# PERFORMANCE FIX: Thread-local connection pool
class ConnectionPool:
    """Thread-safe SQLite connection pool.

    Maintains one connection per thread to avoid connection overhead
    while ensuring thread safety.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()

    def get_connection(self) -> sqlite3.Connection:
        """Get connection for current thread."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
        return self._local.connection  # type: ignore[no-any-return]

    def close_all(self) -> None:
        """Close all connections (call on shutdown)."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


@dataclass
class RecordingMetadata:
    """Metadata for a golf swing recording."""

    id: int | None = None
    filename: str = ""
    golfer_name: str = "Unknown"
    date_recorded: str = ""  # ISO format
    club_type: str = "Driver"  # Driver, Iron, Wedge, Putter, Other
    model_name: str = ""
    swing_type: str = "Practice"  # Practice, Competition, Drill, Test
    rating: int = 0  # 0-5 stars
    tags: str = ""  # Comma-separated
    notes: str = ""
    duration: float = 0.0
    peak_club_speed: float = 0.0
    num_frames: int = 0
    checksum: str = ""  # MD5 of data file


class RecordingLibrary:
    """Manage a library of golf swing recordings."""

    def __init__(self, library_path: str = "recordings") -> None:
        """Initialize library.

        Args:
            library_path: Directory for recordings and database
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(exist_ok=True)

        self.db_path = self.library_path / "library.db"

        # PERFORMANCE FIX: Use connection pool instead of repeated connect/disconnect
        self._connection_pool = ConnectionPool(str(self.db_path))

        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection from pool."""
        return self._connection_pool.get_connection()

    def close(self) -> None:
        """Close all database connections."""
        self._connection_pool.close_all()

    def _is_relative_to(self, path: Path, other: Path) -> bool:
        """Check if path is relative to other (Python < 3.9 compat)."""
        try:
            path.relative_to(other)
        except ValueError:
            return False
        else:
            return True

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                golfer_name TEXT,
                date_recorded TEXT,
                club_type TEXT,
                model_name TEXT,
                swing_type TEXT,
                rating INTEGER DEFAULT 0,
                tags TEXT,
                notes TEXT,
                duration REAL,
                peak_club_speed REAL,
                num_frames INTEGER,
                checksum TEXT
            )
        """,
        )

        conn.commit()

    def _insert_recording_db(self, metadata: RecordingMetadata) -> int:
        """Insert recording into database."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO recordings (
                filename, golfer_name, date_recorded, club_type, model_name,
                swing_type, rating, tags, notes, duration, peak_club_speed,
                num_frames, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.filename,
                metadata.golfer_name,
                metadata.date_recorded,
                metadata.club_type,
                metadata.model_name,
                metadata.swing_type,
                metadata.rating,
                metadata.tags,
                metadata.notes,
                metadata.duration,
                metadata.peak_club_speed,
                metadata.num_frames,
                metadata.checksum,
            ),
        )

        recording_id = cursor.lastrowid
        conn.commit()

        if recording_id is None:
            msg = "Failed to get recording ID from database"
            raise RuntimeError(msg)

        return recording_id

    def add_recording(
        self,
        data_file: str,
        metadata: RecordingMetadata,
        copy_to_library: bool = True,
    ) -> int:
        """Add a recording to the library.

        Args:
            data_file: Path to recording data file (JSON or CSV)
            metadata: Recording metadata
            copy_to_library: If True, copy file to library directory

        Returns:
            Recording ID

        Raises:
            FileNotFoundError: If data_file does not exist
            ValueError: If filename is invalid (empty, ".", or "..")
            RuntimeError: If failed to get recording ID from database
        """
        data_path = Path(data_file)

        if not data_path.exists():
            msg = f"Recording file not found: {data_path}"
            raise FileNotFoundError(msg)

        # Generate filename if not provided
        if not metadata.filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize components to prevent invalid filenames
            safe_golfer = re.sub(r"[^\w\-]", "_", metadata.golfer_name)
            safe_club = re.sub(r"[^\w\-]", "_", metadata.club_type)
            metadata.filename = f"{timestamp}_{safe_golfer}_{safe_club}.json"

        # Compute checksum
        metadata.checksum = self._compute_checksum(data_path)

        # Copy file to library if requested
        if copy_to_library:
            self._save_file_to_library(data_path, metadata)
        else:
            # Store relative or absolute path
            metadata.filename = str(data_path)

        # Add to database
        return self._insert_recording_db(metadata)

    def _save_file_to_library(  # noqa: PLR0915
        self,
        data_path: Path,
        metadata: RecordingMetadata,
    ) -> None:
        """Save file to library with sanitization."""
        # Sanitize filename to prevent path traversal
        if not metadata.filename or not metadata.filename.strip():
            msg = "Filename cannot be empty"
            raise ValueError(msg)

        filename = Path(metadata.filename).name
        if not filename or filename in (".", ".."):
            msg = 'Filename cannot be empty, ".", or ".."'
            raise ValueError(msg)

        metadata.filename = filename
        dest_file = self.library_path / metadata.filename

        # Security: Prevent writing if destination is a symlink (avoid TOCTOU/poisoning)
        if dest_file.is_symlink():
            msg = f"Security violation: Destination '{filename}' is a symbolic link"
            raise ValueError(msg)

        # Final safety check: ensure destination is within library
        try:
            resolved_dest = dest_file.resolve()
            resolved_lib = self.library_path.resolve()
        except (OSError, RuntimeError) as e:
            msg = f"Invalid filename: {e}"
            logger.debug(msg)
            raise ValueError(msg) from e

        # Check for path traversal
        is_safe = self._is_relative_to(resolved_dest, resolved_lib)

        if not is_safe:
            msg = (
                f"Security violation: Attempt to save file '{filename}' outside library"
            )
            logger.warning(msg)
            raise ValueError(msg)

        # Secure copy: write to temp file then atomic replace (avoid TOCTOU races)
        # We write to a .tmp file which we create, then replace the target.
        # If dest_file was swapped to a symlink by an attacker, replace() will overwrite
        # the symlink itself, not the target it points to.
        # Note: We compare absolute paths to avoid copying if source == dest.
        if dest_file.absolute() != data_path.absolute():
            # SEC-006: Use SHA-256 instead of MD5 for consistency
            timestamp_hash = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[
                :8
            ]
            temp_name = f".tmp_{filename}_{timestamp_hash}"
            temp_dest = self.library_path / temp_name
            try:
                shutil.copy2(data_path, temp_dest)
                # Atomic replacement
                temp_dest.replace(dest_file)
            except Exception:
                # Cleanup temp file on failure
                if temp_dest.exists():
                    temp_dest.unlink()
                raise

    def get_recording(self, recording_id: int) -> RecordingMetadata | None:
        """Get recording metadata by ID.

        Args:
            recording_id: Recording ID

        Returns:
            RecordingMetadata or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_metadata(row)
        return None

    def update_recording(self, metadata: RecordingMetadata) -> bool:
        """Update recording metadata.

        Args:
            metadata: Updated metadata (must have valid ID)

        Returns:
            True if successful
        """
        if metadata.id is None:
            return False

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE recordings SET
                filename = ?,
                golfer_name = ?,
                date_recorded = ?,
                club_type = ?,
                model_name = ?,
                swing_type = ?,
                rating = ?,
                tags = ?,
                notes = ?,
                duration = ?,
                peak_club_speed = ?,
                num_frames = ?,
                checksum = ?
            WHERE id = ?
        """,
            (
                metadata.filename,
                metadata.golfer_name,
                metadata.date_recorded,
                metadata.club_type,
                metadata.model_name,
                metadata.swing_type,
                metadata.rating,
                metadata.tags,
                metadata.notes,
                metadata.duration,
                metadata.peak_club_speed,
                metadata.num_frames,
                metadata.checksum,
                metadata.id,
            ),
        )

        success = cursor.rowcount > 0
        conn.commit()

        return success

    def delete_recording(self, recording_id: int, delete_file: bool = False) -> bool:
        """Delete a recording.

        Args:
            recording_id: Recording ID
            delete_file: If True, also delete the data file (ONLY if within library)

        Returns:
            True if successful
        """
        if delete_file:
            metadata = self.get_recording(recording_id)
            if metadata:
                file_path = self.library_path / metadata.filename

                # Security check: Ensure file is within library (prevent traversal)
                if file_path.exists():
                    try:
                        resolved_file = file_path.resolve()
                        resolved_lib = self.library_path.resolve()

                        # Check if file is in library
                        is_safe = self._is_relative_to(resolved_file, resolved_lib)

                        if not is_safe:
                            # Security violation: Attempt to delete file outside library
                            # We block deletion of the file, but allow deletion from DB
                            logger.warning(
                                "Skipping file deletion for '%s' (outside library)",
                                Path(metadata.filename).name,
                            )
                        else:
                            file_path.unlink()
                    except (OSError, ValueError, RuntimeError) as e:
                        # Handle filesystem and path resolution errors
                        logger.debug(
                            "Failed to delete file '%s': %s",
                            Path(metadata.filename).name,
                            str(e),
                        )
                        # We continue to delete from DB even if file delete fails

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))

        success = cursor.rowcount > 0
        conn.commit()

        return success

    def search_recordings(
        self,
        golfer_name: str | None = None,
        club_type: str | None = None,
        swing_type: str | None = None,
        min_rating: int = 0,
        tags: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[RecordingMetadata]:
        """Search recordings with filters.

        Args:
            golfer_name: Filter by golfer name (partial match)
            club_type: Filter by club type
            swing_type: Filter by swing type
            min_rating: Minimum rating (0-5)
            tags: List of required tags
            date_from: Minimum date (ISO format)
            date_to: Maximum date (ISO format)

        Returns:
            List of matching RecordingMetadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM recordings WHERE 1=1"
        params: list[str | int] = []

        if golfer_name:
            query += " AND golfer_name LIKE ?"
            params.append(f"%{golfer_name}%")

        if club_type:
            query += " AND club_type = ?"
            params.append(club_type)

        if swing_type:
            query += " AND swing_type = ?"
            params.append(swing_type)

        if min_rating > 0:
            query += " AND rating >= ?"
            params.append(min_rating)

        if date_from:
            query += " AND date_recorded >= ?"
            params.append(date_from)

        if date_to:
            query += " AND date_recorded <= ?"
            params.append(date_to)

        query += " ORDER BY date_recorded DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = [self._row_to_metadata(row) for row in rows]

        # Filter by tags if specified
        if tags:
            results = [
                r for r in results if all(tag in r.tags.split(",") for tag in tags)
            ]

        return results

    def get_all_recordings(self) -> list[RecordingMetadata]:
        """Get all recordings.

        Returns:
            List of all RecordingMetadata
        """
        return self.search_recordings()

    def get_statistics(self) -> dict[str, Any]:
        """Get library statistics.

        PERFORMANCE FIX: Combined multiple queries into fewer round trips.

        Returns:
            Dictionary with statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # PERFORMANCE FIX: Combine basic stats into single query
        cursor.execute("""
            SELECT
                COUNT(*) as total_count,
                AVG(CASE WHEN rating > 0 THEN rating ELSE NULL END) as avg_rating,
                MIN(CASE WHEN peak_club_speed > 0
                    THEN peak_club_speed ELSE NULL END) as min_speed,
                MAX(CASE WHEN peak_club_speed > 0
                    THEN peak_club_speed ELSE NULL END) as max_speed,
                AVG(CASE WHEN peak_club_speed > 0
                    THEN peak_club_speed ELSE NULL END) as avg_speed
            FROM recordings
        """)
        stats_row = cursor.fetchone()
        total_count = stats_row[0]
        avg_rating = stats_row[1] or 0.0
        speed_stats = (stats_row[2], stats_row[3], stats_row[4])

        # Group by queries (still need separate queries for these)
        cursor.execute(
            """
            SELECT club_type, COUNT(*) FROM recordings
            GROUP BY club_type
        """,
        )
        by_club = dict(cursor.fetchall())

        cursor.execute(
            """
            SELECT swing_type, COUNT(*) FROM recordings
            GROUP BY swing_type
        """,
        )
        by_swing_type = dict(cursor.fetchall())

        return {
            "total_recordings": total_count,
            "by_club_type": by_club,
            "by_swing_type": by_swing_type,
            "average_rating": float(avg_rating),
            "speed_stats": {
                "min": speed_stats[0] or 0.0,
                "max": speed_stats[1] or 0.0,
                "average": speed_stats[2] or 0.0,
            },
        }

    def export_library(self, output_file: str) -> None:
        """Export entire library to JSON.

        Args:
            output_file: Output JSON file path
        """
        recordings = self.get_all_recordings()
        data = {
            "library_path": str(self.library_path),
            "export_date": datetime.now().isoformat(),
            "recordings": [asdict(r) for r in recordings],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def import_library(self, input_file: str, merge: bool = True) -> None:
        """Import library from JSON.

        Args:
            input_file: Input JSON file path
            merge: If True, merge with existing library; if False, replace
        """
        with open(input_file) as f:
            data = json.load(f)

        if not merge:
            # Clear existing database
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM recordings")
            conn.commit()

        # Add all recordings
        for rec_dict in data.get("recordings", []):
            rec_dict.pop("id", None)  # Remove ID to generate new one
            metadata = RecordingMetadata(**rec_dict)

            # Check if file exists
            file_path = self.library_path / metadata.filename
            if not file_path.exists():
                continue

            with contextlib.suppress(Exception):
                self.add_recording(str(file_path), metadata, copy_to_library=False)

    def get_unique_values(self, field: str) -> list[str]:
        """Get all unique values for a field.

        Args:
            field: Field name (e.g., 'golfer_name', 'club_type')

        Returns:
            List of unique values
        """
        # Whitelist allowed fields to prevent SQL injection
        allowed_fields = {
            "golfer_name",
            "club_type",
            "model_name",
            "swing_type",
            "tags",
            "notes",
        }

        if field not in allowed_fields:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"SELECT DISTINCT {field} FROM recordings WHERE {field} != ''")
        values = [row[0] for row in cursor.fetchall()]

        return sorted(values)

    def _row_to_metadata(self, row: tuple) -> RecordingMetadata:
        """Convert database row to RecordingMetadata."""
        return RecordingMetadata(
            id=row[0],
            filename=row[1],
            golfer_name=row[2] or "Unknown",
            date_recorded=row[3] or "",
            club_type=row[4] or "Driver",
            model_name=row[5] or "",
            swing_type=row[6] or "Practice",
            rating=row[7] or 0,
            tags=row[8] or "",
            notes=row[9] or "",
            duration=row[10] or 0.0,
            peak_club_speed=row[11] or 0.0,
            num_frames=row[12] or 0,
            checksum=row[13] or "",
        )

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of file.

        SEC-006: Replaced MD5 with SHA-256 to prevent collision attacks.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_recording_path(self, metadata: RecordingMetadata) -> Path:
        """Get full path to recording data file.

        Args:
            metadata: Recording metadata

        Returns:
            Path to data file
        """
        file_path = Path(metadata.filename)
        if not file_path.is_absolute():
            file_path = self.library_path / metadata.filename
        return file_path


def create_metadata_from_recording(
    data_dict: dict[str, Any],
    golfer_name: str = "Unknown",
    club_type: str = "Driver",
    swing_type: str = "Practice",
) -> RecordingMetadata:
    """Create metadata from recording data dictionary.

    Args:
        data_dict: Recording data (with times, states, etc.)
        golfer_name: Golfer name
        club_type: Club type
        swing_type: Swing type

    Returns:
        RecordingMetadata with computed statistics
    """
    times = data_dict.get("times", [])
    duration = times[-1] - times[0] if len(times) > 1 else 0.0

    # Try to find peak club speed
    peak_speed = 0.0
    if "club_head_speed" in data_dict:
        speeds = data_dict["club_head_speed"]
        if len(speeds) > 0:
            peak_speed = float(max(speeds))

    return RecordingMetadata(
        golfer_name=golfer_name,
        date_recorded=datetime.now().isoformat(),
        club_type=club_type,
        model_name=data_dict.get("model_name", ""),
        swing_type=swing_type,
        duration=duration,
        peak_club_speed=peak_speed,
        num_frames=len(times),
    )
