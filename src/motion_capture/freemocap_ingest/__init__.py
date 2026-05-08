"""
FreeMoCap Ingest Module.

This module provides a sidecar integration with FreeMoCap
for motion capture data ingestion via filesystem-based communication.
"""

from .launcher import FreeMoCapLauncher
from .output_adapter import FreeMoCapOutputAdapter, LandmarkFrame

__all__ = ["FreeMoCapLauncher", "FreeMoCapOutputAdapter", "LandmarkFrame"]