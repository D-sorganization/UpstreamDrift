"""
Motion Capture Module for UpstreamDrift.

This module provides motion capture integration capabilities,
including FreeMoCap sidecar pipeline support.
"""

from .freemocap_ingest import launcher, output_adapter

__all__ = ["launcher", "output_adapter"]