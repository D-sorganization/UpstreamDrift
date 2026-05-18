"""Simulation Data Store — professional-grade CRUD for simulation run data.

Public surface::

    from src.shared.python.simulation_store import SimulationDataStore

Implements Epic #5396: Professional-Grade Simulation Data Management Library.
"""

from __future__ import annotations

from src.shared.python.simulation_store._store import SimulationDataStore

__all__ = ["SimulationDataStore"]
