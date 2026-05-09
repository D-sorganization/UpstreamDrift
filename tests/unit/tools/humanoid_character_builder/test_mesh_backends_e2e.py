"""
End-to-end tests for MakeHuman and SMPLX mesh backends.

These tests are marked as `slow` and `live_simulation` because they:
- Require external dependencies (torch, trimesh, smplx)
- Require asset files (SMPL-X models, MakeHuman exports)
- Generate real mesh files on disk

Run with: pytest -m "slow or live_simulation" -v
"""

import os
import tempfile
from pathlib import Path

import pytest

from humanoid_character_builder import BodyParameters, CharacterBuilder
from humanoid_character_builder.generators import MeshGeneratorBackend


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
