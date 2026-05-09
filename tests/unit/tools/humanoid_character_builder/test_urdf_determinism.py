"""Test URDF generation determinism for humanoid character builder."""

import pytest

from humanoid_character_builder import CharacterBuilder
from humanoid_character_builder.core.body_parameters import BodyParameters
from humanoid_character_builder.presets.loader import list_available_presets
