"""
Tests for DRY unification: _HUMANOID_PRESETS module-level constant.

Verifies that:
- _HUMANOID_PRESETS is a module-level constant in model_generation
- Both quick_urdf() and quick_build() use the same preset data
- The preset dict has exactly the expected keys and values
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helper to reload the module cleanly for each test that inspects globals
# ---------------------------------------------------------------------------


def _import_model_generation():
    """Import model_generation, returning the live module."""
    import model_generation

    return model_generation


# ---------------------------------------------------------------------------
# Tests for _HUMANOID_PRESETS constant
# ---------------------------------------------------------------------------


class TestHumanoidPresetsConstant:
    """_HUMANOID_PRESETS must be a module-level dict in model_generation."""

    def test_constant_exists(self):
        """_HUMANOID_PRESETS must exist as a module-level name."""
        import model_generation

        assert hasattr(model_generation, "_HUMANOID_PRESETS"), (
            "_HUMANOID_PRESETS not found in model_generation; "
            "extract the preset dict from quick_urdf/quick_build into a module constant"
        )

    def test_constant_is_dict(self):
        """_HUMANOID_PRESETS must be a dict."""
        import model_generation

        assert isinstance(model_generation._HUMANOID_PRESETS, dict), (
            f"_HUMANOID_PRESETS must be a dict, got {type(model_generation._HUMANOID_PRESETS)}"
        )

    def test_constant_has_exactly_four_keys(self):
        """_HUMANOID_PRESETS must contain exactly four preset keys."""
        import model_generation

        presets = model_generation._HUMANOID_PRESETS
        assert set(presets.keys()) == {
            "athletic",
            "average",
            "heavy",
            "lean",
        }, (
            f"Expected keys {{athletic, average, heavy, lean}}, got {set(presets.keys())}"
        )

    def test_athletic_preset_values(self):
        """athletic preset must set gender_factor=0.7 and shoulder_width_factor=1.1."""
        import model_generation

        athletic = model_generation._HUMANOID_PRESETS["athletic"]
        assert athletic.get("gender_factor") == 0.7, (
            f"athletic gender_factor expected 0.7, got {athletic.get('gender_factor')}"
        )
        assert athletic.get("shoulder_width_factor") == 1.1, (
            f"athletic shoulder_width_factor expected 1.1, got {athletic.get('shoulder_width_factor')}"
        )

    def test_average_preset_values(self):
        """average preset must set gender_factor=0.5 and nothing else."""
        import model_generation

        average = model_generation._HUMANOID_PRESETS["average"]
        assert average.get("gender_factor") == 0.5, (
            f"average gender_factor expected 0.5, got {average.get('gender_factor')}"
        )
        # average should not have extra keys
        assert set(average.keys()) == {"gender_factor"}, (
            f"average preset should only have 'gender_factor', got {set(average.keys())}"
        )

    def test_heavy_preset_values(self):
        """heavy preset must set gender_factor=0.5 and hip_width_factor=1.15."""
        import model_generation

        heavy = model_generation._HUMANOID_PRESETS["heavy"]
        assert heavy.get("gender_factor") == 0.5, (
            f"heavy gender_factor expected 0.5, got {heavy.get('gender_factor')}"
        )
        assert heavy.get("hip_width_factor") == 1.15, (
            f"heavy hip_width_factor expected 1.15, got {heavy.get('hip_width_factor')}"
        )

    def test_lean_preset_values(self):
        """lean preset must set gender_factor=0.5 and shoulder_width_factor=0.95."""
        import model_generation

        lean = model_generation._HUMANOID_PRESETS["lean"]
        assert lean.get("gender_factor") == 0.5, (
            f"lean gender_factor expected 0.5, got {lean.get('gender_factor')}"
        )
        assert lean.get("shoulder_width_factor") == 0.95, (
            f"lean shoulder_width_factor expected 0.95, got {lean.get('shoulder_width_factor')}"
        )


# ---------------------------------------------------------------------------
# Tests that quick_urdf and quick_build use the shared constant
# ---------------------------------------------------------------------------


class TestQuickFunctionsUseSharedPresets:
    """quick_urdf() and quick_build() must read from _HUMANOID_PRESETS, not a local copy."""

    def _make_builder_mock(self):
        """Create a mock ParametricBuilder that records set_parameters calls."""
        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.urdf_xml = "<robot name='test'/>"
        mock_builder.build.return_value = mock_result
        return mock_builder

    def test_quick_urdf_athletic_uses_correct_params(self):
        """quick_urdf with 'athletic' preset must pass gender_factor=0.7, shoulder_width_factor=1.1."""
        import model_generation

        mock_builder = self._make_builder_mock()

        with patch(
            "model_generation.builders.parametric_builder.ParametricBuilder",
            return_value=mock_builder,
        ):
            model_generation.quick_urdf(height_m=1.80, preset="athletic")

        mock_builder.set_parameters.assert_called_once()
        call_kwargs = mock_builder.set_parameters.call_args[1]
        assert call_kwargs.get("gender_factor") == 0.7
        assert call_kwargs.get("shoulder_width_factor") == 1.1

    def test_quick_urdf_heavy_uses_correct_params(self):
        """quick_urdf with 'heavy' preset must pass hip_width_factor=1.15."""
        import model_generation

        mock_builder = self._make_builder_mock()

        with patch(
            "model_generation.builders.parametric_builder.ParametricBuilder",
            return_value=mock_builder,
        ):
            model_generation.quick_urdf(height_m=1.80, preset="heavy")

        call_kwargs = mock_builder.set_parameters.call_args[1]
        assert call_kwargs.get("hip_width_factor") == 1.15

    def test_quick_build_lean_uses_correct_params(self):
        """quick_build with 'lean' preset must pass shoulder_width_factor=0.95."""
        import model_generation

        mock_builder = self._make_builder_mock()

        with patch(
            "model_generation.builders.parametric_builder.ParametricBuilder",
            return_value=mock_builder,
        ):
            model_generation.quick_build(height_m=1.75, preset="lean")

        call_kwargs = mock_builder.set_parameters.call_args[1]
        assert call_kwargs.get("shoulder_width_factor") == 0.95
        assert call_kwargs.get("gender_factor") == 0.5

    def test_quick_urdf_and_quick_build_yield_identical_config_for_same_preset(self):
        """
        For the same preset name, quick_urdf and quick_build must apply the
        identical parameter set (i.e. both read from the same shared dict).
        """
        import model_generation

        urdf_kwargs = {}
        build_kwargs = {}

        mock_builder_urdf = self._make_builder_mock()
        mock_builder_build = self._make_builder_mock()

        def capture_urdf(*a, **kw):
            urdf_kwargs.update(kw)

        def capture_build(*a, **kw):
            build_kwargs.update(kw)

        mock_builder_urdf.set_parameters.side_effect = capture_urdf
        mock_builder_build.set_parameters.side_effect = capture_build

        builders = [mock_builder_urdf, mock_builder_build]
        call_count = [0]

        def builder_factory(*a, **kw):
            idx = call_count[0]
            call_count[0] += 1
            return builders[idx]

        with patch(
            "model_generation.builders.parametric_builder.ParametricBuilder",
            side_effect=builder_factory,
        ):
            model_generation.quick_urdf(height_m=1.80, preset="athletic")
            model_generation.quick_build(height_m=1.80, preset="athletic")

        # Both should have received the same preset-derived kwargs
        # (height_m and mass_kg may differ by default, so only compare preset keys)
        preset_keys = {"gender_factor", "shoulder_width_factor", "hip_width_factor"}
        urdf_preset_kw = {k: v for k, v in urdf_kwargs.items() if k in preset_keys}
        build_preset_kw = {k: v for k, v in build_kwargs.items() if k in preset_keys}
        assert urdf_preset_kw == build_preset_kw, (
            f"quick_urdf and quick_build applied different preset configs: "
            f"urdf={urdf_preset_kw}, build={build_preset_kw}"
        )

    def test_unknown_preset_falls_back_to_no_extra_params(self):
        """An unrecognised preset name must fall through to empty config (no crash)."""
        import model_generation

        mock_builder = self._make_builder_mock()

        with patch(
            "model_generation.builders.parametric_builder.ParametricBuilder",
            return_value=mock_builder,
        ):
            # Should not raise
            model_generation.quick_urdf(height_m=1.80, preset="nonexistent_preset")

        call_kwargs = mock_builder.set_parameters.call_args[1]
        # No extra preset keys should leak in
        extra_keys = {"gender_factor", "shoulder_width_factor", "hip_width_factor"}
        assert not extra_keys.intersection(call_kwargs.keys()), (
            f"Unknown preset should not set any extra params, got {call_kwargs}"
        )


# ---------------------------------------------------------------------------
# Test: _HUMANOID_PRESETS values are immutable-safe (not mutated between calls)
# ---------------------------------------------------------------------------


class TestPresetsNotMutated:
    """Ensure the shared constant is not mutated by quick_urdf / quick_build calls."""

    def test_presets_unchanged_after_quick_urdf(self):
        """The module-level constant must be identical before and after calling quick_urdf."""
        import copy

        import model_generation

        before = copy.deepcopy(model_generation._HUMANOID_PRESETS)

        mock_builder = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.urdf_xml = "<robot/>"
        mock_builder.build.return_value = mock_result

        with patch(
            "model_generation.builders.parametric_builder.ParametricBuilder",
            return_value=mock_builder,
        ):
            model_generation.quick_urdf(preset="athletic")

        assert before == model_generation._HUMANOID_PRESETS, (
            "_HUMANOID_PRESETS was mutated during quick_urdf call"
        )
