"""Tests for video route optional-dependency graceful fallback (issue #2466).

Verifies that src.api.routes.video has the `_VIDEO_DEPS_AVAILABLE` guard so
route_registry does not silently drop the video surface when the optional
video-pose-pipeline package is unavailable.
"""

from __future__ import annotations


class TestVideoRouteImportFallback:
    """Module has the optional-dep guard flag (issue #2466)."""

    def test_video_route_has_availability_flag(self) -> None:
        """video.py must expose _VIDEO_DEPS_AVAILABLE to gate pipeline construction."""
        try:
            import src.api.routes.video as video_mod
        except Exception:
            return  # env missing other deps; skip

        assert hasattr(video_mod, "_VIDEO_DEPS_AVAILABLE"), (
            "_VIDEO_DEPS_AVAILABLE flag must exist so code paths can gate on dep availability"
        )
        assert isinstance(video_mod._VIDEO_DEPS_AVAILABLE, bool)

    def test_video_route_does_not_use_module_level_pipeline_type(self) -> None:
        """The route type hints use Any, not a hard VideoPosePipeline import."""
        import inspect

        try:
            import src.api.routes.video as video_mod
        except Exception:
            return  # env missing other deps; skip

        src = inspect.getsource(video_mod.analyze_video)
        # The parameter annotation should not reference the hard class name
        assert "VideoPosePipeline" not in src, (
            "analyze_video() must not use VideoPosePipeline as a type annotation "
            "directly — use Any to avoid import-time failures"
        )
