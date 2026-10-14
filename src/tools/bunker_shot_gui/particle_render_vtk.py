"""3-D particles in the canonical VTK viewport (issue #9545, folded-in #9693).

Draws a :class:`~.particle_playback.ParticleFrameView` into a PyVista
``Plotter`` -- either one this module opens, or the plotter of the
canonical shot viewport (:attr:`.render3d_vtk.VtkSceneArtists.plotter`),
so particles share its millimetre world frame, camera and stamp
conventions.

Particles are drawn with one ``vtkGlyph3DMapper`` over one sphere source:
VTK instances the sphere per point on the GPU, so a frame update rewrites
one point array, not a mesh per grain. Particles that are inactive or lost
in a frame are masked out rather than drawn at a NaN position.

PyVista and VTK are imported lazily, through :func:`.render3d_vtk.require_pyvista`,
never at module load. When the extra is missing, or a render window cannot
be opened (no OpenGL), :func:`open_particle_viewport` returns a
:class:`ParticleViewportUnavailable` that states the reason; the headless
playback model keeps serving frames either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .render3d_vtk import PyVistaNotAvailableError, pyvista_available, require_pyvista
from .shot3d import CameraPreset

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyvista as pv

    from .particle_playback import BodyPose, ParticleFrameView, ParticlePlayback

__all__ = [
    "CLIP_DESCRIPTION",
    "ParticleGlyphLayer",
    "ParticleViewport",
    "ParticleViewportUnavailable",
    "open_particle_viewport",
]

_MM_PER_M = 1e3
_DEFAULT_WINDOW_SIZE = (960, 720)
_PARTICLE_COLOUR = "#c2a36b"
_BODY_COLOURS = {"club": "#5a5a5a", "ball": "#f5f5f5"}
_BODY_FALLBACK_COLOUR = "#1f4e79"
_BODY_MARKER_RADIUS_MM = 1.0
_SPHERE_RESOLUTION = 12
_STAMP_NAME = "particle_stamp"
_STAMP_FONT_PT = 9
_RADIUS_ARRAY = "radius_mm"
_MASK_ARRAY = "visible"
_VIEW_UP = (0.0, 0.0, 1.0)
_CLIP_NORMAL = (0.0, 1.0, 0.0)
CLIP_DESCRIPTION = "clip at y = 0 mm: hide particles on the -y side of the x-z plane"
"""What the clipping toggle does, in the words the panel shows."""


@dataclass(frozen=True)
class ParticleViewportUnavailable:
    """Why the 3-D particle viewport could not open."""

    reason: str

    def describe(self) -> str:
        """The sentence shown where the viewport would have been."""
        return (
            "3-D particle viewport unavailable: "
            f"{self.reason} Frame playback, scrub and stamps still work."
        )


def _vtk_plane() -> Any:
    """A ``vtkPlane``, imported on first use."""
    from vtkmodules.vtkCommonDataModel import vtkPlane

    return vtkPlane()


def _vtk_classes() -> tuple[Any, Any]:
    """``vtkGlyph3DMapper`` and ``vtkActor``, imported on first use."""
    from vtkmodules.vtkRenderingCore import vtkActor, vtkGlyph3DMapper

    return vtkGlyph3DMapper, vtkActor


class ParticleGlyphLayer:
    """Instanced particle glyphs plus body markers on one plotter."""

    def __init__(self, plotter: pv.Plotter, view: ParticleFrameView) -> None:
        """Build the layer at ``view``'s frame.

        Raises:
            PyVistaNotAvailableError: If PyVista is not installed.
        """
        pv = require_pyvista()
        mapper_class, actor_class = _vtk_classes()
        self._pv = pv
        self._plotter = plotter
        count = int(view.grain_ids.size)
        self._points = pv.PolyData(np.zeros((count, 3)))
        self._points.point_data[_RADIUS_ARRAY] = view.radius_m * _MM_PER_M
        self._points.point_data[_MASK_ARRAY] = np.zeros(count, dtype=np.uint8)
        self._mapper = mapper_class()
        self._mapper.SetInputData(self._points)
        self._mapper.SetSourceData(
            pv.Sphere(
                radius=1.0,
                theta_resolution=_SPHERE_RESOLUTION,
                phi_resolution=_SPHERE_RESOLUTION,
            )
        )
        self._mapper.SetScaleArray(_RADIUS_ARRAY)
        self._mapper.SetScaleModeToScaleByMagnitude()
        self._mapper.SetMaskArray(_MASK_ARRAY)
        self._mapper.MaskingOn()
        self._mapper.ScalarVisibilityOff()
        actor = actor_class()
        actor.SetMapper(self._mapper)
        actor.GetProperty().SetColor(pv.Color(_PARTICLE_COLOUR).float_rgb)
        plotter.add_actor(actor, name="particles", reset_camera=False)
        self._bodies = {pose.name: self._add_body(pose) for pose in view.bodies}
        self.update(view)

    def _add_body(self, pose: BodyPose) -> Any:
        colour = _BODY_COLOURS.get(pose.name, _BODY_FALLBACK_COLOUR)
        marker = self._pv.Sphere(radius=_BODY_MARKER_RADIUS_MM)
        return self._plotter.add_mesh(
            marker, color=colour, name=f"body_{pose.name}", reset_camera=False
        )

    @property
    def mapper(self) -> Any:
        """The instancing ``vtkGlyph3DMapper``."""
        return self._mapper

    @property
    def glyph_count(self) -> int:
        """Particles the layer can draw (the decimated roster)."""
        return int(self._points.n_points)

    def points_mm(self) -> NDArray[np.float64]:
        """Glyph centres [mm]; masked particles hold a finite placeholder."""
        return np.asarray(self._points.points, dtype=np.float64)

    def visible_mask(self) -> NDArray[np.bool_]:
        """Which glyphs are drawn this frame."""
        return np.asarray(self._points.point_data[_MASK_ARRAY], dtype=bool)

    def body_visible(self, name: str) -> bool:
        """Whether the named body's marker is drawn this frame."""
        return bool(self._bodies[name].GetVisibility())

    def set_clipping(self, enabled: bool) -> None:
        """Clip the glyphs at the ``y = 0`` plane, or stop clipping.

        Postcondition: exactly one clipping plane when ``enabled``, none
        otherwise. Body markers are never clipped.
        """
        self._mapper.RemoveAllClippingPlanes()
        if enabled:
            plane = _vtk_plane()
            plane.SetOrigin(0.0, 0.0, 0.0)
            plane.SetNormal(*_CLIP_NORMAL)
            self._mapper.AddClippingPlane(plane)

    def update(self, view: ParticleFrameView) -> None:
        """Move glyphs and bodies to ``view`` and redraw the stamp.

        Raises:
            ValueError: If ``view`` draws a different number of particles.
        """
        if int(view.grain_ids.size) != self.glyph_count:
            raise ValueError(
                f"the layer draws {self.glyph_count} particles; the view has "
                f"{view.grain_ids.size}"
            )
        centres = np.where(view.active[:, None], view.positions_m, 0.0) * _MM_PER_M
        self._points.points = centres
        self._points.point_data[_MASK_ARRAY] = view.active.astype(np.uint8)
        self._points.Modified()
        for pose in view.bodies:
            self._place_body(pose)
        self._plotter.add_text(
            "\n".join(view.stamp_lines),
            position="upper_left",
            font_size=_STAMP_FONT_PT,
            name=_STAMP_NAME,
        )

    def _place_body(self, pose: BodyPose) -> None:
        actor = self._bodies[pose.name]
        if pose.position_m is None:
            actor.VisibilityOff()
            return
        actor.SetPosition(*(pose.position_m * _MM_PER_M))
        actor.VisibilityOn()


class ParticleViewport:
    """A playback bound to a glyph layer: show a frame, get pixels."""

    def __init__(
        self,
        playback: ParticlePlayback,
        plotter: pv.Plotter,
        *,
        owns_plotter: bool,
    ) -> None:
        self._playback = playback
        self._plotter = plotter
        self._owns = owns_plotter
        view = playback.view()
        self._stamp = "\n".join(view.stamp_lines)
        self._layer = ParticleGlyphLayer(plotter, view)
        self._camera: CameraPreset | None = None
        self._clipping = False
        if owns_plotter:
            plotter.reset_camera()

    @property
    def layer(self) -> ParticleGlyphLayer:
        """The drawn glyphs and bodies."""
        return self._layer

    @property
    def plotter(self) -> pv.Plotter:
        """The plotter drawn into."""
        return self._plotter

    @property
    def stamp_text(self) -> str:
        """The stamp of the frame last shown."""
        return self._stamp

    @property
    def camera(self) -> CameraPreset | None:
        """The named view last applied; ``None`` until one is."""
        return self._camera

    @property
    def clipping(self) -> bool:
        """Whether the particle glyphs are clipped at ``y = 0``."""
        return self._clipping

    def set_camera(self, camera: CameraPreset | str) -> None:
        """Look at the scene from one of the shot's named views.

        The eye direction is :attr:`~.shot3d.CameraPreset.eye_direction`,
        the same backend-neutral vector the shot viewport uses, framed on
        what this plotter draws.

        Raises:
            ValueError: If ``camera`` is not a preset name.
        """
        chosen = CameraPreset(camera)
        self._plotter.view_vector(
            np.asarray(chosen.eye_direction, dtype=np.float64), viewup=_VIEW_UP
        )
        self._camera = chosen

    def set_clipping(self, enabled: bool) -> None:
        """Turn the ``y = 0`` clipping plane on or off (see :data:`CLIP_DESCRIPTION`)."""
        self._layer.set_clipping(bool(enabled))
        self._clipping = bool(enabled)

    def show_frame(self, index: int) -> None:
        """Seek the playback to ``index``, update and render."""
        self._playback.seek_frame(index)
        view = self._playback.view()
        self._stamp = "\n".join(view.stamp_lines)
        self._layer.update(view)
        self._plotter.render()

    def image_array(self) -> NDArray[np.uint8]:
        """The rendered frame as an ``(H, W, 3)`` array."""
        self._plotter.render()
        return np.asarray(self._plotter.screenshot(return_img=True), dtype=np.uint8)

    def close(self) -> None:
        """Release the render window if this viewport opened it."""
        if self._owns:
            self._plotter.close()


def open_particle_viewport(
    playback: ParticlePlayback,
    *,
    plotter: pv.Plotter | None = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW_SIZE,
) -> ParticleViewport | ParticleViewportUnavailable:
    """Open the particle viewport, or say exactly why it cannot open.

    Args:
        playback: The transport whose frames are drawn.
        plotter: An existing plotter, e.g. the canonical shot viewport's
            :attr:`~.render3d_vtk.VtkSceneArtists.plotter`; ``None`` opens
            an offscreen one this viewport owns.
        window_size: Pixel size of an owned plotter.

    Returns:
        The viewport, or the fallback carrying the reason.
    """
    if not pyvista_available():
        return ParticleViewportUnavailable(
            "pyvista is not installed. Install the viz3d extra: "
            "pip install 'upstream-drift[viz3d]'."
        )
    try:
        pv = require_pyvista()
        target = plotter or pv.Plotter(off_screen=True, window_size=list(window_size))
    except (PyVistaNotAvailableError, RuntimeError, OSError) as error:
        return ParticleViewportUnavailable(f"the render window did not open ({error}).")
    return ParticleViewport(playback, target, owns_plotter=plotter is None)
