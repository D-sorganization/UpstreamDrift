"""The VTK/PyVista renderer for the 3-D shot scene (issue #8706, epic #8699).

ADR-0027 evaluates MeshCat, Rerun and VTK/PyVista as canonical 3-D viewport
providers and records an explicit degradation reason when none is
import-discoverable. :mod:`~.render3d` is that degradation: a headless
matplotlib ``Axes3D`` scatter drawn only because nothing better was
installed. This module is the other half -- the concrete VTK adapter ADR-0027
names but does not build, executed once PyVista (which imports ``vtk``, the
provider :data:`~src.shared.python.visualization.viewport.ViewportProvider.VTK`
already detects) is installed as the optional ``viz3d`` extra.

Nothing here changes what a frame is allowed to say. It draws the same
:class:`~.shot3d.ShotScene` :mod:`~.render3d` draws, framed by the same
injected :class:`~.render3d.SceneScale` so switching renderers cannot also
switch which world box or which depth ramp a comparison is judged against,
and it carries the same in-frame validity stamp
(:func:`~.render.validity_stamp`) -- a prettier picture is not licence to
drop the honesty rules :mod:`~.render` established. What changes is *how* the
clubhead is drawn: :mod:`~.render3d` has no mesh to work with and draws the
solver's element centroids as a marker cloud; this module is handed the
:class:`~.bridge.HeadBuild` the centroids came from and poses its actual
watertight triangle mesh per frame, so the head is a solid, lit, depth-buffered
surface rather than a scatter that only *looks* three-dimensional from one
angle.

PyVista is imported lazily, inside :func:`require_pyvista`, never at module
load. So is everything else this module needs at runtime from its own
package and from :mod:`bunkershot3d`: the sibling modules that carry the
real value objects (:mod:`.bridge`, :mod:`.render`, :mod:`.report`,
:mod:`.shot3d`, :mod:`.traces`) all import ``bunkershot3d.solvers``, and
``bunkershot3d`` eagerly imports ``mujoco`` for its grain-scale backends --
and ``mujoco`` touches an OpenGL/OSMesa binding *at that import*, not at
first render. A CI runner with no working GL driver raises there (PyOpenGL's
``_ErrorChecker`` construction fails with
``AttributeError: 'NoneType' object has no attribute 'glGetError'``, seen in
PR #9138's ``unit-test-gate``), so a bare ``import render3d_vtk`` must never
reach it. Every name this module needs from those siblings at runtime is
therefore imported inside the function or method that uses it, mirroring
:func:`require_pyvista`; only type annotations -- safe, thanks to
``from __future__ import annotations`` -- reference them at module scope,
under ``TYPE_CHECKING``.

Importing this module is always safe; only constructing
:class:`VtkSceneArtists` -- or calling :func:`draw_scene_frame_vtk` /
:func:`shot_scene_still_vtk`, which construct one -- requires the extra
(and the rest of the package), and a missing PyVista fails with an install
hint that mirrors the viewport layer's own degradation reasons rather than a
bare ``ModuleNotFoundError``.

Rendering is offscreen throughout (``pyvista.Plotter(off_screen=True)``,
matching PyVista's documented Windows/native-OpenGL offscreen path). This
module produces pixels -- a screenshot or an RGBA array -- for a caller to
place; it embeds nothing in Qt itself, the same boundary ADR-0027 draws
between "a provider is available" and "app-shell embedding is someone else's
follow-up" (`#6805`).
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyvista as pv

    from bunkershot3d.solvers import EnvelopeStatus
    from src.shared.python.visualization.viewport import ViewportOverlayPayload

    from .bridge import HeadBuild
    from .render import ViewportFallback
    from .render3d import SceneScale
    from .shot3d import CameraPreset, ShotScene
    from .traces import ValidityBand

__all__ = [
    "RENDERER",
    "PyVistaNotAvailableError",
    "VtkSceneArtists",
    "draw_scene_frame_vtk",
    "pyvista_available",
    "require_pyvista",
    "shot_scene_still_vtk",
]

RENDERER = "pyvista"
"""What draws the frame once the VTK/PyVista provider is selected."""

_INSTALL_HINT = (
    "pyvista is not installed. Install the viz3d extra: "
    "pip install 'upstream-drift[viz3d]'"
)

_MM_PER_M = 1e3

# Kept numerically in sync with render3d.py's own palette by eye, not by
# import: that module's colour constants are private, and the point of two
# renderers is that a reader can tell them apart from the ``renderer:``
# stamp, not from unrelated colours.
_SURFACE_COLOUR = "#d9c9a3"
_SURFACE_ALPHA = 0.35
_HEAD_COLOUR = "#5a5a5a"
_SOLE_COLOUR = "#1f4e79"
_PATH_COLOUR = "#9b1c1c"
_FLOOR_COLORMAP = "YlOrBr"

_DEFAULT_WINDOW_SIZE = (960, 720)
_CAMERA_RADIUS_FACTOR = 1.5
"""How far past the world box's own diagonal the eye sits, so all three
named views clear the scene without the caller re-tuning per shot."""
_MIN_CAMERA_RADIUS_MM = 50.0

_TRAIL_NAME = "trail"
_STAMP_FONT_PT = 9
_NOTE_FONT_PT = 7


class PyVistaNotAvailableError(RuntimeError):
    """Raised when PyVista is required but not importable."""


def pyvista_available() -> bool:
    """Whether the ``pyvista`` module is importable (mock-tolerant probe)."""
    try:
        return find_spec("pyvista") is not None
    except (ValueError, ModuleNotFoundError):
        return False


def require_pyvista() -> ModuleType:
    """Import and return ``pyvista``, raising with an install hint if absent.

    Returns:
        The imported module, with ``OFF_SCREEN`` set so a caller that forgets
        to pass ``off_screen=True`` to a :class:`~pyvista.Plotter` still gets
        a headless render rather than a window this process has no display
        for.

    Raises:
        PyVistaNotAvailableError: If ``pyvista`` is not import-discoverable.
    """
    if not pyvista_available():
        raise PyVistaNotAvailableError(_INSTALL_HINT)
    module = import_module("pyvista")
    # ``pyvista.OFF_SCREEN`` is a real module attribute, but ``ModuleType``
    # itself declares no such thing, and no stub package is installed for
    # mypy to see the real one (this whole extra is optional).
    module.OFF_SCREEN = True  # type: ignore[attr-defined]
    return module


def _coerce_camera(camera: CameraPreset | str | None) -> CameraPreset:
    """Resolve a ``camera`` argument, importing :class:`~.shot3d.CameraPreset`
    lazily.

    Deferred for the same reason :func:`require_pyvista` defers ``pyvista``:
    see the module docstring. :mod:`.shot3d` imports ``bunkershot3d``, whose
    grain-scale backends import ``mujoco`` at module load, and ``mujoco``
    touches an OpenGL/OSMesa binding at *that* import -- a probe that raises
    on a CI runner with no working GL driver. Nothing about picking or
    validating a camera preset needs that price paid merely for importing
    this module, so this is the one place ``CameraPreset`` is imported for
    real.

    Args:
        camera: The caller's preset, its string name, or ``None`` for the
            default down-the-line view.

    Returns:
        A concrete :class:`~.shot3d.CameraPreset`.

    Raises:
        ValueError: If ``camera`` is a string that names no preset.
    """
    from .shot3d import CameraPreset

    return CameraPreset.DOWN_THE_LINE if camera is None else CameraPreset(camera)


def _hex_to_rgb01(colour: str) -> tuple[float, float, float]:
    """Convert a ``#rrggbb`` string to 0-1 floats, for VTK colour setters.

    Args:
        colour: A ``#rrggbb`` string.

    Returns:
        ``(r, g, b)`` in ``[0, 1]``.

    Raises:
        ValueError: If ``colour`` is not six hex digits after the ``#``.
    """
    digits = colour.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"expected a #rrggbb colour, got {colour!r}")
    red = int(digits[0:2], 16) / 255.0
    green = int(digits[2:4], 16) / 255.0
    blue = int(digits[4:6], 16) / 255.0
    return (red, green, blue)


def _posed_mm(
    body_points_m: NDArray[np.float64],
    rotation: NDArray[np.float64],
    position_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pose body-axis points into world millimetres.

    The same transform :meth:`~.shot3d.ShotScene.head_world_m` uses
    (``p + R c``), so the posed mesh and the posed centroids
    :mod:`~.render3d` draws can never disagree about where the head is.
    """
    rotated = body_points_m @ rotation.T
    return np.asarray((rotated + position_m) * _MM_PER_M, dtype=np.float64)


def _pv_faces(faces: NDArray[np.int64]) -> NDArray[np.int64]:
    """Pad a ``(m, 3)`` triangle index array to PyVista's flat face format."""
    counts = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    return np.asarray(np.hstack([counts, faces]).ravel(), dtype=np.int64)


def _polyline_connectivity(n_points: int) -> NDArray[np.int64]:
    """Return the ``lines`` array for one open polyline over ``n_points``."""
    return np.asarray(np.hstack([[n_points], np.arange(n_points)]), dtype=np.int64)


def _camera_geometry(
    scale: SceneScale, camera: CameraPreset
) -> tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]:
    """Return ``(eye, focal_point, up)`` in millimetres for one named view.

    Fixed from the scene's own :class:`~.render3d.SceneScale`, never from the
    current frame or the mesh's own extent, so switching frames cannot nudge
    the camera and switching cameras cannot reintroduce a per-frame
    autoscale (the #8728 defect :mod:`~.render3d` already refuses).

    Args:
        scale: The fixed world box.
        camera: The named preset.

    Returns:
        The eye position, the focal point, and the up vector PyVista's
        ``Plotter.camera_position`` takes as one three-tuple.
    """
    focal = (
        float(np.mean(scale.x_m)) * _MM_PER_M,
        float(np.mean(scale.y_m)) * _MM_PER_M,
        float(np.mean(scale.z_m)) * _MM_PER_M,
    )
    span_mm = np.array(
        [
            (scale.x_m[1] - scale.x_m[0]) * _MM_PER_M,
            (scale.y_m[1] - scale.y_m[0]) * _MM_PER_M,
            (scale.z_m[1] - scale.z_m[0]) * _MM_PER_M,
        ]
    )
    radius = max(
        float(np.linalg.norm(span_mm)) * _CAMERA_RADIUS_FACTOR, _MIN_CAMERA_RADIUS_MM
    )
    direction = np.asarray(camera.eye_direction, dtype=np.float64)
    eye = (
        focal[0] + float(direction[0]) * radius,
        focal[1] + float(direction[1]) * radius,
        focal[2] + float(direction[2]) * radius,
    )
    return eye, focal, (0.0, 0.0, 1.0)


class VtkSceneArtists:
    """One shot's scene, built once and redrawn through PyVista per frame.

    Mirrors :class:`~.render3d.ShotSceneArtists`: the same
    :class:`~.shot3d.ShotScene`, the same injected
    :class:`~.render3d.SceneScale`, the same validity stamp -- everything
    that is not *how the surface is drawn* is deliberately identical, so the
    ``renderer:`` line in the stamp is the only thing telling the two apart.

    What only this class has is a :class:`~.bridge.HeadBuild`: the solver's
    watertight triangle mesh, posed per frame by the scene's own recorded
    rotation and position, and rendered as a real lit surface rather than a
    point cloud.

    PyVista is required at construction time (see :func:`require_pyvista`)
    and never at import time, so a caller can hold a reference to this class
    -- for ``isinstance`` checks, type hints, dispatch tables -- without the
    extra installed; only building one raises.
    """

    def __init__(
        self,
        scene: ShotScene,
        build: HeadBuild,
        scale: SceneScale,
        *,
        camera: CameraPreset | str | None = None,
        band: ValidityBand | None = None,
        off_screen: bool = True,
        window_size: tuple[int, int] = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        """Build the scene.

        Args:
            scene: The scene to draw.
            build: The lofted head the scene's centroids came from -- the
                source of the solid mesh this renderer exists to draw.
            scale: The fixed world box and depth ramp.
            camera: Which named view to open on; ``None`` (the default)
                opens the down-the-line view.
            band: The per-sample validity band, when there is one.
            off_screen: Whether the underlying ``Plotter`` renders headless.
                ``False`` is for interactive debugging only; every shipped
                caller leaves this at its default.
            window_size: Pixel size of the render target.

        Raises:
            PyVistaNotAvailableError: If ``pyvista`` is not installed.
            ValueError: If the band does not describe this scene, or
                ``camera`` is a string that names no preset.
        """
        if band is not None and band.n_frames != scene.n_frames:
            raise ValueError(
                "the scene and the validity band must come from one shot; got "
                f"{scene.n_frames} poses against {band.n_frames} verdicts"
            )
        # Deferred like ``pv`` below: see the module docstring for why a
        # sibling import as ordinary-looking as ``.shot3d.viewport_payload``
        # or ``.render.ViewportFallback`` must not happen at module load.
        from .render import ViewportFallback
        from .shot3d import viewport_payload

        pv = require_pyvista()
        self._pv = pv
        self._scene = scene
        self._build = build
        self._scale = scale
        self._band = band
        self._camera = _coerce_camera(camera)
        self._payload = viewport_payload(scene)
        self._fallback = ViewportFallback(
            provider="VTK/PyVista", reason="", renderer=RENDERER
        )
        self._stamp_text = ""
        self._note_text = ""

        mesh = build.loft.mesh
        self._body_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        body_faces = np.asarray(mesh.faces, dtype=np.int64)

        orientation0 = scene.orientation[0]
        position0 = scene.position_m[0]
        self._head_mesh = pv.PolyData(
            _posed_mm(self._body_vertices, orientation0, position0),
            _pv_faces(body_faces),
        )
        self._head_mesh.compute_normals(inplace=True, auto_orient_normals=True)

        self._plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
        self._plotter.set_background("white")
        self._plotter.add_mesh(
            self._head_mesh,
            color=_HEAD_COLOUR,
            smooth_shading=True,
            specular=0.35,
            ambient=0.25,
            name="head",
        )

        self._sole_points = pv.PolyData(scene.sole_world_m(0) * _MM_PER_M)
        self._plotter.add_mesh(
            self._sole_points,
            color=_SOLE_COLOUR,
            point_size=6.0,
            render_points_as_spheres=True,
            name="sole",
        )

        self._build_surface()
        self._floor_line = self._build_floor()

        self.set_camera(self._camera)
        self._draw_stamp(scene.status, extra=f"renderer: {RENDERER}")
        self._refresh_note()

    # ------------------------------------------------------------ accessors

    @property
    def camera(self) -> CameraPreset:
        """Which named view is showing."""
        return self._camera

    @property
    def fallback(self) -> ViewportFallback:
        """What the ADR-0027 viewport layer left this frame drawing with."""
        return self._fallback

    @property
    def payload(self) -> ViewportOverlayPayload:
        """The backend-neutral payload this frame is drawn from."""
        return self._payload

    @property
    def scale(self) -> SceneScale:
        """The fixed world box in force."""
        return self._scale

    @property
    def plotter(self) -> pv.Plotter:
        """The underlying PyVista plotter, for a caller that needs it."""
        return self._plotter

    @property
    def head_mesh(self) -> pv.PolyData:
        """The posed clubhead mesh currently drawn, in world millimetres.

        The solid mesh this renderer exists to draw (see the module
        docstring): :attr:`~pyvista.PolyData.points` is the lofted head's
        watertight triangle mesh posed by the current frame's recorded
        rotation and position, and :attr:`~pyvista.PolyData.faces` is its
        unchanged connectivity.
        """
        return self._head_mesh

    @property
    def stamp_text(self) -> str:
        """The validity stamp currently drawn in-frame.

        Tracked as plain text rather than read back off the PyVista actor:
        :class:`~pyvista.plotting.text.CornerAnnotation` indexes corners by
        an integer VTK enum, not the ``position="upper_left"`` string this
        class draws with, so re-deriving the text from the actor would be
        reaching past PyVista's own public, stable surface. This is the
        honesty stamp -- the one piece of text this module is not allowed to
        drop -- so a caller (or a test) can check it without knowing that.
        """
        return self._stamp_text

    @property
    def note_text(self) -> str:
        """The camera/surface/divot qualifier currently drawn in-frame."""
        return self._note_text

    # ------------------------------------------------------------- building

    def _build_surface(self) -> None:
        """Draw the free surface, once, as a flat translucent plane."""
        scene = self._scene
        surface = scene.surface
        low_x, high_x = surface.along_extent_m
        low_y, high_y = surface.across_extent_m
        center = (
            (low_x + high_x) / 2.0 * _MM_PER_M,
            (low_y + high_y) / 2.0 * _MM_PER_M,
            surface.height_m * _MM_PER_M,
        )
        plane = self._pv.Plane(
            center=center,
            direction=(0.0, 0.0, 1.0),
            i_size=max((high_x - low_x) * _MM_PER_M, 1e-3),
            j_size=max((high_y - low_y) * _MM_PER_M, 1e-3),
        )
        self._plotter.add_mesh(
            plane, color=_SURFACE_COLOUR, opacity=_SURFACE_ALPHA, name="surface"
        )

    def _build_floor(self) -> pv.PolyData:
        """Add the divot profile: a tube coloured by depth on a fixed ramp.

        Coloured, unlike :mod:`~.render3d`'s solid-colour line, because a
        colour-mapped surface is what PyVista is for and a matplotlib
        ``Line3D`` cannot gradient a stroke without real work. The colour
        limits come from :attr:`~.render3d.SceneScale.depth_m`, fixed once
        here and never touched by :meth:`update`, so the ramp cannot drift
        per frame the way #8728 forbids for the sole load field.
        """
        scene = self._scene
        divot = scene.divot
        stations_mm = divot.station_m * _MM_PER_M
        across_mm = float(np.mean(self._scale.y_m)) * _MM_PER_M
        n_stations = stations_mm.size
        points = np.column_stack(
            [
                stations_mm,
                np.full(n_stations, across_mm),
                np.full(n_stations, divot.surface_height_m * _MM_PER_M),
            ]
        )
        floor = self._pv.PolyData(points, lines=_polyline_connectivity(n_stations))
        floor.point_data["depth_mm"] = np.zeros(n_stations, dtype=np.float64)
        depth_low, depth_high = self._scale.depth_m
        self._plotter.add_mesh(
            floor,
            scalars="depth_mm",
            cmap=_FLOOR_COLORMAP,
            clim=(depth_low * _MM_PER_M, depth_high * _MM_PER_M),
            render_lines_as_tubes=True,
            line_width=6.0,
            name="floor",
            show_scalar_bar=True,
            scalar_bar_args={"title": "divot depth [mm]", "fmt": "%.1f"},
        )
        return floor

    def _update_trail(self, points_mm: NDArray[np.float64]) -> None:
        """Rebuild the sole-reference path up to the current frame.

        Rebuilt rather than mutated in place: unlike the head, the sole and
        the divot floor, the trail's own *point count* grows with the frame
        index, so its connectivity cannot be fixed up front. PyVista replaces
        the actor named ``"trail"`` in place, so this does not accumulate
        artists the way a naive per-frame ``add_mesh`` without a stable name
        would.
        """
        points = points_mm
        if points.shape[0] < 2:
            points = np.vstack([points, points])
        trail = self._pv.PolyData(points, lines=_polyline_connectivity(points.shape[0]))
        self._plotter.add_mesh(
            trail,
            color=_PATH_COLOUR,
            render_lines_as_tubes=True,
            line_width=4.0,
            name=_TRAIL_NAME,
        )

    def _draw_stamp(self, status: EnvelopeStatus, *, extra: str) -> None:
        """Draw the validity stamp inside the frame and return nothing.

        In-frame, like :func:`~.render.stamp_axes`, and never optional: the
        text is :func:`~.render.validity_stamp`, the exact sentence
        :mod:`~.render3d` stamps, with the same coloured backing PyVista's
        ``vtkTextProperty`` can carry directly.
        """
        # Deferred: see the module docstring. By the time this method runs,
        # :func:`require_pyvista` has already paid the pyvista half of that
        # price; this pays the ``bunkershot3d``/``mujoco`` half, only here.
        from .render import validity_stamp
        from .report import status_colour

        scene = self._scene
        text = validity_stamp(status, scene.fidelity_tier)
        full = text if not extra else f"{text}\n{extra}"
        actor = self._plotter.add_text(
            full,
            position="upper_left",
            font_size=_STAMP_FONT_PT,
            color="white",
            shadow=True,
            name="stamp",
        )
        text_property = actor.GetTextProperty()
        text_property.SetBackgroundColor(*_hex_to_rgb01(status_colour(status)))
        text_property.SetBackgroundOpacity(0.88)
        self._stamp_text = full

    def _refresh_note(self, *, title: str = "") -> None:
        """Rewrite the qualifier text: the camera, the surface, the divot."""
        scene = self._scene
        surface = scene.surface
        divot = scene.divot
        camera = self._camera
        lines = [
            f"{camera.label} - {camera.description}",
            surface.describe(),
            divot.describe(),
        ]
        if title:
            lines.insert(0, title)
        full = "\n".join(lines)
        self._plotter.add_text(
            full,
            position="lower_left",
            font_size=_NOTE_FONT_PT,
            color="#222222",
            name="note",
        )
        self._note_text = full

    # ------------------------------------------------------------ the frame

    def set_camera(self, camera: CameraPreset | str) -> None:
        """Point the view at one of the named presets.

        Args:
            camera: The preset.

        Raises:
            ValueError: If the name is not one of the three.
        """
        chosen = _coerce_camera(camera)
        self._camera = chosen
        eye, focal, up = _camera_geometry(self._scale, chosen)
        self._plotter.camera_position = (eye, focal, up)
        self._refresh_note()

    def _check_frame(self, frame: int) -> int:
        """Validate a frame index against the scene.

        Args:
            frame: The requested sample index.

        Returns:
            The index.

        Raises:
            ValueError: If it is outside the recorded shot.
        """
        scene = self._scene
        if not 0 <= int(frame) < scene.n_frames:
            raise ValueError(
                f"frame {frame} is outside the recorded shot, which has "
                f"{scene.n_frames} samples"
            )
        return int(frame)

    def update(self, frame: int) -> None:
        """Show one sample.

        Args:
            frame: The sample index.

        Raises:
            ValueError: If the index is outside the recorded shot.
        """
        index = self._check_frame(frame)
        scene = self._scene
        orientation = scene.orientation[index]
        position = scene.position_m[index]

        self._head_mesh.points = _posed_mm(self._body_vertices, orientation, position)
        self._head_mesh.compute_normals(inplace=True, auto_orient_normals=True)

        self._sole_points.points = scene.sole_world_m(index) * _MM_PER_M

        trail_mm = scene.sole_reference_world_m[: index + 1] * _MM_PER_M
        self._update_trail(trail_mm)

        divot = scene.divot
        points = np.asarray(self._floor_line.points, dtype=np.float64)
        points[:, 2] = divot.floor_m[index] * _MM_PER_M
        self._floor_line.points = points
        self._floor_line.point_data["depth_mm"] = divot.depth_m[index] * _MM_PER_M

        moment_ms = float(scene.time_s[index]) * 1e3
        depth_mm = float(scene.sole_depth_m[index]) * 1e3
        area_cm2 = float(divot.section_area_m2[index]) * 1e4
        status = scene.status if self._band is None else self._band.status_at(index)
        self._draw_stamp(status, extra=f"renderer: {RENDERER}")
        self._refresh_note(
            title=(
                f"{moment_ms:.2f} ms - sole {depth_mm:+.2f} mm below the free "
                f"surface - divot section {area_cm2:.2f} cm^2"
            )
        )

    # ------------------------------------------------------------- output

    def screenshot(self, path: str | Path) -> Path:
        """Render the current frame to a PNG file.

        Args:
            path: Where to write the image.

        Returns:
            The path written.
        """
        target = Path(path)
        self._plotter.screenshot(str(target))
        return target

    def image_array(self) -> NDArray[np.uint8]:
        """Render the current frame and return it as an RGB(A) array."""
        return np.asarray(self._plotter.screenshot(return_img=True))

    def close(self) -> None:
        """Release the underlying render window."""
        self._plotter.close()


def draw_scene_frame_vtk(
    scene: ShotScene,
    build: HeadBuild,
    *,
    frame: int = 0,
    scale: SceneScale | None = None,
    camera: CameraPreset | str | None = None,
    band: ValidityBand | None = None,
    off_screen: bool = True,
    window_size: tuple[int, int] = _DEFAULT_WINDOW_SIZE,
) -> VtkSceneArtists:
    """Draw one sample of one scene through PyVista.

    Mirrors :func:`~.render3d.draw_scene_frame`. Hold the returned
    :class:`VtkSceneArtists` and call :meth:`~VtkSceneArtists.update` for an
    animation rather than rebuilding per frame -- rebuilding costs a mesh
    upload and a normals pass every time.

    Args:
        scene: The scene.
        build: The lofted head backing the scene's centroids.
        frame: Which sample to show.
        scale: The fixed world box, from :func:`~.render3d.scene_scale`.
            Defaults to this scene's own, which is correct for a single
            design and wrong for a comparison -- pass the merged scale there.
        camera: Which named view to open on; ``None`` (the default) opens
            the down-the-line view.
        band: The per-sample validity band, when there is one.
        off_screen: Whether the plotter renders headless.
        window_size: Pixel size of the render target.

    Returns:
        The built artists, ready to be updated to another frame or rendered.

    Raises:
        PyVistaNotAvailableError: If ``pyvista`` is not installed.
        ValueError: If the frame is outside the shot, the band does not
            describe the same shot as the scene, or ``camera`` is a string
            that names no preset.
    """
    if scale is None:
        # Deferred: see the module docstring -- ``.render3d`` is another of
        # the siblings that reaches ``bunkershot3d``/``mujoco``.
        from .render3d import scene_scale

        limits = scene_scale((scene,))
    else:
        limits = scale
    artists = VtkSceneArtists(
        scene,
        build,
        limits,
        camera=camera,
        band=band,
        off_screen=off_screen,
        window_size=window_size,
    )
    artists.update(frame)
    return artists


def shot_scene_still_vtk(
    scene: ShotScene,
    build: HeadBuild,
    *,
    frame: int | None = None,
    scale: SceneScale | None = None,
    camera: CameraPreset | str | None = None,
    band: ValidityBand | None = None,
    window_size: tuple[int, int] = _DEFAULT_WINDOW_SIZE,
) -> VtkSceneArtists:
    """Render one frame headless -- the ADR-0027 VTK/PyVista adapter.

    Mirrors :func:`~.render3d.shot_scene_still`.

    Args:
        scene: The scene.
        build: The lofted head backing the scene's centroids.
        frame: Which sample to show; defaults to the deepest moment.
        scale: The fixed world box; see :func:`draw_scene_frame_vtk`.
        camera: Which named view; ``None`` (the default) opens the
            down-the-line view.
        band: The per-sample validity band, when there is one.
        window_size: Pixel size of the render target.

    Returns:
        The built artists; call :meth:`~VtkSceneArtists.screenshot` or
        :meth:`~VtkSceneArtists.image_array` for pixels.

    Raises:
        PyVistaNotAvailableError: If ``pyvista`` is not installed.
        ValueError: If the frame is outside the shot, or ``camera`` is a
            string that names no preset.
    """
    chosen = int(np.argmax(scene.sole_depth_m)) if frame is None else frame
    return draw_scene_frame_vtk(
        scene,
        build,
        frame=chosen,
        scale=scale,
        camera=camera,
        band=band,
        off_screen=True,
        window_size=window_size,
    )
