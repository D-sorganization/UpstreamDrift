"""Safe teardown of ``mujoco.Renderer`` instances (UD #12884).

``mujoco.Renderer.close()`` (mujoco 3.x, classic GLFW/WGL backend) destroys its
own GL window first and only then frees its ``MjrContext``. ``mjr_freeContext``
issues ``glDelete*`` calls against whatever GL context is *current* at that
moment -- typically the renderer that was created or rendered most recently.
GL object names are per-context counters, so they collide: freeing renderer A
silently deletes renderer B's offscreen framebuffer and textures, and every
later ``B.render()`` reads back pure black.

That is exactly what happened in the launcher: the MuJoCo dashboard loads a
model twice, the second load replaced (and garbage-collected) the first
renderer while the new one was current, and the viewport went black.

:func:`release_renderer` frees the ``MjrContext`` while the renderer's *own* GL
context is current, then closes the renderer (which is idempotent for an
already-freed ``MjrContext``).
"""

from __future__ import annotations

from typing import Any

__all__ = ["release_renderer"]


def release_renderer(renderer: Any) -> None:
    """Free ``renderer``'s GL resources without touching any other context.

    ``None`` is accepted and ignored so callers can release unconditionally.

    Raises:
        TypeError: if ``renderer`` is not ``None`` and has no ``close()``.

    Postcondition: the renderer is closed; GL objects owned by other live
    renderers are untouched.
    """
    if renderer is None:
        return
    if not callable(getattr(renderer, "close", None)):
        raise TypeError(f"expected a mujoco.Renderer, got {type(renderer).__name__}")
    # Private attributes of mujoco's classic Renderer; absent on other backends
    # or after close(), in which case close() alone is already safe.
    gl_context = getattr(renderer, "_gl_context", None)
    mjr_context = getattr(renderer, "_mjr_context", None)
    if gl_context is not None and mjr_context is not None:
        gl_context.make_current()
        mjr_context.free()
    renderer.close()
