# Eagerly import engine subpackages so they are accessible as attributes of this
# package.  This is required for Python 3.10 compatibility: unittest.mock.patch
# navigates dotted attribute chains (e.g. "src.engines.physics_engines.mujoco...")
# and raises AttributeError when intermediate packages are not set as attributes
# on their parent module.  Each subpackage's own __init__.py is empty, so this
# import is a no-op beyond registering the attribute — no heavy C-library deps
# are loaded here.
from . import drake, mujoco, myosuite, opensim, pinocchio  # noqa: F401
