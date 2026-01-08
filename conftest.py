import mujoco  # noqa: F401

# Import MuJoCo early to avoid Windows DLL initialization conflicts (Access Violation)
# that occur when MuJoCo is loaded during pytest collection with certain plugins.
