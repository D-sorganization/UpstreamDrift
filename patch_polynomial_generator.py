import re

def modify_file(path):
    with open(path, "r") as f:
        content = f.read()

    # The original instructions were:
    # Replace np.array([p[0] for p in points]) / np.array([p[1] for p in points]) with np.asarray(points) + column slicing (pts[:, 0] / pts[:, 1]).
    # We need to find this pattern in the files.

    # Check if the pattern exists:
    if "np.array([p[0] for p in points])" in content:
        print(f"Found in {path}")

modify_file("src/shared/python/signal_toolkit/polynomial_generator.py")
modify_file("src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/polynomial_generator.py")
