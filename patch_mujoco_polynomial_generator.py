import re

with open("src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/polynomial_generator.py", "r") as f:
    content = f.read()

content = content.replace(
    "xs, ys = zip(*self.current_points, strict=True)",
    "pts = np.asarray(self.current_points)\n            xs, ys = pts[:, 0], pts[:, 1]"
)

with open("src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/polynomial_generator.py", "w") as f:
    f.write(content)
