def modify_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    new_content = content.replace(
        "float(100.0 * np.linalg.norm(self.gravity) / total_magnitude)",
        "float(100.0 * math.sqrt(np.vdot(self.gravity, self.gravity)) / total_magnitude)"
    ).replace(
        "float(100.0 * np.linalg.norm(self.coriolis) / total_magnitude)",
        "float(100.0 * math.sqrt(np.vdot(self.coriolis, self.coriolis)) / total_magnitude)"
    ).replace(
        "float(\n                100.0 * np.linalg.norm(self.applied_torque) / total_magnitude\n            )",
        "float(100.0 * math.sqrt(np.vdot(self.applied_torque, self.applied_torque)) / total_magnitude)"
    ).replace(
        "float(\n                100.0 * np.linalg.norm(self.constraint) / total_magnitude\n            )",
        "float(100.0 * math.sqrt(np.vdot(self.constraint, self.constraint)) / total_magnitude)"
    ).replace(
        "float(100.0 * np.linalg.norm(self.external) / total_magnitude)",
        "float(100.0 * math.sqrt(np.vdot(self.external, self.external)) / total_magnitude)"
    )

    if 'import math' not in new_content:
        new_content = new_content.replace('import numpy as np', 'import numpy as np\nimport math')

    with open(filepath, 'w') as f:
        f.write(new_content)

modify_file('src/shared/python/spatial_algebra/indexed_acceleration.py')
