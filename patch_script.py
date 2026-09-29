import re

file_path = "src/shared/python/physics/_pre_impact_contracts.py"
with open(file_path, "r") as f:
    content = f.read()

content = re.sub(r'import enum', 'import enum\nimport math', content)
content = content.replace(
    'norm = float(np.linalg.norm(quaternion))',
    'norm = math.sqrt(np.vdot(quaternion, quaternion))'
)
content = content.replace(
    'if abs(float(np.linalg.norm(vector)) - 1.0) > UNIT_VECTOR_TOLERANCE:',
    'if abs(math.sqrt(np.vdot(vector, vector)) - 1.0) > UNIT_VECTOR_TOLERANCE:'
)

with open(file_path, "w") as f:
    f.write(content)
