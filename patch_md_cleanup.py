import re
filepath = "src/engines/physics_engines/mujoco/INTERACTIVE_VISUALIZATION_GUIDE.md"
with open(filepath, 'r') as f:
    content = f.read()

# Fix the redundant imports in the loops
content = content.replace('        import math\n        force = np.zeros(6)', '        force = np.zeros(6)')

# Ensure import math is at the top of the functions instead
content = content.replace('def analyze_contacts(model, data):\n    """Analyze all active contacts."""\n    import math', 'def analyze_contacts(model, data):\n    """Analyze all active contacts."""\n    import math')

content = content.replace('def compute_stability_metrics(model, data):\n    """Calculate ZMP and stability margins."""\n    contact_forces = []', 'def compute_stability_metrics(model, data):\n    """Calculate ZMP and stability margins."""\n    import math\n    contact_forces = []')

with open(filepath, 'w') as f:
    f.write(content)
