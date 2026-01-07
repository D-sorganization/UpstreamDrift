"""Script to add ZTCF/ZVCF stub implementations to all physics engines.

This fixes the mypy errors caused by adding abstract methods to PhysicsEngine protocol.
"""

import re
from pathlib import Path

# Stub implementation to add
STUB_METHODS = '''
    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-Torque Counterfactual (ZTCF) - Guideline G1.
        
        TODO: Implement ZTCF for this engine.
        For now, returns NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement ZTCF. "
            f"See pendulum_physics_engine.py for reference implementation."
        )

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.
        
        TODO: Implement ZVCF for this engine.
        For now, returns NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not yet implement ZVCF. "
            f"See pendulum_physics_engine.py for reference implementation."
        )
'''

# Engines to update
engines = [
    "engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py",
    "engines/physics_engines/drake/python/drake_physics_engine.py",
    "engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py",
    "engines/physics_engines/opensim/python/opensim_physics_engine.py",
    "engines/physics_engines/myosuite/python/myosuite_physics_engine.py",
]

def add_stub_methods(filepath: Path) -> bool:
    """Add stub ZTCF/ZVCF methods to engine file."""
    if not filepath.exists():
        print(f"⚠️  Skipping {filepath} (not found)")
        return False
    
    content = filepath.read_text(encoding="utf-8")
    
    # Check if already has the methods
    if "def compute_ztcf" in content:
        print(f"✓ {filepath.name} already has ZTCF/ZVCF")
        return False
    
    # Find the last method in the class (look for the last indented def)
    # We'll insert before the final line of the class
    lines = content.split('\n')
    
    # Find class definition
    class_line_idx = None
    for i, line in enumerate(lines):
        if 'class ' in line and 'PhysicsEngine' in line:
            class_line_idx = i
            break
    
    if class_line_idx is None:
        print(f"⚠️  Could not find PhysicsEngine class in {filepath.name}")
        return False
    
    # Find last method (last line starting with "    def ")
    last_method_end = None
    in_class = False
    indent_level = 0
    
    for i in range(class_line_idx, len(lines)):
        line = lines[i]
        
        if i == class_line_idx:
            in_class = True
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # Check if we've left the class (dedent)
        if in_class and line and not line[0].isspace() and line.strip():
            last_method_end = i
            break
        
        # Track last method
        if in_class and line.strip().startswith('def '):
            # Find the end of this method
            method_indent = len(line) - len(line.lstrip())
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                if next_line.strip() and not next_line[0].isspace():
                    last_method_end = j
                    break
                if next_line.strip().startswith('def ') and (len(next_line) - len(next_line.lstrip())) == method_indent:
                    last_method_end = j
                    break
    
    if last_method_end is None:
        last_method_end = len(lines)
    
    # Insert stub methods before the end
    new_lines = lines[:last_method_end] + STUB_METHODS.split('\n') + lines[last_method_end:]
    
    filepath.write_text('\n'.join(new_lines), encoding="utf-8")
    print(f"✅ Added ZTCF/ZVCF stubs to {filepath.name}")
    return True

if __name__ == "__main__":
    repo_root = Path("C:/Users/diete/Repositories/Golf_Modeling_Suite")
    
    print("Adding ZTCF/ZVCF stub implementations...")
    print("=" * 60)
    
    modified_count = 0
    for engine_path in engines:
        full_path = repo_root / engine_path
        if add_stub_methods(full_path):
            modified_count += 1
    
    print("=" * 60)
    print(f"✓ Modified {modified_count} engine(s)")
    print("\nNote: These are stubs that raise NotImplementedError.")
    print("Full implementation should be done per-engine as needed.")
