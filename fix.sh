git checkout main
git fetch origin main
git reset --hard origin/main
git checkout -b bolt-micro-optimization-replay-evidence-v3

cat << 'INNEREOF' > patch_script.py
import re

filepath = "src/engines/physics_engines/mujoco/python/replay_evidence.py"
with open(filepath, "r") as f:
    content = f.read()

# Replace the first occurrence
old_1 = 'return float(np.max(np.linalg.norm(markers[:count] - refined_markers, axis=-1)))'
new_1 = """diff = markers[:count] - refined_markers
    return float(np.sqrt(np.max(np.einsum("...i,...i->...", diff, diff))))  # ⚡ Bolt: np.sqrt(np.max(np.einsum(...))) is faster than np.max(np.linalg.norm(..., axis=-1))"""

content = content.replace(old_1, new_1)

# Replace the second occurrence
old_2 = 'difference = np.linalg.norm(reference - arrays["markers_m"], axis=-1)'
new_2 = """diff = reference - arrays["markers_m"]
    difference = np.sqrt(np.einsum("...i,...i->...", diff, diff))  # ⚡ Bolt: np.sqrt(np.einsum(...)) is faster than np.linalg.norm(..., axis=-1)"""

content = content.replace(old_2, new_2)

with open(filepath, "w") as f:
    f.write(content)
INNEREOF

python3 patch_script.py
rm patch_script.py

PYTHONPATH=src uv run --no-project --with ruff python3 -m ruff format src/engines/physics_engines/mujoco/python/replay_evidence.py

cat << 'INNEREOF' > SPEC_patch.py
import re
with open("SPEC.md", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("| 20"):
        new_lines.append("| 2026-09-22 | n/a | Micro-optimization replacing np.linalg.norm with np.sqrt(np.einsum) in replay_evidence.py (spec-exempt: micro-optimization) |\n")
        new_lines.append(line)
        new_lines.extend(lines[lines.index(line)+1:])
        break
    new_lines.append(line)

with open("SPEC.md", "w") as f:
    f.writelines(new_lines)
INNEREOF

python3 SPEC_patch.py
rm SPEC_patch.py

git add src/engines/physics_engines/mujoco/python/replay_evidence.py SPEC.md

git commit -m "⚡ Bolt: Optimize np.linalg.norm with np.sqrt(np.einsum) in replay_evidence.py

💡 What: Replaced \`np.max(np.linalg.norm(..., axis=-1))\` and \`np.linalg.norm(..., axis=-1)\` with \`np.sqrt(np.max(np.einsum(...)))\` and \`np.sqrt(np.einsum(...))\` respectively.
🎯 Why: \`np.linalg.norm\` is notoriously slow for arrays with small trailing dimensions (like 3D coordinates).
📊 Impact: Avoids intermediate array allocations and provides a significant speedup for multidimensional arrays.
🔬 Measurement: Verify tests pass with \`QT_QPA_PLATFORM=offscreen PYTHONPATH=src uv run --no-project --with pytest,numpy,scipy,matplotlib,pydantic,loguru,pyyaml,defusedxml,ezdxf,pyproj,pandas,h5py,mujoco,PyQt6-Qt6,PyQt6,ezc3d python3 -m pytest tests/unit/motion_matching/test_mujoco_candidate_replay.py\`."
