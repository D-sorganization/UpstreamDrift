import re

file_path = "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/motion_capture_plotter_visualization.py"

with open(file_path, "r") as f:
    content = f.read()

# Replace block 1 (mid-hands path)
search1 = """            # ⚡ Bolt: Vectorized array construction is much faster than row-by-row iterrows()"""
replace1 = """            # ⚡ Bolt: Vectorized stack is much faster than iterrows()"""
content = content.replace(search1, replace1)

search2 = """            # Column check happens once outside the loop since dataframes have uniform columns"""
replace2 = """            # Column check happens once outside the loop for uniform columns"""
content = content.replace(search2, replace2)

with open(file_path, "w") as f:
    f.write(content)

print("Patched.")
