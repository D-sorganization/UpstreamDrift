with open("src/bunkershot3d/metrics/trace.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "import math" in line and "    import math" in line:
        continue # skip the indented one
    new_lines.append(line)

with open("src/bunkershot3d/metrics/trace.py", "w") as f:
    f.writelines(new_lines)
