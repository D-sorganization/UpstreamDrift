import os


def fix_conflicts_line_by_line(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    in_conflict = False
    side = None  # 1 for HEAD, 2 for other
    part1 = []
    part2 = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<<<<<<< HEAD"):
            in_conflict = True
            side = 1
            part1 = []
            part2 = []
        elif stripped == "=======":
            side = 2
        elif stripped.startswith(">>>>>>> "):
            in_conflict = False
            # Resolve
            p1 = "".join(part1)
            p2 = "".join(part2)
            if not p1.strip():
                new_lines.extend(part2)
            elif not p2.strip():
                new_lines.extend(part1)
            elif "ONLINE" in p1 and "ONLINE" in p2:
                if "! [[" in p2:
                    new_lines.extend(part2)
                else:
                    new_lines.extend(part1)
            else:
                new_lines.extend(part1)  # Fallback to HEAD
        else:
            if in_conflict:
                if side == 1:
                    part1.append(line)
                else:
                    part2.append(line)
            else:
                new_lines.append(line)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(new_lines)


workflow_dir = ".github/workflows"
for filename in os.listdir(workflow_dir):
    if filename.endswith(".yml"):
        fix_conflicts_line_by_line(os.path.join(workflow_dir, filename))
