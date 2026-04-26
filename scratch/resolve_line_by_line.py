import os


def resolve_file(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    new_lines = []
    in_head = False
    in_origin = False

    for line in lines:
        if line.startswith(("<<<<<<<", "<<<<<<<<")):
            in_head = True
            in_origin = False
            continue
        if line.startswith("======="):
            in_head = False
            in_origin = True
            continue
        if line.startswith(">>>>>>>"):
            in_head = False
            in_origin = False
            continue

        if in_origin or not in_head:
            new_lines.append(line)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# Files to resolve
files = [
    "src/shared/python/calc_backend/routers/rotation_converter.py",
    "src/shared/python/upstream_drift_tools/process_calculators/acid_gas_dewpoint_calculator.py",
    "src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/engine/pressure_drop_calculation_engine.py",
    "src/shared/python/upstream_drift_tools/process_calculators/pressure_drop_calculator/utils/gas_properties.py",
    "src/shared/python/upstream_drift_tools/process_calculators/syngas_compression_calculator.py",
    "scripts/db_migrate.py",
    ".gitignore",
    "CLAUDE.md",
    "Dockerfile",
    "install.sh",
]

for f in files:
    if os.path.exists(f):
        resolve_file(f)
