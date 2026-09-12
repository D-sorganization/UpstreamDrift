"""Archive the bounded reconstructed-seed angular-velocity diagnostic."""

from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

base = Path("/mnt/c/Users/diete")
output = base / "drake-restart-angular-evidence-10022-01.zip"
if output.exists():
    raise FileExistsError(output)
files = []
for directory in (
    "drake-restart-source-10022-01",
    "drake-restart-reconstructed-10022-01",
    "drake-restart-angular-10022-01",
):
    files.extend(
        (p, str(p.relative_to(base)))
        for p in (base / directory).rglob("*")
        if p.is_file()
    )
for name in (
    "diagnose_restart_angular_10022.py",
    "replay_drake_candidate_10022_18_02.py",
    "native-ms-fit-9967-18/returned-candidate.json",
    "native-ms-recenter-audit-9967-19/initial-candidate.json",
    "native_geometry_spec_9967.json",
    "native-golf-9967-01.urdf",
    "native-golf-9967-01.sidecar.json",
    "driver_marker_payload_9967.json",
):
    files.append((base / name, "inputs/" + name))
for name, path in (
    ("pinocchio19", "/home/dieterolson/native-ms-pilot-9967-19"),
    ("drake02", "/home/dieterolson/drake-native-runtime-10022-02"),
):
    root = Path(path)
    files.extend((p, name + "/" + str(p.relative_to(root))) for p in root.rglob("*.py"))
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as z:
    for path, name in files:
        z.write(path, name)
    z.writestr("sha256.json", json.dumps(manifest, indent=2))
    for name, python in (
        ("drake", "/home/dieterolson/drake-native-10022/bin/python"),
        ("pinocchio", "/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python"),
    ):
        z.writestr(
            name + "-versions.json",
            subprocess.check_output(
                [python, str(base / "drake_runtime_versions_10022.py")]
            ),
        )
