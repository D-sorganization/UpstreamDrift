"""Preserve exact run18 parity inputs, failed receipts and runtime identities."""

from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

base = Path("/mnt/c/Users/diete")
output = base / "drake-run18-evidence-10022-02.zip"
if output.exists():
    raise FileExistsError(output)
files = []
for name in (
    "drake-run18-pin-reference-10022-01",
    "drake-run18-pin-reference-10022-02",
    "drake-run18-parity-10022-01",
    "drake-run18-parity-10022-02",
):
    files.extend(
        (path, str(path.relative_to(base)))
        for path in (base / name).rglob("*")
        if path.is_file()
    )
for name in (
    "drake-run18-diagnostic-10022-01.json",
    "replay_drake_candidate_10022_18.py",
    "replay_drake_candidate_10022_18_02.py",
    "diagnose_drake_run18_10022.py",
    "native_geometry_spec_9967.json",
    "native-golf-9967-01.urdf",
    "native-golf-9967-01.sidecar.json",
    "driver_marker_payload_9967.json",
    "native-ms-fit-9967-18/returned-candidate.json",
):
    files.append((base / name, "inputs/" + name))
runtime = Path("/home/dieterolson/drake-native-runtime-10022-02")
files.extend(
    (path, "runtime/" + str(path.relative_to(runtime)))
    for path in runtime.rglob("*.py")
)
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
    for path, name in files:
        archive.write(path, name)
    archive.writestr("sha256.json", json.dumps(manifest, indent=2))
    for name, python in (
        ("drake", "/home/dieterolson/drake-native-10022/bin/python"),
        ("pinocchio", "/home/dieterolson/simscape-pinocchio-9967/.venv/bin/python"),
    ):
        archive.writestr(
            name + "-versions.json",
            subprocess.check_output(
                [python, str(base / "drake_runtime_versions_10022.py")]
            ),
        )
