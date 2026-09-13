"""Archive bounded reaction-eliminated polynomial identification evidence."""

from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

base = Path("/mnt/c/Users/diete")
output = base / "reaction-identification-evidence-10022-01.zip"
if output.exists():
    raise FileExistsError(output)
files = []
for directory in (
    "native-reaction-reference-10022-01",
    "native-reaction-identification-10022-01",
    "native-reaction-replay-10022-01",
    "drake-native-reference-10022-01",
):
    files.extend(
        (p, str(p.relative_to(base)))
        for p in (base / directory).rglob("*")
        if p.is_file()
    )
for name in (
    "reaction_identification.py",
    "study_reaction_identification_10022.py",
    "replay_identified_baseline_10022.py",
    "compare_identified_profile_10022.py",
    "native_geometry_spec_9967.json",
    "native-golf-9967-01.urdf",
    "native-golf-9967-01.sidecar.json",
    "native-root-force-9967-02/returned-candidate.json",
):
    files.append((base / name, "inputs/" + name))
root = Path("/home/dieterolson/drake-native-runtime-10022-02")
files.extend((p, "runtime/" + str(p.relative_to(root))) for p in root.rglob("*.py"))
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
