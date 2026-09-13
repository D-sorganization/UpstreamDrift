from pathlib import Path
import hashlib
import json
import subprocess
import zipfile

base = Path("/mnt/c/Users/diete")
output = base / "drake-native-evidence-10022.zip"
assert not output.exists()
files = []
for directory in (
    "drake-native-reference-10022-01",
    "drake-native-check-10022-01",
    "drake-native-check-10022-02",
    "drake-native-check-10022-03",
    "drake-native-r2025b-10022-01",
):
    files.extend(
        (p, str(p.relative_to(base)))
        for p in (base / directory).rglob("*")
        if p.is_file()
    )
runtime = Path("/home/dieterolson/drake-native-runtime-10022-02")
files.extend(
    (p, "runtime/" + str(p.relative_to(runtime))) for p in runtime.rglob("*.py")
)
for name in (
    "qualify_drake_native_10022.py",
    "qualify_drake_native_10022_03.py",
    "qualify_drake_native_10022_04.py",
    "qualify_drake_r2025b_10022.py",
    "test_drake_native_live_10022.py",
    "native_geometry_spec_9967.json",
    "native-golf-9967-01.urdf",
    "native-golf-9967-01.sidecar.json",
    "native-root-force-9967-02/returned-candidate.json",
):
    files.append((base / name, "inputs/" + name))
manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
    for path, name in files:
        archive.write(path, name)
    archive.writestr("sha256.json", json.dumps(manifest, indent=2))
    archive.writestr(
        "requirements-freeze.txt",
        subprocess.check_output(
            ["/home/dieterolson/drake-native-10022/bin/python", "-m", "pip", "freeze"]
        ),
    )
