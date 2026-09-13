from pathlib import Path
import shutil

root = Path("/home/dieterolson/drake-native-runtime-10022-02")
assert not root.exists()
shutil.copytree("/home/dieterolson/drake-native-runtime-10022-01", root)
shutil.copy2(
    "/mnt/c/Users/diete/drake_native_model_10022_02.py",
    root / "src/engines/physics_engines/drake/python/native_model.py",
)
shutil.copy2(
    "/mnt/c/Users/diete/native_urdf_contract_10022.py",
    root / "src/shared/python/motion_matching/native_urdf_contract.py",
)
