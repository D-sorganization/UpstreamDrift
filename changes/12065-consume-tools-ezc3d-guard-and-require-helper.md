---
issue: 12065
summary: "Consume the Tools-owned ezc3d native-DLL guard (vendor pin bump to Tools #5581); data_io guard becomes a re-export, remaining importorskip(ezc3d) sites use require_ezc3d, AST guard forbids importorskip (step 2 of 12065)"
branch: "fix/12065-s2-ezc3d-guard"
---

Refs #12065. Bumps vendor/ud-tools to bdc3eed0e934 (Tools #5581), syncs the sidekick child copies (utils/native_dll_guard.py, lab/bio/c3d_reader.py), reduces data_io/native_dll_guard.py to a re-export and migrates 15 test sites from pytest.importorskip("ezc3d") to tests.helpers.ezc3d_support.require_ezc3d. Removes TOOLS_CHILD_COPY_PENDING.
