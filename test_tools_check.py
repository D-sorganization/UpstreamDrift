from tests.unit.repo_hygiene.test_tools_child_copy_contract import _changed_shared_python_paths, _merge_base_with_main, _direct_tools_edit_offenders, _require_tools_shared_paths, _tools_shared_paths
paths = _require_tools_shared_paths(_tools_shared_paths())
offenders = _direct_tools_edit_offenders(_merge_base_with_main(), paths)
print("Offenders:", offenders)
