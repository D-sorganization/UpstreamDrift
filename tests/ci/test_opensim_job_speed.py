"""The real-OpenSim job reuses per-runner state instead of rebuilding it (#13510).

Before #13510 every ``Real OpenSim 4.6`` run on the saturated docker fleet:

* created a fresh venv in ``$RUNNER_TEMP`` and wrote the whole dev lock into it
  (~210 s of disk I/O, mean venv step 274 s);
* passed ``--no-cache-dir``, defeating the per-runner ``PIP_CACHE_DIR`` from
  Runner_Dashboard ``configure-runner-pipcache.sh``;
* let ``actions/setup-python`` ``cache: pip`` miss and then upload ~135 MB
  to the GitHub cache over <host>'s cellular uplink (mean 128 s);
* re-cloned opensim-models from GitHub.

These tests pin the replacement: a persistent, validated, per-runner venv
keyed by the lock, the opensim pin and the interpreter, with the editable
install redone every run. The GitHub pip cache is used on hosted runners only.
The opensim-models clone is cached per commit. The test step itself (paths,
markers, timeouts, zero-passed guard) must not change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-requires-opensim.yml"
JOB = "requires-opensim"
OSIM = "Models/Rajagopal/RajagopalLaiUhlrich2023.osim"


def _job() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data["jobs"][JOB]


def _step(name: str) -> dict[str, Any]:
    for step in _job()["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"step {name!r} missing from {WORKFLOW.name}")


def _uses_step(prefix: str) -> dict[str, Any]:
    for step in _job()["steps"]:
        if str(step.get("uses", "")).startswith(prefix):
            return step
    raise AssertionError(f"no step uses {prefix}")


VENV_STEP = "Restore or build persistent OpenSim venv"
EDITABLE_STEP = "Install UpstreamDrift (editable) into the OpenSim venv"
MODELS_STEP = "Fetch OpenSim Rajagopal body model"
TEST_STEP = "Run requires_opensim tests"


class TestGitHubPipCacheOnlyOnHostedRunners:
    def test_setup_python_cache_is_hosted_only(self) -> None:
        # Self-hosted runners keep a persistent per-runner PIP_CACHE_DIR; the
        # GitHub cache only missed and then uploaded ~135 MB per run.
        cache = _uses_step("actions/setup-python")["with"]["cache"]
        assert cache == "${{ runner.environment == 'github-hosted' && 'pip' || '' }}"

    def test_job_env_has_no_runner_context(self) -> None:
        # ``${{ runner.* }}`` in job-level env makes GitHub reject the workflow.
        for value in _job().get("env", {}).values():
            assert "runner." not in str(value)


class TestPersistentVenv:
    def test_old_temp_venv_step_is_gone(self) -> None:
        names = [step.get("name") for step in _job()["steps"]]
        assert "Set up Python venv and install dependencies" not in names
        assert "opensim-ci-venv" not in WORKFLOW.read_text(encoding="utf-8")

    def test_venv_lives_in_per_runner_tool_cache(self) -> None:
        run = _step(VENV_STEP)["run"]
        assert '"${RUNNER_TOOL_CACHE:-$RUNNER_TEMP}/upstreamdrift-opensim-venv' in run
        # Per runner name, so a host-shared tool cache can never race.
        assert "${RUNNER_NAME:-local}" in run

    def test_key_covers_lock_opensim_pin_and_interpreter(self) -> None:
        step = _step(VENV_STEP)
        assert step["env"]["OPENSIM_PIN"] == "4.6"
        run = step["run"]
        assert "sha256sum requirements-dev.lock" in run
        assert "opensim=$OPENSIM_PIN" in run
        assert "sys.version, sys.executable" in run
        assert "schema=$OPENSIM_VENV_SCHEMA" in run

    def test_reuse_is_validated_and_falls_back_to_rebuild(self) -> None:
        run = _step(VENV_STEP)["run"]
        # A venv is reused only when its completion marker carries the key,
        # the OpenSim/pytest stack imports and the installed set is unchanged.
        assert ".opensim-venv-complete" in run
        assert "opensim.GetVersion().startswith" in run
        assert "pip freeze --exclude-editable" in run
        assert ".opensim-venv-freeze" in run
        assert "if venv_ok; then" in run
        # Anything else (corrupt, partial, stale) is wiped and rebuilt.
        assert 'rm -rf "$venv"' in run

    def test_marker_is_written_last(self) -> None:
        run = _step(VENV_STEP)["run"]
        marker = run.index('> "$venv/.opensim-venv-complete"')
        for earlier in (
            "-r requirements-dev.lock",
            '"opensim==$OPENSIM_PIN"',
            '> "$venv/.opensim-venv-freeze"',
        ):
            assert run.index(earlier) < marker

    def test_other_keys_are_pruned(self) -> None:
        run = _step(VENV_STEP)["run"]
        assert 'find "$base" -mindepth 1 -maxdepth 1 ! -name "$key"' in run

    def test_lock_install_keeps_ignore_installed(self) -> None:
        run = _step(VENV_STEP)["run"]
        assert (
            '-m pip install "${no_cache[@]}" --ignore-installed -r requirements-dev.lock'
            in run
        )

    def test_no_cache_dir_only_without_per_runner_cache(self) -> None:
        # --no-cache-dir guarded the shared ~/.cache/pip http-v2 race. A
        # per-runner PIP_CACHE_DIR removes that race, so only keep the guard
        # when no per-runner cache is configured.
        for name in (VENV_STEP, EDITABLE_STEP):
            run = _step(name)["run"]
            assert (
                'if [ -z "${PIP_CACHE_DIR:-}" ]; then no_cache=(--no-cache-dir); fi'
                in run
            )
            for line in run.splitlines():
                if "pip install" in line:
                    assert "--no-cache-dir" not in line, line

    def test_venv_is_exported_for_later_steps(self) -> None:
        run = _step(VENV_STEP)["run"]
        assert 'echo "$venv/bin" >> "$GITHUB_PATH"' in run
        assert 'echo "VIRTUAL_ENV=$venv" >> "$GITHUB_ENV"' in run

    def test_editable_install_runs_every_time(self) -> None:
        names = [step.get("name") for step in _job()["steps"]]
        assert names.index(VENV_STEP) < names.index(EDITABLE_STEP)
        assert names.index(EDITABLE_STEP) < names.index(
            "Verify OpenSim 4.6 is importable"
        )
        step = _step(EDITABLE_STEP)
        assert step.get("if") == "needs.classify-changes.outputs.run == 'true'"
        run = step["run"]
        assert (
            'python -m pip install "${no_cache[@]}" --force-reinstall --no-deps -e ".[dev]"'
            in run
        )
        assert "--no-deps" in run


class TestCachedModelsClone:
    def test_clone_is_cached_per_upstream_commit(self) -> None:
        run = _step(MODELS_STEP)["run"]
        assert 'git ls-remote "$MODELS_URL" HEAD' in run
        assert '"${RUNNER_TOOL_CACHE:-$RUNNER_TEMP}/upstreamdrift-opensim-models' in run
        assert "${RUNNER_NAME:-local}" in run

    def test_cached_copy_is_verified_before_reuse(self) -> None:
        run = _step(MODELS_STEP)["run"]
        assert 'rev-parse HEAD 2>/dev/null)" = "$sha"' in run
        assert "diff --quiet HEAD -- Models/Rajagopal" in run

    def test_fresh_clone_fallback_and_final_check_remain(self) -> None:
        run = _step(MODELS_STEP)["run"]
        assert (
            'git clone --depth 1 --filter=blob:none --sparse "$MODELS_URL" "$dest"'
            in run
        )
        assert "sparse-checkout set Models/Rajagopal" in run
        assert f'test -f "$dest/{OSIM}"' in run


class TestTestStepUnchanged:
    def test_paths_markers_and_timeouts(self) -> None:
        run = _step(TEST_STEP)["run"]
        assert "tests/unit/lifting tests/opensim" in run
        assert '-m "requires_opensim and not slow and not heavy"' in run
        assert "--timeout=300" in run
        assert "--timeout-method=thread" in run
        assert _job()["timeout-minutes"] == 25

    def test_zero_passed_guard_remains(self) -> None:
        run = _step(TEST_STEP)["run"]
        assert "if passed == 0:" in run
        assert "sys.exit(1)" in run

    def test_opensim_version_is_still_asserted(self) -> None:
        run = _step("Verify OpenSim 4.6 is importable")["run"]
        assert 'version.startswith("4.6")' in run
