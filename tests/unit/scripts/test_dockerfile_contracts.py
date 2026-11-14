from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci import check_dockerfile_contracts as guard

pytestmark = pytest.mark.unit

_EXPLORER_DOCKERFILE_OK = """
FROM python:3.12-slim AS runtime
LABEL keep="true"
FROM runtime AS explorer-base
ENV PYTHONDONTWRITEBYTECODE=1
FROM explorer-base AS explorer-headless
ENV QT_QPA_PLATFORM=offscreen \\
    MUJOCO_GL=egl
FROM explorer-base AS explorer-gui
ENV QT_QPA_PLATFORM=xcb \\
    MUJOCO_GL=egl \\
    DISPLAY=:99
"""

_COMPOSE_OK = """
x-explorer-build: &explorer-build
  labels:
    keep: "true"
  args:
    TOOLS_GITLINK_SHA: ${TOOLS_GITLINK_SHA:-}
    TOOLS_SOURCE_SHA256: ${TOOLS_SOURCE_SHA256:-}
services:
  frontend:
    depends_on:
      backend:
        condition: service_healthy
  explorer-headless:
    profiles: ["explorer-headless"]
    build:
      dockerfile: Dockerfile.modular
      target: explorer-headless
    environment:
      - QT_QPA_PLATFORM=offscreen
      - MUJOCO_GL=${MUJOCO_GL:-egl}
  explorer-gui:
    profiles: ["explorer-gui"]
    build:
      dockerfile: Dockerfile.modular
      target: explorer-gui
    ports:
      - "127.0.0.1:6080:6080"
    environment:
      - QT_QPA_PLATFORM=xcb
      - MUJOCO_GL=${MUJOCO_GL:-egl}
      - DISPLAY=:99
      - NOVNC_PORT=6080
      - VNC_PORT=5900
  explorer-gui-wslg:
    profiles: ["explorer-gui-wslg"]
    build:
      dockerfile: Dockerfile.modular
      target: explorer-gui
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - /mnt/wslg:/mnt/wslg
    environment:
      - QT_QPA_PLATFORM=xcb
      - MUJOCO_GL=${MUJOCO_GL:-egl}
      - DISPLAY=${DISPLAY:-:0}
      - UD_EXPLORER_DISPLAY_MODE=host
  ud-sim:
    profiles: ["explorer-images"]
    build:
      dockerfile: Dockerfile.modular
      target: explorer-headless
    image: ud-sim:latest
    environment:
      - QT_QPA_PLATFORM=offscreen
      - MUJOCO_GL=${MUJOCO_GL:-egl}
"""

_EXPLORER_SH_OK = """#!/bin/sh
ensure_image() { docker image inspect "$1" || docker compose build "$2"; }
ensure_image upstream-drift:explorer-headless explorer-headless
ensure_image upstream-drift:explorer-gui explorer-gui
ensure_image ud-sim:latest ud-sim
ensure_image upstream-drift:full upstream-drift-full
"""

_ENTRYPOINT_OK = """#!/bin/sh
set -eu
die() { echo "ERROR: $*" >&2; exit 64; }
[ -n "${QT_QPA_PLATFORM:-}" ] || die "QT_QPA_PLATFORM is not set"
[ -n "${MUJOCO_GL:-}" ] || die "MUJOCO_GL is not set"
[ -n "${DISPLAY:-}" ] || die "no display: DISPLAY is not set"
    *)
        exec "$mode" "$@"
        ;;
esac
"""


_DOCKERIGNORE_OK = """# Exclude root environment secrets, but keep the example template.
.env
.env.*
!.env.example
"""


def _write_explorer_tree(root: Path) -> None:
    (root / "docker").mkdir(exist_ok=True)
    (root / "docker" / "explorer").mkdir(exist_ok=True)
    (root / "docker" / "ud-sim").mkdir(exist_ok=True)
    (root / "scripts" / "docker").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "docker" / "explorer.sh").write_text(_EXPLORER_SH_OK)
    (root / "docker" / "explorer" / "entrypoint.sh").write_text(
        _ENTRYPOINT_OK, encoding="utf-8"
    )
    (root / "docker" / "ud-sim" / "README.md").write_text(
        "Dockerfile.modular explorer-headless ud-sim:latest\n",
        encoding="utf-8",
    )


def _write_minimal_tree(root: Path, *, modular_pip: str = "26.1.2") -> None:
    (root / ".dockerignore").write_text(_DOCKERIGNORE_OK, encoding="utf-8")
    (root / "Dockerfile").write_text(
        """
RUN set -eu; for crate in rust_core/upstream-physics; do maturin build --release --features python -m "$crate/Cargo.toml" --out /wheels || exit 1; done; ls -1 /wheels
RUN pip install --upgrade pip==26.1.2
ARG SKIP_AUDIT=false
RUN pip install pip-audit==2.10.0
COPY scripts/ci/check_pip_audit_waivers.py /tmp/check_pip_audit_waivers.py
COPY scripts/config/pip_audit_waivers.json /tmp/pip_audit_waivers.json
ENV PATH="/opt/venv/bin:$PATH" \\
    PYTHONPATH="/workspace" \\
    MUJOCO_GL=osmesa
HEALTHCHECK CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=5)"
""",
        encoding="utf-8",
    )
    (root / "Dockerfile.modular").write_text(
        f"""
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN python scripts/docker/install_features.py --profile standard --dry-run
COPY launch_golf_suite.py ./
COPY launch_upstream_drift.py ./
RUN python scripts/docker/install_features.py --profile "$PROFILE"
RUN pip install --upgrade pip=={modular_pip}
{_EXPLORER_DOCKERFILE_OK}
""",
        encoding="utf-8",
    )
    (root / "Dockerfile.heavy_test").write_text(
        """
RUN pip install PyQt6
RUN pip install drake
RUN pip install opensim
RUN pip install myosuite
""",
        encoding="utf-8",
    )
    (root / "docker-compose.yml").write_text(_COMPOSE_OK, encoding="utf-8")
    _write_explorer_tree(root)


def test_docker_contracts_pass_for_current_shape(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)

    assert guard.docker_contract_failures(tmp_path) == []


def test_docker_contracts_require_root_dockerignore(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / ".dockerignore").unlink()

    assert any("root .dockerignore" in failure for failure in _failures(tmp_path))


def test_docker_contracts_ignore_comments_when_checking_secret_rules(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / ".dockerignore").write_text(
        "# .env rules in comments do not count\n"
        "# !.env.credentials is not an exception\n"
        + _DOCKERIGNORE_OK
        + "\n# Benign trailing comment\n\n",
        encoding="utf-8",
    )

    assert _failures(tmp_path) == []


@pytest.mark.parametrize("directive", [".env", ".env.*", "!.env.example"])
@pytest.mark.parametrize("commented_out", [False, True])
def test_docker_contracts_require_each_secret_rule(
    tmp_path: Path, directive: str, commented_out: bool
) -> None:
    _write_minimal_tree(tmp_path)
    lines = _DOCKERIGNORE_OK.splitlines()
    index = lines.index(directive)
    if commented_out:
        lines[index] = f"# {directive}"
    else:
        lines.pop(index)
    (tmp_path / ".dockerignore").write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert any("secret rules" in failure for failure in _failures(tmp_path))


def test_docker_contracts_require_secret_exclusions_before_example_reinclude(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / ".dockerignore").write_text(
        "!.env.example\n.env\n.env.*\n", encoding="utf-8"
    )

    assert any("secret rules" in failure for failure in _failures(tmp_path))


@pytest.mark.parametrize("negation", ["!.env.local", "!*.env.*"])
def test_docker_contracts_reject_later_secret_reinclude(
    tmp_path: Path, negation: str
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / ".dockerignore").write_text(
        _DOCKERIGNORE_OK + negation + "\n", encoding="utf-8"
    )

    assert any("secret rules" in failure for failure in _failures(tmp_path))


def test_docker_contracts_reject_pip_pin_skew(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, modular_pip="25.3")

    assert any(
        "pip pins diverge" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_masked_heavy_install(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.heavy_test").write_text(
        "RUN pip install drake || echo skip\n",
        encoding="utf-8",
    )

    assert any(
        "masks drake install failure" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_modular_profile_dry_run_before_registry_copy(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.modular").write_text(
        """
RUN python scripts/docker/install_features.py --profile standard --dry-run
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN pip install --upgrade pip==26.1.2
""",
        encoding="utf-8",
    )

    assert any(
        "copy feature_registry before profile dry-run" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def test_docker_contracts_reject_modular_profile_dry_run_before_engine_core_copy(
    tmp_path: Path,
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.modular").write_text(
        """
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
RUN python scripts/docker/install_features.py --profile standard --dry-run
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN pip install --upgrade pip==26.1.2
""",
        encoding="utf-8",
    )

    assert any(
        "copy engine_core before profile dry-run" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


@pytest.mark.parametrize(
    "launcher", ["launch_golf_suite.py", "launch_upstream_drift.py"]
)
def test_docker_contracts_reject_modular_feature_install_before_launcher_copy(
    tmp_path: Path, launcher: str
) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile.modular").write_text(
        f"""
COPY src/shared/python/feature_registry/ ./src/shared/python/feature_registry/
COPY src/shared/python/engine_core/ ./src/shared/python/engine_core/
RUN python scripts/docker/install_features.py --profile standard --dry-run
RUN python scripts/docker/install_features.py --profile "$PROFILE"
COPY {launcher} ./
RUN pip install --upgrade pip==26.1.2
""",
        encoding="utf-8",
    )

    assert any(
        f"copy {launcher} before feature install" in failure
        for failure in guard.docker_contract_failures(tmp_path)
    )


def _failures(tmp_path: Path) -> list[str]:
    return guard.docker_contract_failures(tmp_path)


def test_explorer_contract_requires_explorer_stages(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    modular = (tmp_path / "Dockerfile.modular").read_text()
    (tmp_path / "Dockerfile.modular").write_text(
        modular.replace("AS explorer-gui", "AS something-else"), encoding="utf-8"
    )

    assert any("explorer-gui" in f for f in _failures(tmp_path))


def test_explorer_targets_share_one_base_stage(tmp_path: Path) -> None:
    """DRY: headless and GUI targets must both derive from explorer-base."""
    _write_minimal_tree(tmp_path)
    modular = (tmp_path / "Dockerfile.modular").read_text()
    (tmp_path / "Dockerfile.modular").write_text(
        modular.replace(
            "FROM explorer-base AS explorer-headless",
            "FROM runtime AS explorer-headless",
        ),
        encoding="utf-8",
    )

    assert any("explorer-base" in f for f in _failures(tmp_path))


@pytest.mark.parametrize(
    "service", ["explorer-headless", "explorer-gui", "explorer-gui-wslg"]
)
def test_explorer_contract_requires_compose_services(
    tmp_path: Path, service: str
) -> None:
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    (tmp_path / "docker-compose.yml").write_text(
        compose.replace(f"  {service}:", f"  renamed-{service}:"), encoding="utf-8"
    )

    assert any(service in f for f in _failures(tmp_path))


@pytest.mark.parametrize(
    ("service", "needle"),
    [
        ("explorer-headless", "QT_QPA_PLATFORM=offscreen"),
        ("explorer-headless", "MUJOCO_GL="),
        ("explorer-gui", "QT_QPA_PLATFORM=xcb"),
        ("explorer-gui", "MUJOCO_GL="),
        ("explorer-gui", "DISPLAY=:99"),
        ("explorer-gui", "NOVNC_PORT=6080"),
        ("explorer-gui", "VNC_PORT=5900"),
        ("explorer-gui-wslg", "DISPLAY="),
        ("explorer-gui-wslg", "/tmp/.X11-unix:/tmp/.X11-unix"),
    ],
)
def test_explorer_contract_requires_service_env(
    tmp_path: Path, service: str, needle: str
) -> None:
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    head, _, tail = compose.partition(f"  {service}:")
    block, sep, rest = tail.partition("\n  explorer-")
    assert needle in block
    (tmp_path / "docker-compose.yml").write_text(
        head + f"  {service}:" + block.replace(needle, "REMOVED") + sep + rest,
        encoding="utf-8",
    )

    assert any(service in f and needle.split("=")[0] in f for f in _failures(tmp_path))


def test_explorer_gui_ports_bind_loopback_only(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    (tmp_path / "docker-compose.yml").write_text(
        compose.replace('"127.0.0.1:6080:6080"', '"6080:6080"'), encoding="utf-8"
    )

    assert any("loopback" in f for f in _failures(tmp_path))


@pytest.mark.parametrize("variable", ["QT_QPA_PLATFORM", "MUJOCO_GL", "DISPLAY"])
def test_explorer_entrypoint_fails_fast_on_missing_prerequisites(
    tmp_path: Path, variable: str
) -> None:
    _write_minimal_tree(tmp_path)
    entry = tmp_path / "docker" / "explorer" / "entrypoint.sh"
    entry.write_text(
        entry.read_text().replace(f'[ -n "${{{variable}:-}}" ]', "true"),
        encoding="utf-8",
    )

    assert any(variable in f and "entrypoint" in f for f in _failures(tmp_path))


def test_explorer_entrypoint_missing_is_reported(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "docker" / "explorer" / "entrypoint.sh").unlink()

    assert any("entrypoint" in f for f in _failures(tmp_path))


@pytest.mark.parametrize("arg", ["TOOLS_GITLINK_SHA", "TOOLS_SOURCE_SHA256"])
def test_explorer_build_must_pass_pinned_tools_provenance(
    tmp_path: Path, arg: str
) -> None:
    """The image build fails closed without the pinned-Tools build args."""
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    (tmp_path / "docker-compose.yml").write_text(
        compose.replace(arg, "RENAMED"), encoding="utf-8"
    )

    assert any(arg in f for f in _failures(tmp_path))


def test_explorer_launcher_script_is_required(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "scripts" / "docker" / "explorer.sh").unlink()

    assert any("explorer.sh" in f for f in _failures(tmp_path))


def test_explorer_images_must_be_labelled_keep(tmp_path: Path) -> None:
    """Host prune jobs spare label keep; unlabelled images were deleted before."""
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    (tmp_path / "docker-compose.yml").write_text(
        compose.replace('keep: "true"', "other: x"), encoding="utf-8"
    )

    assert any("keep" in f for f in _failures(tmp_path))


def test_runtime_stage_must_be_labelled_keep(tmp_path: Path) -> None:
    """UD #13113: profile images (upstream-drift:full et al.) build from runtime.

    Without the label, ControlTower's disk guard deleted every profile image and
    the ud-sim image built on top of upstream-drift:full.
    """
    _write_minimal_tree(tmp_path)
    modular = (tmp_path / "Dockerfile.modular").read_text()
    (tmp_path / "Dockerfile.modular").write_text(
        modular.replace(
            'AS runtime\nLABEL keep="true"\n', "AS runtime\nLABEL other=x\n"
        ),
        encoding="utf-8",
    )

    assert any("runtime" in f and "keep" in f for f in _failures(tmp_path))


def test_keep_label_in_a_later_stage_does_not_cover_runtime(tmp_path: Path) -> None:
    """The label must sit inside the runtime stage, not merely somewhere in the file."""
    _write_minimal_tree(tmp_path)
    modular = (tmp_path / "Dockerfile.modular").read_text()
    modular = modular.replace('AS runtime\nLABEL keep="true"\n', "AS runtime\n")
    (tmp_path / "Dockerfile.modular").write_text(
        modular + '\nFROM runtime AS other\nLABEL keep="true"\n', encoding="utf-8"
    )

    assert any("runtime" in f and "keep" in f for f in _failures(tmp_path))


@pytest.mark.parametrize(
    "image",
    [
        "upstream-drift:explorer-headless",
        "upstream-drift:explorer-gui",
        "ud-sim:latest",
        "upstream-drift:full",
    ],
)
def test_explorer_launcher_health_checks_each_required_image(
    tmp_path: Path, image: str
) -> None:
    _write_minimal_tree(tmp_path)
    launcher = tmp_path / "scripts" / "docker" / "explorer.sh"
    launcher.write_text(launcher.read_text().replace(image, "x"), encoding="utf-8")

    assert any(image in f for f in _failures(tmp_path))


def test_ud_sim_readme_documents_compose_source_of_truth(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    (tmp_path / "docker" / "ud-sim" / "README.md").unlink()

    assert any("ud-sim" in f and "README" in f for f in _failures(tmp_path))


def test_ud_sim_service_must_tag_explorer_headless_build(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    start = compose.index("  ud-sim:\n")
    ud_sim_tail = compose[start:].replace(
        "target: explorer-headless", "target: runtime", 1
    )
    (tmp_path / "docker-compose.yml").write_text(
        compose[:start] + ud_sim_tail, encoding="utf-8"
    )

    assert any("explorer-headless" in f and "ud-sim" in f for f in _failures(tmp_path))


def test_ud_sim_service_must_not_set_default_command(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path)
    compose = (tmp_path / "docker-compose.yml").read_text()
    block_start = compose.index("  ud-sim:")
    (tmp_path / "docker-compose.yml").write_text(
        compose[:block_start]
        + compose[block_start:].replace(
            "    image: ud-sim:latest\n",
            '    image: ud-sim:latest\n    command: ["engines"]\n',
            1,
        ),
        encoding="utf-8",
    )

    assert any("command" in f and "ud-sim" in f for f in _failures(tmp_path))


def test_dockerfile_maturin_loop_rejects_unguarded_for_loop(tmp_path: Path) -> None:
    """Issue #15112: unguarded maturin loop exits 0 when earlier crates fail."""
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile").write_text(
        (tmp_path / "Dockerfile")
        .read_text(encoding="utf-8")
        .replace(
            'RUN set -eu; for crate in rust_core/upstream-physics; do maturin build --release --features python -m "$crate/Cargo.toml" --out /wheels || exit 1; done; ls -1 /wheels',
            'RUN mkdir -p /wheels && cd /tmp/rust && for crate in a b; do maturin build -m "$crate/Cargo.toml" --out /wheels; done && ls -1 /wheels',
        ),
        encoding="utf-8",
    )

    failures = _failures(tmp_path)
    assert any("set -e" in f for f in failures)
    assert any("done &&" in f for f in failures)
    assert any("|| exit 1" in f for f in failures)


def test_dockerfile_maturin_loop_rejects_done_and_and_compound_chain(
    tmp_path: Path,
) -> None:
    """POSIX /bin/sh suppresses set -e inside any `for ... done && ...` command."""
    _write_minimal_tree(tmp_path)
    (tmp_path / "Dockerfile").write_text(
        (tmp_path / "Dockerfile")
        .read_text(encoding="utf-8")
        .replace(
            'RUN set -eu; for crate in rust_core/upstream-physics; do maturin build --release --features python -m "$crate/Cargo.toml" --out /wheels || exit 1; done; ls -1 /wheels',
            'RUN set -eu; for crate in a b; do maturin build -m "$crate/Cargo.toml" --out /wheels || exit 1; done && ls -1 /wheels',
        ),
        encoding="utf-8",
    )

    assert any("done &&" in f for f in _failures(tmp_path))


def test_repository_dockerfiles_satisfy_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    assert guard.docker_contract_failures(repo_root) == []
