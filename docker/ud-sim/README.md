# Ud-Sim:Latest Image

<host> jobs (including Brick Model Parity `ct-run.sh`) expect a local
Docker image tagged **`ud-sim:latest`**. There is **no** `docker/ud-sim/Dockerfile`
and no registry publish step for this tag.

## Source of Truth

| Field           | Value                                                            |
| --------------- | ---------------------------------------------------------------- |
| Image tag       | `ud-sim:latest`                                                  |
| Dockerfile      | `Dockerfile.modular` (repo root)                                 |
| Build target    | `explorer-headless`                                              |
| Compose service | `ud-sim`                                                         |
| Compose profile | `explorer-images`                                                |
| Default profile | `full` (`EXPLORER_PROFILE`, same as the Model Explorer services) |

The `ud-sim` service builds the same filesystem as `upstream-drift:explorer-headless`
but keeps the historical tag so existing `docker run ud-sim:latest …` scripts keep
working.

## Build and Health Check

From the repository root (WSL distro that owns the Docker daemon):

```bash
scripts/docker/explorer.sh ensure
# or rebuild only this tag:
docker compose --profile explorer-images build ud-sim
```

`scripts/docker/explorer.sh ensure` runs `docker image inspect` on `ud-sim:latest`
and rebuilds the `ud-sim` service when the tag is missing after a host prune.

## Ct-Run.Sh Contract

Jobs assume:

1. **Tag** — `ud-sim:latest` exists locally (never pulled from a registry).
2. **Entrypoint** — `docker/explorer/entrypoint.sh` forwards any unknown first
   argument with `exec` (for example `docker run --rm ud-sim:latest python3 -m …`).
3. **Headless GL** — pass `-e QT_QPA_PLATFORM=offscreen -e MUJOCO_GL=egl` when
   not using the compose service defaults (the `ud-sim` service sets these).
4. **Workspace** — mount the repo at `/workspace` the same way other <host>
   receipts document.

The default `docker run ud-sim:latest` with no command runs the `engines` smoke
mode; batch scripts should always pass an explicit command.

## Related Docs

- Model Explorer compose layout: `docs/development/DOCKER_SETUP.md`
- Explorer launcher: `scripts/docker/explorer.sh`
- Image retention (`keep=true`): `docs/development/DOCKER_SETUP.md` (Image
  Retention and Health Check)
