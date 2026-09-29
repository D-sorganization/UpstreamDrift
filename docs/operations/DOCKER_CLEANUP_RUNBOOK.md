# Docker Cleanup Runbook (Windows + WSL2 + Docker Desktop)

## Purpose

Provide a repeatable, low-risk cleanup process for reclaiming disk space while preserving recoverability.

## Applies To

- Primary machine (current)
- Secondary machine with similar setup

## Safety Principles

- Capture inventory before deleting anything.
- Retag before removing legacy names.
- Remove containers first, then obviously unused images.
- Keep rollback path documented.

## 0) Pre-Cleanup Inventory (required)

Run in WSL/Linux shell:

```bash
date
mkdir -p /tmp/docker-cleanup

docker ps -a > /tmp/docker-cleanup/containers.before.txt
docker images > /tmp/docker-cleanup/images.before.txt
docker volume ls > /tmp/docker-cleanup/volumes.before.txt
docker system df -v > /tmp/docker-cleanup/system-df.before.txt
```

## 1) Normalize Image Tags (non-destructive)

```bash
# If legacy images exist, add canonical tags
(docker image inspect robotics_env:latest >/dev/null 2>&1 && docker tag robotics_env:latest upstream-drift:engine) || true
(docker image inspect golf-suite:latest >/dev/null 2>&1 && docker tag golf-suite:latest upstream-drift:runtime) || true
(docker image inspect upstream-drift:engine >/dev/null 2>&1 && docker tag upstream-drift:engine upstream-drift:dev) || true
```

## 2) Remove Stale Containers and Cache

```bash
docker container prune -f
docker builder prune -a -f
```

## 3) Remove Known Unused/Low-Value Images (safe candidates)

```bash
# Legacy NGC bases no longer referenced by current MLProjects Dockerfiles
(docker image rm nvcr.io/nvidia/pytorch:24.05-py3) || true
(docker image rm nvcr.io/nvidia/tensorflow:24.05-tf2-py3) || true

# Optional management UI images (if unused)
(docker image rm fnsys/dockhand:latest) || true
(docker image rm portainer/portainer-ce:latest) || true
(docker image rm docker/welcome-to-docker:latest) || true

# Remove legacy aliases after canonical tags exist
(docker image rm robotics_env:latest) || true
(docker image rm golf-suite:latest) || true
```

## 4) Optional Volume Cleanup

Only if you are sure no data is needed from old management tools:

```bash
(docker volume rm dockhand_data) || true
(docker volume rm portainer_data) || true
```

## 5) Post-Cleanup Inventory (required)

```bash
docker ps -a > /tmp/docker-cleanup/containers.after.txt
docker images > /tmp/docker-cleanup/images.after.txt
docker volume ls > /tmp/docker-cleanup/volumes.after.txt
docker system df -v > /tmp/docker-cleanup/system-df.after.txt
```

## 6) Reclaim Host Disk (Windows VHDX compaction)

Without this, Windows file size often does not shrink even after Docker prune.

Run in **Admin PowerShell**:

```powershell
wsl --shutdown
Optimize-VHD -Path "C:\Users\<USER>\AppData\Local\Docker\wsl\disk\docker_data.vhdx" -Mode Full
```

## Rollback Commands

```bash
# Restore legacy tags if older scripts expect them
(docker image inspect upstream-drift:engine >/dev/null 2>&1 && docker tag upstream-drift:engine robotics_env:latest) || true
(docker image inspect upstream-drift:runtime >/dev/null 2>&1 && docker tag upstream-drift:runtime golf-suite:latest) || true
```

## Notes for ML Workloads

- Keep ML training images separate from core product runtime.
- Prefer versioned tags over reusing `latest`.
- Use a documented artifact handoff into UpstreamDrift instead of embedding all training deps into `upstream-drift:runtime`.
