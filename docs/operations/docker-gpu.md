# Docker GPU Support

UpstreamDrift Docker containers work with or without a GPU.

### Requirements for GPU Mode

- NVIDIA GPU (any recent GeForce, Quadro, or Tesla)
- NVIDIA drivers installed on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- Docker Desktop (Windows/Mac) or Docker Engine (Linux) with NVIDIA runtime

### Running with GPU

```bash
# CPU (default — works everywhere)
docker compose up

# GPU-accelerated
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

# GPU with specific device
NVIDIA_VISIBLE_DEVICES=0 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Configuration

Copy `.env.docker.example` to `.env` and adjust:

- `NVIDIA_GPU_COUNT` — How many GPUs to expose (`all`, `1`, etc.)
- `MUJOCO_GL` — Rendering backend (`egl` for GPU, `osmesa` for CPU)
- `NVIDIA_VISIBLE_DEVICES` — Which GPU(s) to use

### eGPU Users

If using an external GPU (Thunderbolt eGPU enclosure), the setup works
identically — NVIDIA Container Toolkit handles the passthrough. Note that
disconnecting the eGPU while containers are running will crash GPU containers.
Use `docker compose restart` after reconnecting.

### Verifying GPU Access

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm backend nvidia-smi
```
