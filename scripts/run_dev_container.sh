#!/bin/bash
# Enter Development Environment
# -----------------------------
# This script builds (if missing) and runs the Docker development container
# for the Golf Modeling Suite.

IMAGE_NAME="golf-suite-dev"

# 1. Check if image exists, build if needed
if [[ "$(docker images -q ${IMAGE_NAME} 2> /dev/null)" == "" ]]; then
  echo "Image '${IMAGE_NAME}' not found. Building..."
  docker build -t ${IMAGE_NAME} -f engines/physics_engines/mujoco/Dockerfile .
else
  echo "Image '${IMAGE_NAME}' found."
fi

# 2. Run Container
echo "Starting development container..."
echo "Mounting: $(pwd) -> /workspace"

# Detect OS for TTY handling if needed, but standard run is usually fine
docker run -it --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  ${IMAGE_NAME} /bin/bash
