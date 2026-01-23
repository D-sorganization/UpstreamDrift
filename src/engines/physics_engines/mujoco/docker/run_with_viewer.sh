#!/bin/bash
# Script to run the robotics environment with X11 forwarding for real-time visualization.

# Host IP for WSL2/Docker Desktop on Windows
export DISPLAY=host.docker.internal:0.0

echo "Setting DISPLAY to $DISPLAY"
echo "Ensure VcXsrv is running on Windows with 'Disable access control' checked!"

# Run container
# We override MUJOCO_GL to 'glfw' to use the window system instead of osmesa.
# We set -w /workspace so the output video is saved to the mounted volume.
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -e MUJOCO_GL="glfw" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    robotics_env \
    python3 /workspace/example_dynamic_stance.py
