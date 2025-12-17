# MuJoCo Golf Swing Model - Docker Development Environment
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libosmesa6-dev \
    patchelf \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Create Python venv for MuJoCo environment
RUN python3.11 -m venv /opt/mujoco-env && \
    /opt/mujoco-env/bin/pip install --upgrade pip setuptools wheel

# Install MuJoCo and dependencies
RUN /opt/mujoco-env/bin/pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    "matplotlib>=3.8.0,<=3.9.1" \
    scipy==1.13.1 \
    mujoco>=3.0.0 \
    PyQt6>=6.6.0 \
    opencv-python>=4.8.0 \
    imageio>=2.31.0 \
    scikit-learn>=1.3.0 \
    h5py>=3.9.0 \
    pillow>=10.0.0 \
    pytest==8.2.1 \
    pytest-cov==5.0.0 \
    pytest-mock==3.14.0 \
    ruff==0.5.0 \
    black==24.4.2 \
    mypy==1.10.0 \
    pre-commit==4.0.0 \
    pyyaml==6.0.1 \
    "pathlib2>=2.3.6,<=2.3.7" \
    types-PyYAML==6.0.12.12 \
    "defusedxml>=0.7.1"

# Optional: Install Pinocchio for advanced dynamics (uncomment if needed)
# Optional: Install Pinocchio for advanced dynamics
# Uncommented and verified package name 'pin' works for pinocchio bindings
RUN /opt/mujoco-env/bin/pip install pin>=2.6.0

# Install DeepMind Robotics Tools
# Includes dm_control (Humanoid CMU), JAX (for MJX), and Mujoco Playground
RUN /opt/mujoco-env/bin/pip install \
    dm_control>=1.0.14 \
    jax[cuda12_pip] \
    git+https://github.com/google-deepmind/mujoco_playground.git@v0.1.0

# Install Advanced Robotics & Visualization
# pin-pink: Inverse kinematics based on Pinocchio
# meshcat: Web-based 3D visualizer (alternative to Geppetto)
# drake: Model-based design and verification for robotics
# NOTE: Drake is a very large package with extensive dependencies.
# Installing it will significantly increase build time and image size.
# Uncomment the following line ONLY if you require Drake for your workflow.
#     drake \
RUN /opt/mujoco-env/bin/pip install \
    pin-pink \
    meshcat \
    qpsolvers

# Install MoCapAct (Microsoft)
# Using --no-deps to avoid downgrading dm_control/mujoco (MoCapAct pins older versions)
RUN /opt/mujoco-env/bin/pip install --no-deps git+https://github.com/microsoft/MoCapAct.git@v1.0.0 && \
    /opt/mujoco-env/bin/pip install azure-storage-blob cloudpickle

# Activate virtual environment by default
ENV PATH="/opt/mujoco-env/bin:$PATH"
ENV VIRTUAL_ENV="/opt/mujoco-env"

# Set working directory
WORKDIR /workspace

# Default command (can be overridden)
CMD ["/bin/bash"]


