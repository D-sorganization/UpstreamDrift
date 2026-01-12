# Professional Windows Installation Guide

## Golf Modeling Suite - Professional Edition

This guide covers the professional Windows installation process for Golf Modeling Suite, designed for research institutions, equipment manufacturers, and professional coaches.

## 🎯 Quick Start (Recommended)

### Option 1: MSI Installer (Coming Soon)
The easiest way to install Golf Modeling Suite on Windows with all dependencies managed automatically.

1. **Download the MSI installer** from our releases page
2. **Run the installer** with administrator privileges
3. **Select physics engines** during installation:
   - ✅ **MuJoCo** (Always included - core engine)
   - ☐ **Drake** (Trajectory optimization)
   - ☐ **Pinocchio** (High-performance dynamics)
   - ☐ **MyoSuite** (Muscle simulation)
   - ☐ **OpenSim** (Biomechanics standards)
4. **Launch** from Start Menu or Desktop shortcut

### Option 2: Professional Python Installation

For developers and advanced users who need full control over the installation.

## 📋 Prerequisites

### System Requirements
- **Windows 10/11** (64-bit)
- **8GB RAM minimum** (16GB recommended for multi-engine use)
- **5GB free disk space** (more for models and data)
- **Graphics card** with OpenGL 3.3+ support

### Software Prerequisites
- **Python 3.11+** ([Download from python.org](https://www.python.org/downloads/))
- **Git with Git LFS** ([Download from git-scm.com](https://git-scm.com/))
- **Visual Studio Build Tools** (for some physics engines)

## 🔧 Step-by-Step Installation

### 1. Install Python and Git

```powershell
# Download and install Python 3.11+ from python.org
# Make sure to check "Add Python to PATH" during installation

# Verify Python installation
python --version
pip --version

# Install Git with Git LFS
# Download from git-scm.com and install with default options
git --version
git lfs install
```

### 2. Create Virtual Environment (Recommended)

```powershell
# Create a dedicated virtual environment
python -m venv golf_modeling_suite
cd golf_modeling_suite
Scripts\activate

# Verify virtual environment
where python
# Should show path to your virtual environment
```

### 3. Clone Repository

```powershell
# Clone the repository with LFS support
git clone https://github.com/D-sorganization/Golf_Modeling_Suite.git
cd Golf_Modeling_Suite

# Pull large files (models, meshes)
git lfs pull
```

### 4. Install Base Package

```powershell
# Install core dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import shared.python.version; print(shared.python.version.__version__)"
```

### 5. Install Physics Engines (Modular)

Choose which physics engines to install based on your needs:

#### MuJoCo (Always Recommended)
```powershell
# MuJoCo is included in base installation
python -c "import mujoco; print('MuJoCo:', mujoco.__version__)"
```

#### Drake (Trajectory Optimization)
```powershell
pip install "drake>=1.22.0"
python -c "import pydrake; print('Drake installed successfully')"
```

#### Pinocchio (High-Performance Dynamics)
```powershell
pip install "pin>=2.6.0" "pin-pink>=1.0.0"
python -c "import pinocchio; print('Pinocchio:', pinocchio.__version__)"
```

#### MyoSuite (Muscle Simulation)
```powershell
pip install "myosuite>=2.0.0"
python -c "import myosuite; print('MyoSuite installed successfully')"
```

#### OpenSim (Biomechanics)
```powershell
# OpenSim requires additional setup - see OpenSim documentation
# pip install opensim
```

### 6. Setup Standard Models

```powershell
# Download and setup standard models
python -c "
from shared.python.standard_models import StandardModelManager
manager = StandardModelManager()
success = manager.setup_all_models()
print('Models setup:', 'Success' if success else 'Failed')
"
```

### 7. Verify Installation

```powershell
# Run comprehensive installation test
python -m pytest tests/test_installation.py -v

# Launch the application
python launchers/golf_launcher.py
```

## 🚀 Professional Features Setup

### API Server Setup

```powershell
# Install API dependencies
pip install "fastapi>=0.104.0" "uvicorn>=0.24.0"

# Start API server
python api/server.py
# Server will be available at http://localhost:8000
```

### Video Analysis Setup

```powershell
# Install video analysis dependencies
pip install "opencv-python>=4.8.0" "mediapipe>=0.10.0"

# Test video analysis
python -c "
from shared.python.video_pose_pipeline import VideoPosePipeline
pipeline = VideoPosePipeline()
print('Video analysis ready')
"
```

### Cloud Integration Setup

```powershell
# Install cloud dependencies
pip install "boto3>=1.34.0" "azure-storage-blob>=12.19.0"

# Configure cloud storage (optional)
# Set environment variables for your cloud provider
```

## 🔧 Troubleshooting

### Common Issues

#### "Python not found"
- Ensure Python is added to PATH during installation
- Restart command prompt after Python installation

#### "Git LFS not working"
- Run `git lfs install` in command prompt
- Re-clone repository if LFS files are missing

#### "Physics engine import errors"
- Check that virtual environment is activated
- Verify engine-specific dependencies are installed
- Some engines require Visual Studio Build Tools

#### "Model files not found"
- Run the model setup command again
- Check internet connection for model downloads
- Verify Git LFS is working properly

### Performance Optimization

#### For Research Use
```powershell
# Install all engines for cross-validation
pip install -e ".[engines,analysis,optimization]"
```

#### For Production Use
```powershell
# Install only required engines
pip install -e ".[analysis]"  # Core + video analysis
```

#### For Development
```powershell
# Install development tools
pip install -e ".[dev,engines,analysis,optimization]"
pre-commit install
```

## 📞 Professional Support

### Documentation
- **User Guide**: `docs/user_guide/`
- **API Documentation**: `http://localhost:8000/docs` (when server running)
- **Developer Guide**: `docs/developer_guide/`

### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Community Q&A and best practices

### Professional Support
- **Enterprise Support**: Available for research institutions and companies
- **Custom Development**: Tailored solutions for specific needs
- **Training**: Professional training programs available

## 🎓 Next Steps

1. **Complete the Tutorial**: `docs/tutorials/getting_started.md`
2. **Explore Examples**: `examples/` directory
3. **Read the User Guide**: `docs/user_guide/`
4. **Join the Community**: GitHub Discussions

## 📊 Installation Verification Checklist

- [ ] Python 3.11+ installed and in PATH
- [ ] Git with Git LFS installed and configured
- [ ] Virtual environment created and activated
- [ ] Repository cloned with LFS files
- [ ] Base package installed successfully
- [ ] At least one physics engine working
- [ ] Standard models downloaded
- [ ] Application launches without errors
- [ ] API server starts (if using API features)
- [ ] Video analysis works (if using video features)

**Congratulations!** You now have a professional-grade biomechanical analysis platform ready for research and development.