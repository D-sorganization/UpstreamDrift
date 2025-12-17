# Golf Swing Visualizer - Wiffle_ProV1 Edition

## Overview

This is the most refined version of the golf swing visualization system, specifically enhanced to handle Wiffle_ProV1 motion capture data from Excel files. The system provides advanced 3D visualization, real-time analysis, and comparison capabilities between different ball types.

## Key Features

### 🎯 **Core Capabilities**
- **Excel Data Loading**: Direct support for Wiffle_ProV1 Excel files
- **3D Visualization**: High-performance OpenGL rendering with realistic lighting
- **Motion Capture Display**: Full body segment visualization with club and ball tracking
- **Real-time Analysis**: Live metrics calculation and comparison
- **Interactive Controls**: Playback, camera controls, filtering options

### 🔧 **Advanced Features**
- **Ball Comparison**: Side-by-side ProV1 vs Wiffle analysis
- **Performance Monitoring**: FPS tracking and optimization
- **Data Processing**: Noise filtering, interpolation, and normalization
- **Export Capabilities**: Screenshot and data export functionality
- **Modern UI**: Professional PyQt6 interface with dockable panels

## System Architecture

### **Most Refined Version: `integrated_golf_gui_r0/`**

This directory contains the most sophisticated implementation with:

1. **`golf_wiffle_main.py`** - Main application with Wiffle_ProV1 support
2. **`wiffle_data_loader.py`** - Excel data loading and processing
3. **`golf_gui_application.py`** - Core GUI framework
4. **`golf_data_core.py`** - Data structures and processing
5. **`golf_opengl_renderer.py`** - High-performance 3D rendering

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install required packages
pip install -r requirements.txt
```

### Required Dependencies

```txt
# Core dependencies
numpy>=1.20.0
scipy>=1.8.0
pandas>=1.3.0
matplotlib>=3.5.0

# GUI and visualization
PyQt6>=6.2.0
moderngl>=5.6.0

# Data processing
numba>=0.55.0

# Testing and development
pytest>=6.2.0
flake8>=4.0.0

# Biomechanics
filterpy>=1.4.5
```

### Quick Start

1. **Navigate to the integrated GUI directory:**
   ```bash
   cd integrated_golf_gui_r0
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main application:**
   ```bash
   python golf_wiffle_main.py
   ```

4. **Load your Wiffle_ProV1 data:**
   - The application will automatically try to load `Matlab Inverse Dynamics/Wiffle_ProV1_club_3D_data.xlsx`
   - Or use File → Load Excel File to select your data

## Data Format Requirements

### Excel File Structure

The system expects an Excel file with two sheets:

1. **ProV1 Sheet**: Contains ProV1 ball motion capture data
2. **Wiffle Sheet**: Contains Wiffle ball motion capture data

### Required Columns

Each sheet should contain the following columns (or similar):

```python
# Position data columns
'Time'          # Time vector
'CHx', 'CHy', 'CHz'    # Clubhead position
'Bx', 'By', 'Bz'       # Butt position
'MPx', 'MPy', 'MPz'    # Midpoint position
'LWx', 'LWy', 'LWz'    # Left wrist position
'LEx', 'LEy', 'LEz'    # Left elbow position
'LSx', 'LSy', 'LSz'    # Left shoulder position
'RWx', 'RWy', 'RWz'    # Right wrist position
'REx', 'REy', 'REz'    # Right elbow position
'RSx', 'RSy', 'RSz'    # Right shoulder position
'Hx', 'Hy', 'Hz'       # Hub position
```

## Usage Guide

### Main Interface

The application features a modern interface with dockable panels:

1. **Data Loading Panel** (Left)
   - File selection and loading progress
   - Data information display

2. **Wiffle Controls Panel** (Left)
   - Ball type selection (ProV1/Wiffle/Difference)
   - Data processing options
   - Reload functionality

3. **Analysis Panel** (Right)
   - Comparison controls
   - Performance metrics
   - Export options

4. **3D Visualization** (Center)
   - Interactive 3D view
   - Camera controls
   - Playback controls

### Key Controls

#### **Playback Controls**
- **Play/Pause**: Start/stop animation
- **Frame Slider**: Manual frame selection
- **Speed Control**: Adjust playback speed

#### **Camera Controls**
- **Mouse Drag**: Rotate camera
- **Mouse Wheel**: Zoom in/out
- **Reset Camera**: Return to default view

#### **Visualization Options**
- **Ball Type**: Switch between ProV1, Wiffle, or Difference view
- **Show/Hide**: Toggle visibility of different elements
- **Filtering**: Apply noise reduction and smoothing

### Analysis Features

#### **Real-time Metrics**
- Maximum clubhead speed comparison
- Trajectory difference analysis
- Frame-by-frame distance calculations

#### **Data Export**
- Export comparison data to CSV
- Screenshot capture
- Video recording capabilities

## Data Processing Options

### **Noise Filtering**
- **Savitzky-Golay Filter**: Smooths position data
- **Configurable Window**: Automatic window size calculation
- **Edge Handling**: Proper boundary condition handling

### **Data Interpolation**
- **Missing Value Handling**: Linear interpolation for gaps
- **Time Normalization**: Optional time scaling
- **Data Validation**: Automatic error detection

### **Performance Optimization**
- **Numba Acceleration**: JIT-compiled calculations
- **OpenGL Rendering**: Hardware-accelerated visualization
- **Memory Management**: Efficient data structures

## Troubleshooting

### Common Issues

1. **OpenGL Errors**
   ```bash
   # Check OpenGL support
   python -c "import moderngl; print('OpenGL supported')"
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install --upgrade -r requirements.txt
   ```

3. **Data Loading Errors**
   - Verify Excel file format
   - Check column names match expected format
   - Ensure both ProV1 and Wiffle sheets exist

4. **Performance Issues**
   - Reduce visualization quality settings
   - Close other applications
   - Update graphics drivers

### Debug Mode

Enable debug output by setting environment variable:
```bash
export PYTHONPATH=.
python golf_wiffle_main.py
```

## Comparison with Other Versions

### **SkeletonPlotter** (MATLAB)
- ✅ Good for basic visualization
- ❌ Limited to MATLAB environment
- ❌ No Excel data support
- ❌ Basic UI controls

### **Matlab Inverse Dynamics** (MATLAB)
- ✅ Excel data support
- ✅ Basic comparison features
- ❌ Limited 3D visualization
- ❌ MATLAB dependency

### **integrated_golf_gui_r0** (Python - **RECOMMENDED**)
- ✅ Full Excel data support
- ✅ Advanced 3D visualization
- ✅ Professional UI
- ✅ Cross-platform compatibility
- ✅ Performance optimization
- ✅ Comprehensive analysis tools

## Development

### Project Structure

```
integrated_golf_gui_r0/
├── golf_wiffle_main.py          # Main application
├── wiffle_data_loader.py        # Excel data loader
├── golf_gui_application.py      # Core GUI framework
├── golf_data_core.py           # Data processing
├── golf_opengl_renderer.py     # 3D rendering
├── golf_inverse_dynamics.py    # Dynamics calculations
├── requirements.txt            # Dependencies
└── README_WIFFLE.md           # This file
```

### Testing

```bash
# Test data loader
python wiffle_data_loader.py

# Test core functionality
python simple_data_test.py

# Run full application
python golf_wiffle_main.py
```

### Contributing

1. Follow the existing code style
2. Add comprehensive error handling
3. Include performance optimizations
4. Update documentation for new features

## Performance Considerations

### **Optimization Features**
- **Numba JIT Compilation**: Accelerated mathematical operations
- **OpenGL Hardware Acceleration**: GPU-accelerated rendering
- **Efficient Data Structures**: Memory-optimized arrays
- **Background Loading**: Non-blocking data processing

### **Recommended Hardware**
- **GPU**: OpenGL 3.3+ compatible graphics card
- **RAM**: 8GB+ for large datasets
- **CPU**: Multi-core processor for data processing
- **Storage**: SSD for faster data loading

## Future Enhancements

### **Planned Features**
- **Machine Learning Analysis**: Automated swing analysis
- **Cloud Integration**: Remote data processing
- **Mobile Support**: Touch-optimized interface
- **Advanced Metrics**: Biomechanical analysis

### **Performance Improvements**
- **Vulkan Rendering**: Next-generation graphics API
- **Parallel Processing**: Multi-threaded data analysis
- **Memory Optimization**: Reduced memory footprint

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with sample data
4. Verify system requirements

## License

This project is part of the Golf Swing Visualization research system. Please respect the intellectual property and research context of this work.

---

**Version**: 1.0
**Last Updated**: 2024
**Status**: Production Ready
