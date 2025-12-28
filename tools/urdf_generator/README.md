# Interactive URDF Generator

An interactive GUI tool for creating URDF (Unified Robot Description Format) files with support for parallel kinematic configurations, specifically designed for the Golf Modeling Suite.

## Features

- **Interactive GUI**: PyQt6-based interface for intuitive URDF creation
- **Segment-by-Segment Building**: Add links and joints one at a time with real-time preview
- **Parallel Kinematic Support**: Handle complex parallel configurations common in golf swing modeling
- **Multi-Engine Export**: Optimized export for MuJoCo, Drake, and Pinocchio physics engines
- **Golf-Specific Templates**: Pre-configured templates for golf clubs, balls, and tees
- **3D Visualization**: Real-time 3D preview of the URDF model (implementation in progress)
- **Validation**: Built-in URDF validation and error checking

## Installation

### Prerequisites

- Python 3.11+
- PyQt6
- NumPy

### Install Dependencies

```bash
cd tools/urdf_generator
pip install -r requirements.txt
```

## Usage

### Launch the Application

```bash
python tools/urdf_generator/launch_urdf_generator.py
```

Or from the project root:

```bash
python -m tools.urdf_generator.launch_urdf_generator
```

### Basic Workflow

1. **Start the Application**: Launch the URDF Generator
2. **Add Segments**: Use the segment panel to add links and joints
3. **Configure Properties**: Set geometry, physics, and joint properties
4. **Preview**: View the 3D visualization (when implemented)
5. **Export**: Save as URDF or export for specific physics engines

### Segment Types

- **Link**: Basic rigid body component
- **Joint**: Connection between links
- **Golf Club Shaft**: Pre-configured shaft segment
- **Golf Club Head**: Pre-configured club head
- **Golf Ball**: Standard golf ball geometry
- **Tee**: Golf tee geometry
- **Ground Plane**: Ground reference

### Joint Types

- **Fixed**: No movement
- **Revolute**: Rotational joint with limits
- **Prismatic**: Linear joint with limits
- **Continuous**: Unlimited rotation
- **Floating**: 6-DOF free joint
- **Planar**: Planar motion joint

## Architecture

### Core Components

- **URDFBuilder**: Generates URDF XML from segment data
- **SegmentManager**: Manages segment hierarchy and parallel chains
- **SegmentPanel**: GUI for segment creation and editing
- **VisualizationWidget**: 3D preview widget (placeholder implementation)
- **MainWindow**: Main application window and coordination

### Parallel Kinematic Support

The generator supports parallel kinematic chains through:

- Custom parallel chain definitions
- Constraint handling for closed loops
- Export optimizations for different physics engines

## Engine-Specific Features

### MuJoCo Export
- Optimized for MuJoCo's XML format
- Proper handling of parallel constraints
- Material and visual optimizations

### Drake Export
- Compatible with Drake's URDF parser
- Inertia tensor optimizations
- Joint limit handling

### Pinocchio Export
- Optimized for Pinocchio's kinematic chains
- Proper mass distribution
- Efficient joint representations

## File Structure

```
tools/urdf_generator/
├── __init__.py              # Package initialization
├── main_window.py           # Main application window
├── segment_panel.py         # Segment editing interface
├── urdf_builder.py          # URDF XML generation
├── segment_manager.py       # Segment hierarchy management
├── visualization_widget.py  # 3D visualization (placeholder)
├── launch_urdf_generator.py # Application launcher
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_urdf_generator.py -v
```

## Future Enhancements

### Planned Features

1. **3D Visualization**: Complete Open3D or OpenGL integration
2. **URDF Import**: Load and edit existing URDF files
3. **Mesh Support**: Import and use custom 3D meshes
4. **Advanced Physics**: Damping, friction, and contact properties
5. **Templates**: More golf-specific and general robotics templates
6. **Undo/Redo**: Full undo/redo functionality
7. **Scripting**: Python API for automated URDF generation

### Visualization Implementation Options

1. **Open3D Integration**: Recommended for robust 3D visualization
2. **PyOpenGL**: Custom OpenGL implementation
3. **VTK Integration**: Advanced visualization capabilities
4. **Web-based**: Three.js integration for browser-based preview

## Integration with Golf Modeling Suite

The URDF Generator integrates seamlessly with the Golf Modeling Suite:

- **MuJoCo Models**: Direct export to `engines/physics_engines/mujoco/`
- **Drake Models**: Compatible with `engines/physics_engines/drake/`
- **Pinocchio Models**: Optimized for `engines/physics_engines/pinocchio/`
- **Shared Resources**: Uses `shared/` directory for common utilities

## Contributing

1. Follow the project's coding standards (see `AGENTS.md`)
2. Add tests for new features
3. Update documentation
4. Use conventional commit messages

## License

Part of the Golf Modeling Suite - MIT License