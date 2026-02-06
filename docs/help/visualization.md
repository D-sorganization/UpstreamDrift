# Visualization Settings

Configure 3D rendering, camera views, and display options.

## Overview

UpstreamDrift provides real-time 3D visualization of physics simulations with customizable rendering options, camera controls, and overlay displays.

## Camera Controls

### Mouse Controls

| Action | Control |
|--------|---------|
| Rotate view | Left-click + drag |
| Pan view | Right-click + drag |
| Zoom | Scroll wheel |
| Reset view | Middle-click |
| Quick rotate | Ctrl + drag |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1 | Side view (golfer's right) |
| 2 | Front view (face-on) |
| 3 | Top view (bird's eye) |
| 4 | Down-the-line view |
| 5 | Follow clubhead mode |
| F | Focus on selection |
| R | Reset camera |

### Preset Views

#### Side View (Key: 1)
- Camera positioned to golfer's right
- Best for: Swing plane analysis, spine angle
- Shows: Full swing arc, weight shift

#### Front View (Key: 2)
- Camera facing golfer directly
- Best for: Posture, balance, width
- Shows: Shoulder rotation, hip sway

#### Top View (Key: 3)
- Camera positioned above
- Best for: Swing path, body rotation
- Shows: Club path, shoulder turn

#### Down-the-Line (Key: 4)
- Camera behind golfer, facing target
- Best for: Club path, face angle
- Shows: Attack angle, swing direction

#### Follow Mode (Key: 5)
- Camera tracks clubhead position
- Best for: Impact analysis
- Shows: Ball contact, low point

## Display Options

### Model Rendering

| Option | Description |
|--------|-------------|
| Solid | Full textured rendering |
| Wireframe | Mesh structure only |
| Points | Joint/vertex points only |
| Transparent | Semi-transparent surfaces |

### Coordinate Frames

Display reference frames at:
- World origin
- Body/segment centers
- Joint locations
- End-effector (clubhead)

**Settings:**
- Frame size: Adjust arrow length
- Axis colors: X=Red, Y=Green, Z=Blue
- Show/hide specific frames

### Ground and Grid

| Option | Description |
|--------|-------------|
| Ground plane | Show/hide floor surface |
| Grid | Distance grid overlay |
| Target line | Line toward target |
| Impact zone | Ball position area |

### Shadows

- Enable/disable shadow rendering
- Shadow quality: Low/Medium/High
- Soft vs. hard shadows

## Force and Torque Visualization

### Force Vectors

Display force vectors at joints or contact points:

**Settings:**
- Show/hide forces
- Scale factor (cm per Newton)
- Color by: Magnitude, Type, or Direction
- Minimum threshold (hide small forces)

**Colors (by type):**
- Blue: Gravitational
- Green: Reaction forces
- Red: Applied/muscle forces
- Yellow: Contact forces

### Torque Visualization

Display joint torques as curved arrows:

**Settings:**
- Show/hide torques
- Scale factor (degrees per N-m)
- Arrow style: Curved or Straight
- Color coding options

### Ground Reaction Forces

Special visualization for foot-ground interaction:
- Force plate vectors
- Center of pressure path
- Vertical/horizontal components

## Energy Display

### Energy Panel

Real-time display of:
- Kinetic energy (KE)
- Potential energy (PE)
- Total energy (KE + PE)
- Energy conservation check

### Energy Colors

- Green: KE (motion)
- Blue: PE (height)
- Purple: Total
- Red: Energy loss (if any)

## Trajectory Trails

### Trail Options

Show paths traced by selected points:

| Setting | Description |
|---------|-------------|
| Trail length | Number of frames to display |
| Trail style | Line, dots, or fading |
| Trail color | Solid or velocity-coded |
| Points | Select which bodies to trail |

### Common Trail Targets

- Clubhead path
- Hand path
- Hip center
- Shoulder center

## Performance Options

### Render Quality

| Level | Description | Use Case |
|-------|-------------|----------|
| Low | Basic rendering | Slow hardware |
| Medium | Balanced | General use |
| High | Full quality | Screenshots |
| Ultra | Maximum detail | Presentation |

### Frame Rate

- **Sync to simulation:** Render every physics step
- **Fixed rate:** Render at 30/60 fps
- **Render every N:** Skip frames for speed

### Performance Tips

1. **Reduce trail length** for faster rendering
2. **Disable shadows** for complex scenes
3. **Use wireframe mode** during parameter tuning
4. **Lower quality** for real-time feedback

## Recording and Screenshots

### Screenshots

- **F12:** Capture current view
- **Ctrl+P:** Print-quality screenshot
- **Settings:** Resolution, format (PNG/JPG)

### Video Recording

1. Click "Record Video" button
2. Run simulation
3. Click "Stop Recording"
4. Choose format and location

**Formats:**
- MP4 (H.264)
- AVI (uncompressed)
- GIF (animated)

**Settings:**
- Resolution: 720p, 1080p, 4K
- Frame rate: 30, 60, 120 fps
- Quality: Low, Medium, High

## Lighting

### Light Types

| Type | Description |
|------|-------------|
| Ambient | Overall scene brightness |
| Directional | Sun-like parallel rays |
| Point | Local light source |
| Spot | Focused beam |

### Presets

- **Studio:** Even lighting for analysis
- **Outdoor:** Sun-like harsh shadows
- **Soft:** Diffuse, minimal shadows
- **Custom:** User-defined

## Multi-View

### Split Screen

Display multiple views simultaneously:
- **Vertical split:** 2 views side-by-side
- **Quad view:** 4 views (top, side, front, 3D)
- **Custom:** Configure any layout

### Synced Views

- All views follow same simulation time
- Independent camera controls per view
- Linked zoom (optional)

## Troubleshooting

### Black Screen

- Check OpenGL version (3.3+ required)
- Update graphics drivers
- Try lower quality setting

### Slow Rendering

- Enable "Render every N frames"
- Reduce trail length
- Disable shadows
- Lower resolution

### Flickering

- Enable V-Sync
- Check for depth buffer issues
- Reduce transparency effects

---

*See also: [Full User Manual](../USER_MANUAL.md) | [Simulation Controls](simulation_controls.md) | [Analysis Tools](analysis_tools.md)*
