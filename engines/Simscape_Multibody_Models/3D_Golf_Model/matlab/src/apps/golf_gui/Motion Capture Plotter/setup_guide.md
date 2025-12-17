# Golf Swing Motion Capture Analyzer - Python Version

## Installation

1. **Install Python** (3.8 or higher recommended)
   - Download from https://python.org

2. **Install Required Libraries**
   ```bash
   pip install numpy pandas matplotlib scipy openpyxl
   ```

   Or use the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Files**
   - Save `golf_swing_analyzer.py` (the main script)
   - Place your `Wiffle_ProV1_club_3D_data.xlsx` file in the same directory

## Running the Application

```bash
python golf_swing_analyzer.py
```

## Features and Usage

### 1. **Data Loading**
- Click "Load Excel File" to select your motion capture data
- The application automatically processes all swing sheets (TW_wiffle, TW_ProV1, GW_wiffle, GW_ProV11)
- Key frame events (Address, Top, Impact, Finish) are automatically extracted

### 2. **Swing Selection**
- Use the dropdown to switch between different golfers and ball types
- Data updates automatically when selection changes

### 3. **Playback Controls**
- **Play/Pause**: Animate through the swing motion
- **Frame Slider**: Navigate to specific frames manually
- **Speed Control**: Adjust playback speed (0.1x to 3.0x)

### 4. **Camera Views**
- **Face-On**: View from directly in front of the golfer
- **Down-the-Line**: View from behind along the target line
- **Top-Down**: Overhead view of the swing plane
- **Isometric**: 3D perspective view

### 5. **Data Filtering**
- **None**: Raw data
- **Moving Average**: 5-point smoothing
- **Savitzky-Golay**: Polynomial smoothing (3rd order, 9-point)
- **Butterworth**: Low-pass filters at 6Hz, 8Hz, or 10Hz

### 6. **Analysis Options**
- **Evaluation Point Offset**: Adjust analysis point along shaft (-2" to +2")
- **Show Trajectory**: Display complete club path
- **Show Force Vectors**: Display calculated force and torque vectors

### 7. **Real-time Data Display**
Shows current frame information:
- Position coordinates (X, Y, Z)
- Velocity vectors
- Acceleration vectors
- Force calculations
- Torque calculations

## Understanding the Visualization

### Golf Club Representation
- **Black line**: Club shaft from mid-hands to clubface
- **Brown line**: Grip section
- **Gray outline**: Clubhead at impact position
- **Red dashed line**: Trajectory path (if enabled)

### Force Vectors
- **Red arrows**: Force vectors (scaled for visibility)
- **Blue arrows**: Torque vectors (scaled for visibility)

### Key Frame Markers
- **Green**: Address position
- **Yellow**: Top of backswing
- **Red**: Impact position
- **Blue**: Finish position

## Data Format Requirements

The Excel file should contain:
- **Sheet names**: TW_wiffle, TW_ProV1, GW_wiffle, GW_ProV11
- **Row 1**: Key frame markers (A, T, I, F) with frame numbers
- **Row 3+**: Motion data with columns:
  - Sample number, Time
  - X, Y, Z positions (in mm)
  - Xx, Xy, Xz (club X-axis unit vector)
  - Yx, Yy, Yz (club Y-axis unit vector)

## Technical Notes

### Coordinate System Conversion
- Input data: X (target line), Y (toward ball), Z (vertical)
- Visualization: X→X, Y→Z, Z→-Y (for proper 3D display)

### Dynamics Calculations
- **Force**: F = ma (mass × acceleration)
- **Torque**: Simplified calculation based on lever arms
- **Mass**: 0.2 kg (typical driver weight)
- **Shaft Length**: 1.2 m (typical driver length)

### Filtering Implementation
- **Central difference**: Used for velocity/acceleration calculations
- **Signal processing**: SciPy filters for noise reduction
- **Real-time**: All calculations update as you navigate frames

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all required libraries are installed
2. **File not found**: Place Excel file in same directory as script
3. **Slow performance**: Try reducing trajectory points or disabling vectors
4. **Memory issues**: Close other applications or restart Python

### Performance Tips
- Use filtering to smooth noisy data
- Disable force vectors for faster rendering
- Use smaller frame steps for large datasets

## Extensions and Modifications

The code is structured for easy modification:
- **Add new filters**: Extend the `apply_filter()` method
- **Custom analysis**: Modify `calculate_dynamics()` for advanced calculations
- **Export features**: Add CSV export functionality
- **Additional views**: Implement custom camera positions
- **Real-time plotting**: Add velocity/acceleration time series plots
