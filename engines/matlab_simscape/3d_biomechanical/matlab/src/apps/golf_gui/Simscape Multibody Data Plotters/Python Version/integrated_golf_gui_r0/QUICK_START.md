# Quick Start Guide - Wiffle_ProV1 Golf Swing Visualizer

## 🚀 Immediate Setup (5 minutes)

### 1. Install Dependencies
```bash
cd integrated_golf_gui_r0
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python golf_wiffle_main.py
```

### 3. Load Your Data
- The app will automatically try to load the Excel file
- Or use **File → Load Excel File** to select your data
- Supports: `Wiffle_ProV1_club_3D_data.xlsx`

## 🎯 What You Get

### **Professional 3D Visualization**
- Real-time motion capture display
- Interactive camera controls
- High-performance OpenGL rendering
- Professional PyQt6 interface

### **Ball Comparison Analysis**
- **ProV1 vs Wiffle** side-by-side comparison
- Real-time metrics calculation
- Trajectory difference analysis
- Export capabilities

### **Advanced Controls**
- **Playback**: Play/pause, frame-by-frame, speed control
- **Camera**: Rotate, zoom, reset view
- **Analysis**: Toggle visibility, calculate metrics
- **Export**: Screenshot, data export, video recording

## 📊 Data Format Support

### **Excel Sheets Supported**
- `TW_ProV1` - Top Wood ProV1 data
- `TW_wiffle` - Top Wood Wiffle data
- `GW_ProV11` - Ground Wood ProV1 data
- `GW_wiffle` - Ground Wood Wiffle data

### **Automatic Processing**
- ✅ Noise filtering (Savitzky-Golay)
- ✅ Missing value interpolation
- ✅ Time normalization
- ✅ Data validation

## 🎮 Key Controls

### **Mouse Controls**
- **Left Click + Drag**: Rotate camera
- **Mouse Wheel**: Zoom in/out
- **Right Click**: Context menu

### **Keyboard Shortcuts**
- **Space**: Play/pause
- **Left/Right Arrows**: Previous/next frame
- **R**: Reset camera
- **Ctrl+O**: Load file
- **Ctrl+Q**: Quit

### **Panel Controls**
- **Ball Type**: Switch between ProV1/Wiffle/Difference
- **Show/Hide**: Toggle visibility of elements
- **Processing**: Adjust filtering and normalization
- **Export**: Save comparison data

## 🔧 Troubleshooting

### **If the app doesn't start:**
```bash
# Check Python version (3.8+ required)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check OpenGL support
python -c "import moderngl; print('OpenGL OK')"
```

### **If data doesn't load:**
- Verify Excel file exists in `Matlab Inverse Dynamics/`
- Check sheet names match expected format
- Ensure file isn't corrupted or password-protected

### **If visualization is slow:**
- Reduce quality settings in visualization panel
- Close other applications
- Update graphics drivers

## 📈 Performance Tips

### **For Large Datasets**
- Use noise filtering to reduce data points
- Enable performance optimizations
- Close unnecessary applications

### **For Best Quality**
- Enable antialiasing
- Use high-resolution rendering
- Enable shadows and lighting

## 🎯 Next Steps

### **For Research Use**
1. Load your specific Excel data
2. Adjust processing parameters
3. Export comparison metrics
4. Generate publication-ready visualizations

### **For Development**
1. Review the code structure
2. Modify data processing as needed
3. Add custom analysis features
4. Extend visualization capabilities

## 📞 Support

### **Quick Help**
- Check the **README_WIFFLE.md** for detailed documentation
- Review console output for error messages
- Test with sample data first

### **Common Issues**
- **OpenGL errors**: Update graphics drivers
- **Missing dependencies**: Reinstall requirements
- **Data format issues**: Check Excel file structure
- **Performance problems**: Reduce quality settings

---

**Ready to visualize your golf swing data! 🏌️‍♂️**

*This is the most refined version with full Excel support, professional UI, and advanced analysis capabilities.*
