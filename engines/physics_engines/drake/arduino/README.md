# Arduino Source Code

This directory contains Arduino sketches and libraries for embedded programming.

## Structure

```
arduino/
├── src/          # Main Arduino sketches (.ino files)
├── libraries/    # Custom Arduino libraries
└── examples/     # Example sketches
```

## Setup

### Arduino IDE

1. Install [Arduino IDE](https://www.arduino.cc/en/software)
2. Open Arduino IDE
3. Go to File → Preferences
4. Set "Sketchbook location" to this `arduino/` directory (or add it as an additional library location)

### PlatformIO (Recommended)

PlatformIO provides better dependency management and CI/CD support:

```bash
# Install PlatformIO
pip install platformio

# Initialize PlatformIO project
cd arduino
pio project init

# Build project
pio run

# Upload to board
pio run --target upload

# Run tests
pio test
```

## Project Structure

### Main Sketch

Place your main `.ino` file in `src/` directory:

```
arduino/
└── src/
    └── main.ino
```

### Custom Libraries

Place custom libraries in `libraries/` directory:

```
arduino/
└── libraries/
    └── MyCustomLibrary/
        ├── MyCustomLibrary.h
        ├── MyCustomLibrary.cpp
        └── examples/
```

### Examples

Place example sketches in `examples/` directory:

```
arduino/
└── examples/
    ├── blink_example/
    │   └── blink_example.ino
    └── sensor_example/
        └── sensor_example.ino
```

## Best Practices

1. **Use PlatformIO** for better dependency management
2. **Version control** - Use `.gitignore` to exclude build artifacts
3. **Documentation** - Include README.md in each library
4. **Testing** - Use PlatformIO's unit testing framework
5. **CI/CD** - Use PlatformIO in GitHub Actions for automated builds

## Configuration

### platformio.ini

Create `platformio.ini` in the `arduino/` directory:

```ini
[env:uno]
platform = atmelavr
board = uno
framework = arduino

[env:nano]
platform = atmelavr
board = nanoatmega328
framework = arduino

[env:esp32]
platform = espressif32
board = esp32dev
framework = arduino
```

## Resources

- [Arduino Reference](https://www.arduino.cc/reference/en/)
- [PlatformIO Documentation](https://docs.platformio.org/)
- [Arduino Libraries](https://www.arduino.cc/reference/en/libraries/)
