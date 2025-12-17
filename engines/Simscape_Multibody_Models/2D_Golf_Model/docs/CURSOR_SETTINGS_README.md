# Cursor Settings Synchronization Guide

## Overview

This repository contains a comprehensive `cursor-settings.json` file that includes all the optimized Cursor settings for stall prevention, performance optimization, and development workflow enhancement. This file can be used to synchronize your Cursor settings across multiple computers.

## Files

- **`cursor-settings.json`** - Complete Cursor settings with all optimizations
- **`CURSOR_SETTINGS_README.md`** - This documentation file

## How to Use

### Method 1: Copy Settings to Your Cursor Settings File

1. **Open Cursor Settings**:
   - Press `Ctrl+,` (Windows/Linux) or `Cmd+,` (Mac)
   - Or go to `File > Preferences > Settings`

2. **Open Settings JSON**:
   - Click the "Open Settings (JSON)" icon in the top right corner
   - This opens your `settings.json` file

3. **Copy Settings**:
   - Copy the contents of `cursor-settings.json` from this repository
   - Paste them into your Cursor `settings.json` file
   - Save the file

### Method 2: Import Specific Sections

If you only want specific optimizations, you can copy individual sections:

#### Terminal Stall Prevention
```json
{
  "terminal.integrated.scrollback": 10000,
  "terminal.integrated.fastScrollSensitivity": 5,
  "terminal.integrated.smoothScrolling": false,
  "terminal.integrated.enableBracketedPaste": false,
  "terminal.integrated.rendererType": "auto",
  "terminal.integrated.gpuAcceleration": "auto"
}
```

#### Git Performance Settings
```json
{
  "git.pager": "less -FRX",
  "git.diff.ignoreTrimWhitespace": true,
  "git.log.maxEntries": 100,
  "git.terminalAuthentication": false
}
```

#### File Watching Limits
```json
{
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/node_modules/**": true,
    "**/.venv/**": true
  }
}
```

## What These Settings Do

### 🚀 **Performance & Stability**
- **Terminal Optimization**: Prevents stalls, improves responsiveness
- **File Watching Limits**: Reduces system resource usage
- **Memory Management**: Optimizes large file handling

### 🛡️ **Stall Prevention**
- **Git Operations**: Limits output, prevents hanging
- **Terminal Settings**: Optimized for Git Bash stability
- **Bracketed Paste**: Prevents `[200~` errors

### 🐍 **Python Development**
- **Linting**: Ruff, MyPy, and Black integration
- **Type Checking**: Strict mode enabled
- **Auto-formatting**: Format on save with quality checks

### 🔧 **Development Workflow**
- **Auto-save**: Prevents data loss during stalls
- **Git Integration**: Enhanced GitLens and Git features
- **Copilot Optimization**: Large context window support

## Computer-Specific Customizations

### Windows Paths
The settings include Windows-specific paths:
```json
"terminal.integrated.profiles.windows": {
  "Git Bash": {
    "path": "C:\\Program Files\\Git\\bin\\bash.exe"
  }
}
```

### Mac/Linux Users
Update the terminal profile paths:
```json
"terminal.integrated.profiles.windows": {
  "Git Bash": {
    "path": "/usr/bin/bash"
  }
}
```

## Keeping Settings Updated

### 1. **Regular Updates**
- Check this repository for updates to `cursor-settings.json`
- Pull the latest changes: `git pull origin main`
- Copy any new optimizations to your local settings

### 2. **Custom Settings**
- Add your personal preferences to the file
- Commit and push your customizations
- Share improvements with the team

### 3. **Version Control**
- This file is tracked in Git
- Changes are versioned and documented
- Easy rollback if issues occur

## Troubleshooting

### Settings Not Applied
1. **Restart Cursor** after changing settings
2. **Check JSON syntax** for any errors
3. **Verify file location** (should be in Cursor settings directory)

### Performance Issues
1. **Disable heavy features** temporarily
2. **Check system resources** (memory, CPU)
3. **Review file watching exclusions**

### Terminal Problems
1. **Verify Git Bash path** exists
2. **Check environment variables**
3. **Test with different terminal profiles**

## Recommended Workflow

### First Time Setup
1. **Clone this repository** to your new computer
2. **Copy cursor-settings.json** to your Cursor settings
3. **Restart Cursor** to apply changes
4. **Test terminal operations** to ensure stability

### Ongoing Maintenance
1. **Pull repository updates** regularly
2. **Review new settings** for relevance
3. **Test changes** in development environment
4. **Share improvements** with team

## Contributing

### Adding New Settings
1. **Test thoroughly** on your system
2. **Document purpose** and benefits
3. **Add to cursor-settings.json** with clear comments
4. **Update this README** with usage instructions

### Reporting Issues
1. **Describe the problem** clearly
2. **Include system details** (OS, Cursor version)
3. **Provide error messages** if available
4. **Suggest solutions** if possible

## File Structure

```
Project_Template/
├── cursor-settings.json          # Complete Cursor settings
├── CURSOR_SETTINGS_README.md     # This documentation
├── .cursor/                      # Cursor project settings
│   └── rules/                   # Development rules
│       └── .cursorrules.md      # Project-specific rules
└── README.md                     # Project overview
```

## Benefits

- **🚀 Consistent Performance** across all computers
- **🛡️ Stall Prevention** for reliable development
- **🐍 Optimized Python** development environment
- **🔧 Enhanced Workflow** with Git and Copilot
- **📱 Easy Synchronization** between devices
- **🔄 Version Controlled** settings management

## Support

For questions or issues with these settings:
1. **Check this README** for common solutions
2. **Review the settings file** for configuration details
3. **Create an issue** in this repository
4. **Contact the team** for complex problems

---

**Note**: These settings are optimized for scientific computing and development workflows. Adjust them based on your specific needs and system capabilities.

