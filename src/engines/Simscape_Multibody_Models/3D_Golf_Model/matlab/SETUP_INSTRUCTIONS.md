# MATLAB Environment Setup Instructions

This document explains how to fix the MATLAB/Simulink cache and path warnings you're experiencing.

## Problem Summary

You're seeing two types of warnings:

1. **Backup_Scripts directory warning**: MATLAB is looking for an old backup directory that no longer exists
2. **Simulink cache folder warnings**: Simulink is configured to use cache folders that don't exist

## Solution

### Quick Fix (Recommended)

Run the setup script in MATLAB:

```matlab
cd matlab
setup_matlab_environment
```

This script will:
- ✅ Create cache directories in `matlab/cache/simulink/`
- ✅ Configure Simulink to use these cache folders
- ✅ Remove old Backup_Scripts paths from MATLAB path
- ✅ Verify the configuration

### Manual Configuration

If you prefer to configure manually or the script doesn't work:

#### 1. Configure Simulink Cache Folders

In MATLAB, run:
```matlab
cd matlab
configure_simulink_cache
```

Or manually in MATLAB:
1. Go to **Home > Preferences > Simulink > Code Generation**
2. Set **Cache Folder** to: `C:\Users\diete\Repositories\Golf_Model\matlab\cache\simulink\cache`
3. Set **Code Generation Folder** to: `C:\Users\diete\Repositories\Golf_Model\matlab\cache\simulink\codegen`
4. Click **Apply** and **OK**

#### 2. Remove Backup_Scripts from MATLAB Path

1. Go to **Home > Environment > Set Path**
2. Search for "Backup_Scripts" in the path list
3. Select any entries containing "Backup_Scripts" and click **Remove**
4. Click **Save**

Or run the cleanup script:
```matlab
cd matlab/Scripts/Dataset Generator/utils
cleanup_matlab_path
```

## What Was Created

- **Cache directory**: `matlab/cache/simulink/` (excluded from git)
- **Configuration script**: `matlab/configure_simulink_cache.m`
- **Setup script**: `matlab/setup_matlab_environment.m`
- **Documentation**: `matlab/CACHE_SETUP.md`

## Verification

After running the setup, verify everything works:

```matlab
% Check Simulink cache settings
Simulink.fileGenControl('get')

% Check MATLAB path (should not contain Backup_Scripts)
path
```

## Notes

- The cache directories are automatically excluded from git via `.gitignore`
- Settings persist in your MATLAB preferences, so you only need to run this once
- If you move the repository, you may need to re-run the configuration
- The Backup_Scripts directory is no longer used - backups are now stored in `archive/backups/`
