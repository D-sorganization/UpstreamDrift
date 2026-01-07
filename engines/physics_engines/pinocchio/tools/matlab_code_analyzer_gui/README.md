# MATLAB Code Analysis GUI

A user-friendly graphical interface for analyzing MATLAB code using the built-in Code Analyzer (MLint). This tool allows you to select files or folders interactively and configure analysis options through an intuitive GUI.

## Features

- **Interactive File/Folder Selection**: Use GUI dialogs to select individual files or entire folders for analysis
- **Flexible Configuration**: Configure analysis options including:
  - Recursive folder scanning
  - File extension filtering
  - Directory and file exclusion patterns
  - Error handling modes
- **Multiple Output Formats**: Export results to CSV, Excel, JSON, or Markdown
- **Progress Tracking**: Visual feedback during analysis
- **Results Summary**: Automatic summary of findings with issue type breakdown

## Files

- `exportCodeIssues.m` - Core analysis engine (enhanced version from restore-1956-columns branch)
- `codeIssuesGUI.m` - Interactive GUI interface
- `launchCodeAnalyzer.m` - Simple launcher function
- `README.md` - This documentation file

## Quick Start

### Method 1: Simple Launcher
```matlab
% Add the Code_Analysis_GUI folder to your MATLAB path
addpath('matlab/Scripts/Code_Analysis_GUI');

% Launch the GUI
launchCodeAnalyzer();
```

### Method 2: Direct GUI Call
```matlab
% For more control over options
addpath('matlab/Scripts/Code_Analysis_GUI');
results = codeIssuesGUI('DefaultPath', 'C:\MyProject', 'ShowProgress', true);
```

### Method 3: Command Line (Non-GUI)
```matlab
% For scripting/automation
addpath('matlab/Scripts/Code_Analysis_GUI');
results = exportCodeIssues('C:\MyProject\src', 'Output', 'issues.csv');
```

## Usage Examples

### Analyze a Single File
1. Run `launchCodeAnalyzer()`
2. Click "Add File(s)" and select your .m file
3. Configure options as needed
4. Click "Analyze"

### Analyze a Project Folder
1. Run `launchCodeAnalyzer()`
2. Click "Add Folder" and select your project directory
3. Enable "Recursive" to scan subfolders
4. Set exclude patterns (e.g., exclude directories: `.git, build, external`)
5. Click "Analyze"

### Batch Analysis with Custom Options
```matlab
results = codeIssuesGUI(...
    'DefaultPath', 'C:\MyProject', ...
    'IncludeExt', {'.m', '.mlx'}, ...
    'ExcludeDirs', {'.git', 'build', 'external'}, ...
    'AutoSave', true);
```

## Configuration Options

### Analysis Options
- **Recursive**: Scan subfolders when analyzing directories
- **File Extensions**: Comma-separated list of extensions to include (default: `.m`)
- **Exclude Directories**: Directories to skip (e.g., `.git, build, external`)
- **Exclude Files**: Wildcard patterns for files to skip (e.g., `*_autogen.m`)
- **Error Handling**: Choose to record errors as issues or stop on first error

### Output Options
- **Output File**: Specify output file path (optional)
- **Auto-save**: Automatically save with timestamp
- **Supported Formats**: .csv, .xlsx, .json, .md

## Results Table

The analysis returns a table with these columns:
- **File**: Absolute file path
- **RelFile**: Relative file path
- **Line**: Line number where issue occurs
- **Column**: Column number where issue occurs
- **Identifier**: Code Analyzer message identifier
- **Message**: Human-readable description of the issue

## Integration Notes

This tool is designed to be self-contained within the `Code_Analysis_GUI` folder. Simply add this folder to your MATLAB path to use the functionality.

The tool is compatible with MATLAB R2019b+ and handles version differences in the Code Analyzer output gracefully.

## See Also

- `checkcode` - MATLAB's built-in code analyzer
- `mlint` - Legacy name for code analyzer
- Code Analyzer documentation in MATLAB Help
