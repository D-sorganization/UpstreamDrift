# Frankenstein Editor Architecture

## Overview

The Frankenstein Editor (`src/shared/python/model_generation/editor/`) is a text-based URDF editing system that allows direct manipulation of URDF XML content with validation, history tracking, and clipboard support.

## File Structure

```
src/shared/python/model_generation/editor/
├── __init__.py                    # Package exports
├── frankenstein_editor.py         # Main editor class
├── text_editor.py                 # Base text editor functionality
├── text_editor_diff_mixin.py      # Diff/mixin functionality
├── text_editor_history_mixin.py   # Undo/redo history
├── editor_clipboard.py            # Clipboard operations
├── editor_modifications.py        # Modification operations
├── editor_types.py                # Type definitions
├── _text_editor_models.py         # Internal data models
└── _text_editor_validation.py     # Validation logic
```

## Components

### Core Editor

| File                     | Purpose                                | Status         |
| ------------------------ | -------------------------------------- | -------------- |
| `frankenstein_editor.py` | Main editor class combining all mixins | ✅ Implemented |
| `text_editor.py`         | Base text editing with XML parsing     | ✅ Implemented |

### Mixins

| File                           | Purpose                         | Status         |
| ------------------------------ | ------------------------------- | -------------- |
| `text_editor_diff_mixin.py`    | Diff generation and application | ✅ Implemented |
| `text_editor_history_mixin.py` | Undo/redo history stack         | ✅ Implemented |

### Supporting Modules

| File                         | Purpose                      | Status         |
| ---------------------------- | ---------------------------- | -------------- |
| `editor_clipboard.py`        | Copy/paste/cut operations    | ✅ Implemented |
| `editor_modifications.py`    | URDF modification primitives | ✅ Implemented |
| `editor_types.py`            | Type aliases and enums       | ✅ Implemented |
| `_text_editor_models.py`     | Internal data models         | ✅ Implemented |
| `_text_editor_validation.py` | XML/URDF validation          | ✅ Implemented |

## Features

### Implemented

1. **Text-based editing** - Direct XML string manipulation
2. **XML validation** - Well-formedness checks via `xml.etree.ElementTree`
3. **History tracking** - Undo/redo stack for all operations
4. **Clipboard operations** - Copy, paste, cut with internal clipboard
5. **Diff generation** - Generate diffs between editor states
6. **URDF-specific operations**:
   - Add/remove joints
   - Modify joint limits
   - Add/remove links
   - Modify inertial properties
   - Add/remove collision geometries

### Stubbed / Partial

1. **Visual preview** - No GUI preview of changes
2. **Batch operations** - Limited support for multi-element operations
3. **Export formats** - URDF-only, no MJCF/SDF export

## API Usage

```python
from src.shared.python.model_generation.editor import FrankensteinEditor

# Create editor
editor = FrankensteinEditor(urdf_content)

# Modify joint limits
editor.modify_joint_limit("hip_flexion", lower=-0.5, upper=0.5)

# Undo last operation
editor.undo()

# Redo
editor.redo()

# Get modified URDF
modified_urdf = editor.get_content()
```

## Test Coverage

Current coverage targets:

| File                           | Coverage    |
| ------------------------------ | ----------- |
| `frankenstein_editor.py`       | Target: 70% |
| `text_editor.py`               | Target: 70% |
| `text_editor_diff_mixin.py`    | Target: 70% |
| `text_editor_history_mixin.py` | Target: 70% |
| `editor_clipboard.py`          | Target: 70% |
| `editor_modifications.py`      | Target: 70% |
| `editor_types.py`              | Target: 70% |
| `_text_editor_models.py`       | Target: 70% |
| `_text_editor_validation.py`   | Target: 70% |

## Demo Script

See `examples/editor_demo.py` for end-to-end usage examples.

## Related Issues

- #4544 - This audit issue
- #4542 - Cross-engine FK equivalence tests
