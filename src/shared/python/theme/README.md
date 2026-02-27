# Fleet-Wide Theme System

Unified color theme system for all PyQt6 GUI applications across the
D-sorganization repository fleet.

## Features

- **13 built-in themes**: Light, Dark, Slate Gray, Ocean Blue, Forest Green,
  Monokai, Dracula, One Dark, Gitpod Dark, MS Word, MS Excel, Legal Pad,
  High Contrast
- **Custom themes** with QSettings persistence
- **Theme inheritance** for docked/embedded sub-applications
- **Qt stylesheet generation** (QSS) from color dictionaries
- **Matplotlib integration** for consistent plot colors
- **Signal-based notifications** (`themeChanged`) for live updates
- **REST API router** (FastAPI) for web-based theme CRUD
- **Dialog widgets**: color picker, custom theme editor, theme manager

## Quick Start

### One-liner integration (recommended)

```python
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from shared.python.theme.integration import setup_themed_app

app = QApplication(sys.argv)
window = QMainWindow()
setup_themed_app(app, window)       # applies saved theme + adds Theme menu
window.show()
sys.exit(app.exec())
```

### Mixin approach

```python
from PyQt6.QtWidgets import QMainWindow
from shared.python.theme.integration import ThemedWindowMixin

class MyWindow(ThemedWindowMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_theme_support()   # theme menu + auto-apply
```

### Direct ThemeManager usage

```python
from shared.python.theme import ThemeManager

manager = ThemeManager.instance()
manager.change_theme("Dracula")
colors = manager.get_current_colors()   # dict[str, str]
```

## API Reference

### ThemeManager (singleton)

| Method                            | Returns          | Description                       |
| --------------------------------- | ---------------- | --------------------------------- | ----------------------- |
| `ThemeManager.instance()`         | `ThemeManager`   | Get/create the singleton          |
| `get_available_themes()`          | `list[str]`      | All built-in + custom names       |
| `get_builtin_themes()`            | `list[str]`      | Built-in theme names only         |
| `get_current_theme_name()`        | `str`            | Active theme name                 |
| `get_current_colors()`            | `dict[str, str]` | Active theme color map            |
| `get_theme_colors(name)`          | `dict            | None`                             | Color map for any theme |
| `get_theme_stylesheet(name)`      | `str`            | QSS stylesheet for a theme        |
| `change_theme(name)`              | `None`           | Switch theme + emit signal        |
| `apply_theme()`                   | `None`           | Re-apply current theme to windows |
| `apply_theme_to_window(w)`        | `None`           | Apply to a specific window        |
| `save_custom_theme(name, colors)` | `str`            | Persist a custom theme            |
| `delete_custom_theme(name)`       | `bool`           | Remove a custom theme             |

### Color Constants

Each theme provides these 14 color keys:

| Key              | Purpose                     |
| ---------------- | --------------------------- |
| `bg`             | Main background             |
| `group_bg`       | Group box / card background |
| `border`         | Standard border             |
| `text`           | Primary text                |
| `text_secondary` | Secondary / muted text      |
| `label`          | Label text                  |
| `focus`          | Focus ring / highlight      |
| `input_bg`       | Input field background      |
| `accent`         | Primary accent              |
| `title_bg`       | Title / header background   |
| `title_border`   | Title / header border       |
| `table_header`   | Table header background     |
| `table_alt`      | Alternating table row       |
| `button_hover`   | Button hover state          |

### Integration Helpers

```python
from shared.python.theme.integration import (
    setup_themed_app,      # one-liner: theme + menu
    apply_theme_to_window, # apply theme to any window
    create_theme_menu,     # add Theme menu to menubar
    ThemedWindowMixin,     # mixin for QMainWindow subclasses
)
```

### Utility Functions

```python
from shared.python.theme.colors import (
    BUILTIN_THEMES,        # dict[str, dict[str, str]]
    CHART_COLORS,          # list[str] - 8 plot colors
    is_dark_theme,         # bool check by theme name
    get_rgba,              # hex -> (r, g, b, a) for matplotlib
    get_matplotlib_colors, # theme -> matplotlib rcParams dict
    normalise_hex_color,   # "#f00" -> "#ff0000"
    is_valid_hex_color,    # validate hex color strings
)
```

### FastAPI REST Router

```python
from shared.python.theme.api import create_theme_router

router = create_theme_router(theme_manager)
app.include_router(router, prefix="/api/v1/themes")
```

Endpoints: `GET /`, `GET /builtin`, `GET /custom`, `POST /custom`,
`DELETE /custom/{id}`, `GET /active`, `PUT /active`.

## Cross-Repo Vendoring

The theme system lives in `src/shared/python/theme/` and is vendored into
sibling repositories via their `vendor/ud-tools/` directory. Theme
definitions are loaded from `src/shared/theme-definitions/themes.json`
(canonical source) with hardcoded fallbacks in `colors.py`.

## Adding a New Theme

1. Edit `src/shared/theme-definitions/themes.json`
2. Add an entry with all 14 color keys
3. The theme is automatically available via `ThemeManager`
