# Sidekick Standalone

Sidekick can run as a self-contained desktop application — no UpstreamDrift
launcher required. It ships as both a pip-installable console script and as
a one-file binary for macOS, Linux, and Windows.

See [ADR-0018](../adr/0018-standalone-sidekick.md) for design rationale.

---

## Install in 3 minutes

### Option A — pip (Python 3.10+)

```bash
pip install upstream-drift          # or: pip install upstream-drift[gui-tools]
sidekick --help
```

### Option B — pre-built binary (no Python required)

Download the binary for your platform from the latest
[`sidekick-v*` GitHub Release](https://github.com/D-sorganization/UpstreamDrift/releases).

| Platform | Binary         |
| -------- | -------------- |
| macOS    | `sidekick`     |
| Linux    | `sidekick`     |
| Windows  | `sidekick.exe` |

```bash
# macOS / Linux
chmod +x sidekick
./sidekick --help

# Windows
sidekick.exe --help
```

---

## Two layouts

Sidekick opens in one of two profiles, switchable in Preferences (⌘,):

| Profile      | Default focus                          | Use when …                                        |
| ------------ | -------------------------------------- | ------------------------------------------------- |
| `chat-first` | AI assistant panel fills the window    | You want conversational access to the calculators |
| `calc-first` | Calculator sidebar expanded by default | You want direct access to the process calculators |

Set your preferred profile on first run (the onboarding dialog appears
automatically) or via the Preferences dialog at any time.

---

## Headless calculations

The `sidekick run` sub-command runs any registered calculator headlessly and
emits JSON — useful for scripting, CI, and piping results to other tools.

```bash
# List available calculators
sidekick list

# Run the Water-Gas Shift reactor calculator
sidekick run --calculator wgs_reactor --inputs inputs.json

# Write results to a file instead of stdout
sidekick run --calculator wgs_reactor --inputs inputs.json --output results.json
```

### Example: `inputs.json` for `wgs_reactor`

```json
{
  "temperature_c": 350.0,
  "co_fraction": 0.3,
  "h2o_fraction": 0.4,
  "co2_fraction": 0.1,
  "h2_fraction": 0.2,
  "pressure_bar": 20.0
}
```

### Example output

```json
{
  "temperature_c": 350.0,
  "equilibrium_constant": 12.34,
  "extent_of_reaction": 0.18,
  "co_conversion_fraction": 0.6,
  "equilibrium_composition": {
    "co": 0.12,
    "h2o": 0.22,
    "co2": 0.28,
    "h2": 0.38
  }
}
```

---

## Command reference

```
sidekick [--version] [--skip-onboarding] <command>

Commands:
  gui   Launch the standalone window
  run   Run a calculator headlessly and emit JSON
  list  List available headless calculators
```

---

## Configuration

Preferences are stored in your platform config directory:

| Platform | Path                                                      |
| -------- | --------------------------------------------------------- |
| macOS    | `~/Library/Application Support/sidekick/preferences.json` |
| Linux    | `~/.config/sidekick/preferences.json`                     |
| Windows  | `%APPDATA%\sidekick\preferences.json`                     |

The `onboarded` sentinel file (same directory) prevents the first-run wizard
from re-appearing after you complete it.

---

## CI / automation

Pass `--skip-onboarding` to bypass the first-run dialog in non-interactive
environments:

```bash
sidekick --skip-onboarding run --calculator wgs_reactor --inputs inputs.json
```

---

## Next steps

- [Sidekick overview](README.md)
- [Adding agentic tools](agent.md)
- [ADR-0018: design rationale](../adr/0018-standalone-sidekick.md)
