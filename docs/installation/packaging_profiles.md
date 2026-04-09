# Windows Packaging Profiles

UpstreamDrift supports three installer packaging profiles so the launcher can
scale from a local-only desktop install to a provider-first biomechanics suite
without maintaining separate installer scripts.

## Profiles

### `core`

- Discovery mode: `local-only`
- Installs the launcher executable only
- Bundles only the required local engine runtime path
- Best for lightweight workstations and validation kiosks

### `hybrid`

- Discovery mode: `hybrid`
- Installs the launcher and API executables
- Bundles available local engines and supports optional external providers
- Best default for developer and research workstations

### `full`

- Discovery mode: `provider-first`
- Installs the launcher and API executables
- Bundles available local engines and expects external providers to be present
- Best for fully provisioned biomechanics workstations

## Build Usage

```powershell
python installer/windows/build_installer.py --profile core
python installer/windows/build_installer.py --profile hybrid --provider-root ..\MuJoCo_Models
python installer/windows/build_installer.py --profile full --provider-root ..\MuJoCo_Models --provider-root ..\Drake_Models
```

The build script writes the selected profile and provider-root metadata into
`installer/windows/dist/installer_info.json` so CI and release workflows can
validate the packaged discovery mode.
