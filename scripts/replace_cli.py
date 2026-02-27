import os

files_to_update = [
    "install.sh",
    "docs/troubleshooting/installation.md",
    "src/shared/python/config/standard_models.py",
    "src/shared/python/launcher_factory.py",
    "docs/UPSTREAM_DRIFT_USER_MANUAL.md",
    "docs/USER_MANUAL.md",
    "docs/tutorials/content/01_getting_started.md",
    "output/README.md",
]

for file_path in files_to_update:
    if not os.path.exists(file_path):
        continue
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Replace CLI commands
    # Note: trying to avoid replacing things like URL domains if they exist (api.golf-suite.io)
    # But for now, we'll replace the command specifically.
    replacements = [
        ("golf-suite --help", "upstream-drift --help"),
        ("To start:   golf-suite", "To start:   upstream-drift"),
        ("conda activate golf-suite", "conda activate upstream-drift"),
        ("golf-suite --setup-models", "upstream-drift --setup-models"),
        ("'golf-suite' without --engine", "'upstream-drift' without --engine"),
        ("golf-suite --classic", "upstream-drift --classic"),
        ("golf-suite --api-only", "upstream-drift --api-only"),
        ("golf-suite --no-browser", "upstream-drift --no-browser"),
        ("golf-suite output", "upstream-drift output"),
        # General CLI references that are safe to replace:
        ("`golf-suite`", "`upstream-drift`"),
        (" golf-suite ", " upstream-drift "),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("CLI renaming complete.")
