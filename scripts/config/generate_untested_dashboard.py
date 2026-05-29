import defusedxml.ElementTree as ET
import os


def generate_untested_dashboard(
    xml_path="coverage.xml",
    output_path="docs/development/untested_modules_dashboard.md",
):
    if not os.path.exists(xml_path):
        print(f"Coverage file not found at {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    untested_modules = []

    for package in root.findall(".//package"):
        for cls in package.findall(".//class"):
            filename = cls.get("filename")
            line_rate = float(cls.get("line-rate", 0))

            # Identify modules with absolutely no coverage
            if line_rate == 0.0:
                untested_modules.append(filename)

    # Sort for predictability
    untested_modules.sort()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3: Untested Modules Dashboard\n\n")
        f.write(
            "This dashboard tracks all core modules that currently possess **0% test coverage**. These modules form the primary target list for Phase 3 Coverage Ratcheting.\n\n"
        )

        f.write(f"**Total Untested Modules:** {len(untested_modules)}\n\n")

        f.write("## Hit List\n")
        f.write("| Module Path | Status |\n")
        f.write("|-------------|--------|\n")

        f.writelines(f"| `{mod}` | ❌ Untested |\n" for mod in untested_modules)

    print(
        f"Dashboard successfully generated at {output_path} with {len(untested_modules)} modules."
    )


if __name__ == "__main__":
    generate_untested_dashboard()
