import os
import sys
import xml.etree.ElementTree as ET

THRESHOLDS = {
    "src/api/routers": 85.0,
    "src/api/services": 85.0,
    "src/shared/python/engine_core": 85.0,
    "src/engines/physics_engines/mujoco/python": 85.0,  # engine core
    "src/shared/python/tasks": 80.0,
    "src/shared/python/utils": 70.0,
}


def parse_coverage(xml_path):
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        sys.exit(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    packages = {}
    for pkg in root.findall(".//package"):
        name = pkg.get("name").replace(".", "/")
        line_rate = float(pkg.get("line-rate", 0)) * 100
        packages[name] = line_rate

    return packages


def main():
    xml_path = "coverage.xml"
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]

    packages = parse_coverage(xml_path)

    failed = False
    print("--- Coverage Enforcer ---")

    for target_dir, target_thresh in THRESHOLDS.items():
        normalized_target = target_dir.replace("/", ".")
        if normalized_target.startswith("src."):
            short_target = normalized_target[4:]
        else:
            short_target = normalized_target

        matched_pkgs = []
        for pkg_name in packages.keys():
            pkg_norm = pkg_name.replace("/", ".")
            if pkg_norm.startswith((normalized_target, short_target)):
                matched_pkgs.append(pkg_norm)

        if not matched_pkgs:
            continue

        for pkg in matched_pkgs:
            rate = packages[pkg.replace(".", "/")]
            if rate < target_thresh:
                print(
                    f"❌ FAIL: {pkg} coverage is {rate:.1f}% (Required: {target_thresh}%)"
                )
                failed = True
            else:
                print(
                    f"✅ PASS: {pkg} coverage is {rate:.1f}% (Required: {target_thresh}%)"
                )

    if failed:
        print("\nCoverage enforcer failed. Please add tests to meet the thresholds.")
        sys.exit(1)
    else:
        print("\nCoverage enforcer passed all thresholds.")
        sys.exit(0)


if __name__ == "__main__":
    main()
