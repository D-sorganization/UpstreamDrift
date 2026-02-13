#!/usr/bin/env python

import argparse
import re

_normals_pattern = re.compile(
    r'\s*<PointData Normals="Normals".+?</PointData>', flags=re.DOTALL
)


def remove_normals_node(vtp_content) -> Any:
    """Remove PointData Normals XML nodes from VTP content string."""
    return re.sub(_normals_pattern, "", vtp_content)


def remove_normals_from_vtp(vtp_path) -> None:
    """Strip normals data from a VTP file in place."""
    with open(vtp_path, newline="") as fd:
        vtp_content = fd.read()

    modified_vtp_content = remove_normals_node(vtp_content)

    if modified_vtp_content != vtp_content:
        with open(vtp_path, "w", newline="") as fd:
            fd.write(modified_vtp_content)


def main() -> None:
    """Parse arguments and remove normals from specified VTP files."""
    parser = argparse.ArgumentParser(
        description='removes <PointData Normals="Normals">...</PointData> in each given VTP file'
    )
    parser.add_argument("vtp_file", nargs="+", type=str)
    args = parser.parse_args()
    for vtp_file in args.vtp_file:
        remove_normals_from_vtp(vtp_file)


if __name__ == "__main__":
    main()
