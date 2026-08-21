import argparse
import sys

from src.shared.python.humanoid_character_builder.core.builder import CharacterBuilder
from src.shared.python.humanoid_character_builder.presets.loader import load_body_preset


def main():
    parser = argparse.ArgumentParser(
        description="Humanoid Character Builder CLI", prog="humanoid_character_builder"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # "build" command
    build_parser = subparsers.add_parser(
        "build", help="Build a character using a preset"
    )
    build_parser.add_argument(
        "--preset", type=str, required=True, help="Preset name (e.g., athletic)"
    )
    build_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (e.g., my_character.urdf)",
    )

    # "presets" command
    presets_parser = subparsers.add_parser("presets", help="Manage presets")
    presets_subparsers = presets_parser.add_subparsers(
        dest="presets_command", required=True
    )

    # "presets list"
    presets_list_parser = presets_subparsers.add_parser(
        "list", help="List available presets"
    )

    args = parser.parse_args()

    if args.command == "build":
        try:
            builder = CharacterBuilder()
            params = load_body_preset(args.preset)
            urdf_xml = builder.generate_urdf(params)

            with open(args.output, "w", encoding="utf-8") as f:
                f.write(urdf_xml)

            print(f"Successfully built character and saved to {args.output}")
        except Exception as e:
            print(f"Error building character: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "presets":
        if args.presets_command == "list":
            try:
                builder = CharacterBuilder()
                presets = builder.list_presets()
                print("Available presets:")
                for preset in presets:
                    print(f"  - {preset}")
            except Exception as e:
                print(f"Error listing presets: {e}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
