import re
from pathlib import Path


def process_file(filepath):
    content = filepath.read_text(encoding="utf-8")

    # Don't touch 3rd party OpenSim model files directly if they are just scripts
    if "opensim-models" in str(filepath):
        return False

    lines = content.split("\n")
    has_print = False

    new_lines = []
    in_multiline_string = False

    # Very simple line by line processing
    for line in lines:
        stripped = line.strip()

        # Check multiline strings
        if stripped.count('"""') % 2 != 0 or stripped.count("'''") % 2 != 0:
            in_multiline_string = not in_multiline_string

        if (
            not in_multiline_string
            and not stripped.startswith("#")
            and re.search(r"(?<!\w)print\s*\(", line)
        ):
            # Replace print( -> logger.info(
            # Using negative lookbehind to not match inside another word
            line = re.sub(r"(?<!\w)print\s*\(", "logger.info(", line)
            has_print = True

        new_lines.append(line)

    if has_print:
        # Need to add logger imports if not present
        result = "\n".join(new_lines)
        if "import logging" not in result:
            # Find place to put import logging
            import_line = "import logging\nlogger = logging.getLogger(__name__)\n"

            # Put after last future import or docstring, for simplicity just append at top after docstring
            if result.startswith('"""'):
                end_doc = result.find('"""', 3)
                if end_doc != -1:
                    result = (
                        result[: end_doc + 3]
                        + "\n\n"
                        + import_line
                        + result[end_doc + 3 :]
                    )
            else:
                result = import_line + "\n" + result
        elif "logger = logging.getLogger" not in result:
            # Add logger creation
            result = result.replace(
                "import logging", "import logging\nlogger = logging.getLogger(__name__)"
            )

        filepath.write_text(result, encoding="utf-8")
        return True
    return False


src_dir = Path("src")
count = 0
for p in src_dir.rglob("*.py"):
    try:
        if process_file(p):
            count += 1
    except Exception:
        pass
