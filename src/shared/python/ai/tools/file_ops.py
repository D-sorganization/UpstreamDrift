"""File system operations for AI agents (Read-Only)."""

from pathlib import Path

from src.shared.python.ai.tool_registry import ToolCategory, ToolRegistry


def register_file_tools(registry: ToolRegistry) -> None:
    """Register file system tools."""

    @registry.register(
        name="read_file",
        description="Read the contents of a file.",
        category=ToolCategory.CONFIGURATION,
        expertise_level=1,
    )
    def read_file(file_path: str) -> str:
        """Read a file from disk.

        Args:
            file_path: Absolute path to the file.
        """
        path = Path(file_path)
        if not path.is_absolute():
            # Security: Prevent relative path traversal if not careful,
            # though user is local. Better to assume relative to cwd if not absolute?
            # For now, require absolute or resolve relative to CWD.
            path = Path.cwd() / file_path

        if not path.exists():
            return f"Error: File not found: {path}"

        if not path.is_file():
            return f"Error: Not a file: {path}"

        # Basic text file check
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            return "Error: File appears to be binary."
        except Exception as e:
            return f"Error reading file: {e}"

    @registry.register(
        name="list_directory",
        description="List files and directories in a path.",
        category=ToolCategory.CONFIGURATION,
        expertise_level=1,
    )
    def list_directory(directory_path: str = ".") -> str:
        """List contents of a directory.

        Args:
            directory_path: Path to listing.
        """
        path = Path(directory_path)
        if not path.is_absolute():
            path = Path.cwd() / directory_path

        if not path.exists():
            return f"Error: Path not found: {path}"

        if not path.is_dir():
            return f"Error: Not a directory: {path}"

        try:
            items = []
            for item in path.iterdir():
                type_str = "<DIR>" if item.is_dir() else "<FILE>"
                items.append(f"{type_str:6} {item.name}")
            return "\n".join(sorted(items))
        except Exception as e:
            return f"Error listing directory: {e}"
