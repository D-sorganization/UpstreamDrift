"""CLI Tool Adapters for AI Agent Integration.

This module provides adapters for integrating external CLI tools like
Claude Code CLI and Codex CLI as callable tools for the AI agent.

Example:
    >>> from src.shared.python.ai.tools.cli_tools import ClaudeCodeTool
    >>> tool = ClaudeCodeTool()
    >>> result = tool.execute("Review the golf swing model")
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CLIExecutionResult:
    """Result of a CLI tool execution.

    Attributes:
        success: Whether the execution succeeded.
        output: Standard output from the command.
        error: Standard error from the command.
        return_code: Exit code of the process.
        command: The command that was executed.
    """

    success: bool
    output: str = ""
    error: str = ""
    return_code: int = 0
    command: str = ""


class CLIToolBase:
    """Base class for CLI tool adapters."""

    def __init__(self, command: str, working_dir: Path | None = None) -> None:
        """Initialize CLI tool adapter.

        Args:
            command: The base command to execute (e.g., 'claude', 'codex').
            working_dir: Working directory for command execution.
        """
        self._command = command
        self._working_dir = working_dir or Path.cwd()

    def _execute_command(
        self, args: list[str], input_text: str | None = None, timeout: int = 300
    ) -> CLIExecutionResult:
        """Execute a command with the CLI tool.

        Args:
            args: Command arguments.
            input_text: Optional stdin input.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with execution outcome.
        """
        full_command = [self._command] + args

        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                cwd=self._working_dir,
                input=input_text,
                timeout=timeout,
            )
            return CLIExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                return_code=result.returncode,
                command=" ".join(full_command),
            )
        except subprocess.TimeoutExpired:
            return CLIExecutionResult(
                success=False,
                error=f"Command timed out after {timeout} seconds",
                return_code=-1,
                command=" ".join(full_command),
            )
        except FileNotFoundError:
            return CLIExecutionResult(
                success=False,
                error=f"Command '{self._command}' not found. Please install it.",
                return_code=-1,
                command=" ".join(full_command),
            )
        except Exception as e:
            return CLIExecutionResult(
                success=False,
                error=str(e),
                return_code=-1,
                command=" ".join(full_command),
            )

    def is_available(self) -> bool:
        """Check if the CLI tool is available.

        Returns:
            True if the command is found, False otherwise.
        """
        try:
            subprocess.run(
                [self._command, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_version(self) -> str | None:
        """Get the version of the CLI tool.

        Returns:
            Version string or None if unavailable.
        """
        try:
            result = subprocess.run(
                [self._command, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or result.stderr.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None


class ClaudeCodeTool(CLIToolBase):
    """Adapter for Anthropic's Claude Code CLI.

    Enables the AI agent to invoke Claude Code for code analysis,
    refactoring suggestions, and development assistance.

    Installation:
        npm install -g @anthropic-ai/claude-code

    Example:
        >>> tool = ClaudeCodeTool()
        >>> if tool.is_available():
        ...     result = tool.ask("Explain this code")
    """

    def __init__(self, working_dir: Path | None = None) -> None:
        """Initialize Claude Code tool.

        Args:
            working_dir: Working directory for command execution.
        """
        super().__init__("claude", working_dir)

    def ask(self, prompt: str, timeout: int = 300) -> CLIExecutionResult:
        """Ask Claude Code a question.

        Args:
            prompt: The question or request to process.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with Claude's response.
        """
        return self._execute_command(["-p", prompt], timeout=timeout)

    def execute_inline(self, command: str, timeout: int = 300) -> CLIExecutionResult:
        """Execute an inline command.

        Args:
            command: The command to execute.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with execution outcome.
        """
        return self._execute_command(["-c", command], timeout=timeout)

    def run_in_background(self, prompt: str) -> subprocess.Popen:
        """Run Claude Code in background mode.

        Args:
            prompt: Initial prompt for the session.

        Returns:
            Popen instance for the background process.

        Raises:
            FileNotFoundError: If claude command not found.
        """
        return subprocess.Popen(
            ["claude", "-p", prompt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self._working_dir,
        )


class CodexCLITool(CLIToolBase):
    """Adapter for OpenAI's Codex CLI.

    Enables the AI agent to invoke Codex for code generation,
    completion, and transformation tasks.

    Installation:
        pip install openai-cli
        # or
        npm install -g @openai/codex-cli

    Example:
        >>> tool = CodexCLITool()
        >>> if tool.is_available():
        ...     result = tool.generate("Create a function to sort models")
    """

    def __init__(self, working_dir: Path | None = None) -> None:
        """Initialize Codex CLI tool.

        Args:
            working_dir: Working directory for command execution.
        """
        super().__init__("codex", working_dir)

    def generate(
        self, prompt: str, language: str = "python", timeout: int = 300
    ) -> CLIExecutionResult:
        """Generate code from a prompt.

        Args:
            prompt: Description of code to generate.
            language: Target programming language.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with generated code.
        """
        args = ["-p", prompt, "-l", language]
        return self._execute_command(args, timeout=timeout)

    def complete(self, code_prefix: str, timeout: int = 60) -> CLIExecutionResult:
        """Complete a code snippet.

        Args:
            code_prefix: Code prefix to complete.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with completion.
        """
        return self._execute_command(
            ["complete"], input_text=code_prefix, timeout=timeout
        )

    def explain(self, code: str, timeout: int = 120) -> CLIExecutionResult:
        """Explain what code does.

        Args:
            code: Code to explain.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with explanation.
        """
        return self._execute_command(["explain"], input_text=code, timeout=timeout)


class ShellTool(CLIToolBase):
    """Adapter for secure shell command execution.

    Provides sandboxed shell command execution for the AI agent.
    Use with caution - implements allowlist-based command filtering.

    Example:
        >>> tool = ShellTool(allowed_commands=["ls", "pwd", "cat"])
        >>> result = tool.execute("ls -la")
    """

    def __init__(
        self,
        working_dir: Path | None = None,
        allowed_commands: list[str] | None = None,
    ) -> None:
        """Initialize shell tool.

        Args:
            working_dir: Working directory for command execution.
            allowed_commands: List of allowed command prefixes.
        """
        super().__init__("", working_dir)
        self._allowed_commands = allowed_commands or [
            "ls",
            "pwd",
            "cat",
            "head",
            "tail",
            "wc",
            "grep",
            "find",
            "du",
            "df",
        ]

    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command is allowed.

        Args:
            command: Command to check.

        Returns:
            True if allowed, False otherwise.
        """
        # Dangerous commands are never allowed
        dangerous = ["rm", "sudo", "chmod", "chown", "curl", "wget", "ssh"]
        for d in dangerous:
            if command.startswith(d + " ") or command == d:
                return False

        # Check against allowlist
        return any(command.startswith(allowed) for allowed in self._allowed_commands)

    def execute(self, command: str, timeout: int = 60) -> CLIExecutionResult:
        """Execute a shell command.

        Args:
            command: Command to execute.
            timeout: Timeout in seconds.

        Returns:
            CLIExecutionResult with execution outcome.
        """
        if not self._is_command_allowed(command):
            return CLIExecutionResult(
                success=False,
                error=f"Command not allowed: {command}",
                return_code=-1,
                command=command,
            )

        return self._execute_command(["-c", command], timeout=timeout)


@dataclass
class CLIToolConfig:
    """Configuration for CLI tools.

    Attributes:
        claude_enabled: Whether Claude Code CLI is enabled.
        codex_enabled: Whether Codex CLI is enabled.
        shell_enabled: Whether shell tool is enabled.
        shell_allowed_commands: Commands allowed for shell tool.
        default_timeout: Default timeout for CLI operations.
        working_dir: Working directory for CLI execution.
    """

    claude_enabled: bool = True
    codex_enabled: bool = False
    shell_enabled: bool = False
    shell_allowed_commands: list[str] = field(
        default_factory=lambda: [
            "ls",
            "pwd",
            "cat",
            "head",
            "tail",
            "wc",
            "grep",
            "find",
        ]
    )
    default_timeout: int = 300
    working_dir: Path | None = None


class CLIToolManager:
    """Manager for CLI tools integration.

    Provides unified access to all CLI tools and handles
    availability checking and tool registration.

    Example:
        >>> manager = CLIToolManager()
        >>> if manager.claude.is_available():
        ...     result = manager.claude.ask("Hello")
    """

    def __init__(self, config: CLIToolConfig | None = None) -> None:
        """Initialize CLI tool manager.

        Args:
            config: Configuration for CLI tools.
        """
        self._config = config or CLIToolConfig()
        working_dir = self._config.working_dir

        self._claude: ClaudeCodeTool | None = None
        self._codex: CodexCLITool | None = None
        self._shell: ShellTool | None = None

        if self._config.claude_enabled:
            self._claude = ClaudeCodeTool(working_dir)

        if self._config.codex_enabled:
            self._codex = CodexCLITool(working_dir)

        if self._config.shell_enabled:
            self._shell = ShellTool(
                working_dir,
                self._config.shell_allowed_commands,
            )

    @property
    def claude(self) -> ClaudeCodeTool | None:
        """Get Claude Code tool if available."""
        return self._claude

    @property
    def codex(self) -> CodexCLITool | None:
        """Get Codex CLI tool if available."""
        return self._codex

    @property
    def shell(self) -> ShellTool | None:
        """Get shell tool if available."""
        return self._shell

    def get_status(self) -> dict[str, Any]:
        """Get status of all CLI tools.

        Returns:
            Dictionary with tool availability status.
        """
        status: dict[str, dict[str, Any]] = {}

        if self._claude:
            status["claude"] = {
                "available": self._claude.is_available(),
                "version": self._claude.get_version(),
            }

        if self._codex:
            status["codex"] = {
                "available": self._codex.is_available(),
                "version": self._codex.get_version(),
            }

        if self._shell:
            status["shell"] = {
                "available": True,
                "allowed_commands": self._config.shell_allowed_commands,
            }

        return status


def create_cli_tools_for_registry() -> list[dict[str, Any]]:
    """Create tool definitions for the AI tool registry.

    Returns:
        List of tool definitions ready for registration.
    """
    manager = CLIToolManager()
    tools = []

    if manager.claude and manager.claude.is_available():
        tools.append(
            {
                "name": "claude_ask",
                "description": "Ask Claude Code CLI for code analysis or development assistance",
                "handler": manager.claude.ask,
                "parameters": [
                    {
                        "name": "prompt",
                        "type": "string",
                        "required": True,
                        "description": "Question or request for Claude",
                    }
                ],
            }
        )

    if manager.codex and manager.codex.is_available():
        tools.append(
            {
                "name": "codex_generate",
                "description": "Generate code using Codex CLI",
                "handler": manager.codex.generate,
                "parameters": [
                    {
                        "name": "prompt",
                        "type": "string",
                        "required": True,
                        "description": "Description of code to generate",
                    },
                    {
                        "name": "language",
                        "type": "string",
                        "required": False,
                        "description": "Target programming language",
                        "default": "python",
                    },
                ],
            }
        )

    if manager.shell:
        tools.append(
            {
                "name": "shell_execute",
                "description": "Execute a safe shell command (limited to allowed commands)",
                "handler": manager.shell.execute,
                "parameters": [
                    {
                        "name": "command",
                        "type": "string",
                        "required": True,
                        "description": "Shell command to execute",
                    }
                ],
            }
        )

    return tools
