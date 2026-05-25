import re

with open("src/shared/python/logging_pkg/logging_config.py", "r") as f:
    content = f.read()

old_pattern = r'''_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)"
        r"("
        r"password|passwd|pwd"
        r"|api_key|apikey|api[-_]?secret"
        r"|secret_key|secret[-_]?token"
        r"|access_token|auth_token|bearer"
        r"|private_key"
        r")"
        r"[\s]*[=:]\s*['\"]?([^\s'\"]{1,})['\"]?"
    ),
]'''

new_pattern = r'''_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)"
        r"(['\"]?)"
        r"("
        r"password|passwd|pwd"
        r"|api_key|apikey|api[-_]?secret"
        r"|secret_key|secret[-_]?token"
        r"|access_token|auth_token|bearer"
        r"|private_key"
        r")"
        r"\1"
        r"([\s]*[=:]\s*)"
        r"(?:"
        r"(['\"])(.*?)\4"
        r"|"
        r"([^\s'\",]+)"
        r")"
    ),
]'''

content = content.replace(old_pattern, new_pattern)

old_redact = r'''def _redact_sensitive(text: str) -> str:
    """Replace sensitive values in *text* with a redaction placeholder."""
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(r"\1=***REDACTED***", text)
    return text'''

new_redact = r'''def _redact_sensitive(text: str) -> str:
    """Replace sensitive values in *text* with a redaction placeholder."""
    for pattern in _SENSITIVE_PATTERNS:

        def repl(m: re.Match[str]) -> str:
            quote = m.group(4) if m.group(4) else ""
            return f"{m.group(1)}{m.group(2)}{m.group(1)}{m.group(3)}{quote}***REDACTED***{quote}"

        text = pattern.sub(repl, text)
    return text'''

content = content.replace(old_redact, new_redact)

with open("src/shared/python/logging_pkg/logging_config.py", "w") as f:
    f.write(content)
