import ast


def validate_polynomial_text(text: str) -> str | None:
    cleaned = text.replace(" ", "")
    if not cleaned:
        return None  # Empty is valid (treated as 0.0)
    try:
        tuple(float(c) for c in cleaned.split("+"))
    except ValueError:
        return "Invalid polynomial format. Use numbers separated by '+'."
    return None


def validate_torque_text(text: str) -> str | None:
    if not text.strip():
        return None  # Empty is valid (treated as 0.0)
    try:
        ast.parse(text, mode="eval")
    except SyntaxError:
        return "Invalid syntax."
    return None
