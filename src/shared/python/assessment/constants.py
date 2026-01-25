"""Constants for repository assessment."""

CATEGORIES = {
    "A": "Code Structure",
    "B": "Documentation",
    "C": "Test Coverage",
    "D": "Error Handling",
    "E": "Performance",
    "F": "Security",
    "G": "Dependencies",
    "H": "CICD",
    "I": "Code Style",
    "J": "API Design",
    "K": "Data Handling",
    "L": "Logging",
    "M": "Configuration",
    "N": "Scalability",
    "O": "Maintainability",
}

GROUP_WEIGHTS = {
    "Code": 0.25,
    "Testing": 0.15,
    "Docs": 0.10,
    "Security": 0.15,
    "Perf": 0.15,
    "Ops": 0.10,
    "Design": 0.10,
}

GROUP_MAPPING = {
    "A": "Code",
    "D": "Code",
    "I": "Code",
    "O": "Code",
    "K": "Code",
    "L": "Code",
    "C": "Testing",
    "B": "Docs",
    "F": "Security",
    "G": "Security",
    "E": "Perf",
    "H": "Ops",
    "M": "Ops",
    "J": "Design",
    "N": "Design",
}
