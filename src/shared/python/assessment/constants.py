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

PRAGMATIC_PRINCIPLES = {
    "DRY": {
        "name": "Don't Repeat Yourself",
        "description": "Every piece of knowledge must have a single, unambiguous representation",
        "weight": 2.0,
    },
    "ORTHOGONALITY": {
        "name": "Orthogonality & Decoupling",
        "description": "Eliminate effects between unrelated things",
        "weight": 1.5,
    },
    "REVERSIBILITY": {
        "name": "Reversibility & Flexibility",
        "description": "Make decisions reversible; avoid painting yourself into a corner",
        "weight": 1.0,
    },
    "QUALITY": {
        "name": "Code Quality & Craftsmanship",
        "description": "Good enough software; know when to stop",
        "weight": 1.5,
    },
    "ROBUSTNESS": {
        "name": "Error Handling & Robustness",
        "description": "Crash early; use assertions; handle errors gracefully",
        "weight": 2.0,
    },
    "TESTING": {
        "name": "Testing & Validation",
        "description": "Test early, test often, test automatically",
        "weight": 2.0,
    },
    "DOCUMENTATION": {
        "name": "Documentation & Communication",
        "description": "It's all writing; document the why, not just the what",
        "weight": 1.0,
    },
    "AUTOMATION": {
        "name": "Automation & Tooling",
        "description": "Don't use manual procedures; automate everything",
        "weight": 1.5,
    },
}
