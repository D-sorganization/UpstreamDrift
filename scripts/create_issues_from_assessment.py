#!/usr/bin/env python3
"""
Create GitHub issues from assessment findings using shared utilities.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.script_utils import run_main, setup_script_logging

logger = setup_script_logging(__name__)

CATEGORY_MAP = {
    "architecture": "Architecture",
    "implementation": "Architecture",
    "assessment_a": "Architecture",
    "quality": "Code Quality",
    "hygiene": "Code Quality",
    "assessment_b": "Code Quality",
    "documentation": "Documentation",
    "assessment_c": "Documentation",
    "user": "User Experience",
    "ux": "User Experience",
    "assessment_d": "User Experience",
    "performance": "Performance",
    "assessment_e": "Performance",
    "installation": "Installation",
    "deployment": "Installation",
    "assessment_f": "Installation",
    "test": "Testing",
    "assessment_g": "Testing",
    "error": "Error Handling",
    "assessment_h": "Error Handling",
}


def classify_category(source: str, desc: str) -> str:
    """Classify issue category using keyword mapping."""
    combined = (source + " " + desc).lower()
    for key, cat in CATEGORY_MAP.items():
        if key in combined:
            return cat
    return "General"


def get_existing_issues() -> list[dict[str, Any]]:
    """Fetch existing GitHub issues via 'gh' CLI."""
    try:
        res = subprocess.run(
            ["gh", "issue", "list", "--limit", "200", "--json", "number,title,state,labels"],
            capture_output=True, text=True, check=True
        )
        return json.loads(res.stdout)
    except Exception as e:
        logger.warning(f"Could not fetch existing issues: {e}")
        return []


def create_issue(title: str, body: str, labels: list[str], dry_run: bool) -> bool:
    """Create a single GitHub issue."""
    if dry_run:
        logger.info(f"[DRY] Would create: {title}")
        return True
    try:
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]
        if labels:
            cmd.extend(["--label", ",".join(labels)])
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create issue: {e.stderr}")
        return False


def process_findings(path: Path, sevs: list[str], check_exist: bool, dry_run: bool) -> int:
    """Process findings from JSON and create corresponding issues."""
    if not path.exists():
        logger.error(f"Input file not found: {path}")
        return 1

    summary = json.loads(path.read_text())
    criticals = [i for i in summary.get("critical_issues", []) if i.get("severity") in sevs]
    
    existing = get_existing_issues() if check_exist else []
    
    for issue in criticals[:20]:
        severity = issue.get("severity", "CRITICAL")
        desc = issue.get("description", "")
        source = issue.get("source", "Unknown")
        cat = classify_category(source, desc)
        
        title = f"[GolfSuite] {severity} {cat}: {desc[:50]}..."
        if check_exist and any(title.lower() in x["title"].lower() for x in existing if x["state"] == "OPEN"):
            logger.info(f"Skipping existing: {title}")
            continue
            
        body = f"## Description\n\n**Severity**: {severity}\n**Category**: {cat}\n\n{desc}"
        labels = ["auto-generated", "bug" if severity in ("BLOCKER", "CRITICAL") else "enhancement"]
        
        create_issue(title, body, labels, dry_run)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Create GitHub issues from assessment")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--severity", default="BLOCKER,CRITICAL")
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    sevs = [s.strip().upper() for s in args.severity.split(",")]
    return process_findings(args.input, sevs, args.check_existing, args.dry_run)


if __name__ == "__main__":
    run_main(main, logger)
