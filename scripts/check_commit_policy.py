#!/usr/bin/env python3
"""
Commit Policy Enforcer.

This script validates that commits labeled as 'fix' or 'chore' do not exceed
specified size thresholds (files changed, lines of code changed).
This prevents large feature merges from being disguised as minor fixes.
"""

import argparse
import re
import subprocess
import sys


def get_commit_range(base_ref: str, head_ref: str) -> list[str]:
    """Get list of commit hashes between base_ref and head_ref."""
    try:
        # If base_ref is not provided or empty, just check HEAD
        if not base_ref:
            cmd = ["git", "rev-list", "--max-count=1", head_ref]
        else:
            cmd = ["git", "rev-list", f"{base_ref}..{head_ref}"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error getting commit range: {e}")
        sys.exit(1)


def get_commit_info(commit_hash: str) -> tuple[str, int, int]:
    """
    Get commit message and stats (files changed, total lines changed).

    Returns:
        Tuple containing (commit_message, files_changed, total_lines_changed)
    """
    try:
        # Get commit message
        msg_cmd = ["git", "show", "-s", "--format=%s", commit_hash]
        msg_result = subprocess.run(msg_cmd, capture_output=True, text=True, check=True)
        message = msg_result.stdout.strip()

        # Get stats
        # --numstat gives: insertions deletions filename
        stat_cmd = ["git", "show", "--numstat", "--format=", commit_hash]
        stat_result = subprocess.run(
            stat_cmd, capture_output=True, text=True, check=True
        )

        lines = stat_result.stdout.strip().splitlines()
        files_changed = len(lines)
        total_lines = 0

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    insertions = int(parts[0]) if parts[0] != "-" else 0
                    deletions = int(parts[1]) if parts[1] != "-" else 0
                    total_lines += insertions + deletions
                except ValueError:
                    continue

        return message, files_changed, total_lines

    except subprocess.CalledProcessError as e:
        print(f"Error inspecting commit {commit_hash}: {e}")
        sys.exit(1)


def check_commit(commit_hash: str) -> bool:
    """
    Check if a commit violates the policy.

    Policy:
    - If message starts with 'fix' or 'chore':
        - Max files changed: 100
        - Max lines changed: 1000

    Returns:
        True if policy passed, False if violated.
    """
    message, files_changed, total_lines = get_commit_info(commit_hash)

    # Check if message starts with fix or chore (case-insensitive for safety,
    # though conventional commits are lowercase)
    # Regex looks for "fix" or "chore" at the start, optionally followed by scope "(...)", then ":"
    pattern = r"^(fix|chore)(\(.*\))?:"
    match = re.match(pattern, message, re.IGNORECASE)

    if match:
        # It's a fix or chore commit
        file_limit = 100
        line_limit = 1000

        violation = False
        reasons = []

        if files_changed > file_limit:
            violation = True
            reasons.append(f"files changed {files_changed} > {file_limit}")

        if total_lines > line_limit:
            violation = True
            reasons.append(f"lines changed {total_lines} > {line_limit}")

        if violation:
            print(f"❌ Policy Violation in commit {commit_hash[:7]}:")
            print(f"   Message: {message}")
            print(f"   Type: {match.group(1)}")
            print(f"   Issues: {', '.join(reasons)}")
            print(
                "   Action: 'fix' and 'chore' commits must be small. Use 'feat' or split the commit."
            )
            return False

    print(
        f"✅ Commit {commit_hash[:7]} passed: {message} ({files_changed} files, {total_lines} lines)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Check commit size policy.")
    parser.add_argument("--base", help="Base ref (e.g., origin/main)")
    parser.add_argument("--head", default="HEAD", help="Head ref (default: HEAD)")

    args = parser.parse_args()

    commits = get_commit_range(args.base, args.head)

    if not commits:
        print("No commits to check.")
        return

    print(
        f"Checking {len(commits)} commits from {args.base or 'start'} to {args.head}..."
    )

    all_passed = True
    for commit in commits:
        if not check_commit(commit):
            all_passed = False

    if not all_passed:
        sys.exit(1)

    print("All commits passed policy checks.")


if __name__ == "__main__":
    main()
