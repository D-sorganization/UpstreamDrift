"""Read installed distributions without requiring pip in scientific runtimes."""

from importlib.metadata import distributions
import json
import sys

sys.stdout.write(
    json.dumps({d.metadata["Name"]: d.version for d in distributions()}, sort_keys=True)
)
