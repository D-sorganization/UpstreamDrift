"""Module execution entrypoint for python -m src.sim_run."""

import sys
from src.sim_run.cli import main

if __name__ == "__main__":
    sys.exit(main())
