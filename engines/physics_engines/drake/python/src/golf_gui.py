"""Entry point for Drake Golf GUI."""
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("golf_gui")

if __name__ == "__main__":
    logger.info("Starting Drake Golf GUI entry point...")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")

    try:
        if __package__:
            logger.info(f"Running as package: {__package__}")
            from . import drake_gui_app
        else:
            # Fallback for script execution, though likely to fail due to relative imports in drake_gui_app
            logger.warning("Running as script. Relative imports in drake_gui_app may fail.")
            logger.info("Suggest running with: python -m src.golf_gui")

            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))

            import drake_gui_app

        drake_gui_app.main()

    except ImportError as e:
        logger.error(f"Import Error: {e}")
        logger.error("Failed to load application modules.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
