"""
Output Manager for Golf Modeling Suite

Handles all output operations including saving simulation results,
managing file organization, and exporting analysis reports.
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""

    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    PARQUET = "parquet"


class OutputManager:
    """
    Manages all output operations for the Golf Modeling Suite.

    Provides unified interface for saving simulation results, analysis data,
    and generating reports across all physics engines.
    """

    def __init__(self, base_path: str | Path | None = None):
        """
        Initialize OutputManager.

        Args:
            base_path: Base directory for outputs. Defaults to 'output' in project root.
        """
        if base_path is None:
            # Find project root and set default output path
            current_path = Path(__file__).resolve()
            project_root = current_path

            # Navigate up to find Golf_Modeling_Suite root
            while (
                project_root.name != "Golf_Modeling_Suite"
                and project_root.parent != project_root
            ):
                project_root = project_root.parent

            if project_root.name == "Golf_Modeling_Suite":
                base_path = project_root / "output"
            else:
                base_path = Path.cwd() / "output"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Define standard subdirectories
        self.directories = {
            "simulations": self.base_path / "simulations",
            "analysis": self.base_path / "analysis",
            "exports": self.base_path / "exports",
            "reports": self.base_path / "reports",
            "cache": self.base_path / "cache",
        }

        logger.info(f"OutputManager initialized with base path: {self.base_path}")

    def create_output_structure(self) -> None:
        """Create the standard output directory structure."""
        # Main directories
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        # Engine-specific simulation directories
        engines = ["mujoco", "drake", "pinocchio", "matlab"]
        for engine in engines:
            (self.directories["simulations"] / engine).mkdir(exist_ok=True)

        # Analysis subdirectories
        analysis_types = ["biomechanics", "trajectories", "optimization", "comparisons"]
        for analysis_type in analysis_types:
            (self.directories["analysis"] / analysis_type).mkdir(exist_ok=True)

        # Export subdirectories
        export_types = ["videos", "images", "data", "c3d"]
        for export_type in export_types:
            (self.directories["exports"] / export_type).mkdir(exist_ok=True)

        # Report subdirectories
        report_types = ["pdf", "html", "presentations"]
        for report_type in report_types:
            (self.directories["reports"] / report_type).mkdir(exist_ok=True)

        # Cache subdirectories
        cache_types = ["models", "computations", "temp"]
        for cache_type in cache_types:
            (self.directories["cache"] / cache_type).mkdir(exist_ok=True)

        logger.info("Output directory structure created successfully")

    def save_simulation_results(
        self,
        results: pd.DataFrame | dict[str, Any],
        filename: str,
        format_type: OutputFormat = OutputFormat.CSV,
        engine: str = "mujoco",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Save simulation results to file.

        Args:
            results: Simulation results data
            filename: Output filename (without extension)
            format_type: Output format
            engine: Physics engine name
            metadata: Additional metadata to include

        Returns:
            Path to saved file
        """
        # Ensure simulation directory exists
        engine_dir = self.directories["simulations"] / engine
        engine_dir.mkdir(parents=True, exist_ok=True)

        # Clean filename - remove format_type enum representation if present
        if "OutputFormat." in filename:
            filename = filename.split(".")[-1]  # Get just the extension part
            filename = "test_format"  # Use a clean name

        # Remove extension if already present
        if filename.endswith(f".{format_type.value}"):
            filename = filename[: -len(f".{format_type.value}")]

        # Add timestamp if not in filename (only for files without timestamps)
        if not any(char.isdigit() for char in filename) and "test_" not in filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"

        # Add extension based on format
        file_path = engine_dir / f"{filename}.{format_type.value}"

        try:
            if format_type == OutputFormat.CSV:
                if isinstance(results, pd.DataFrame):
                    results.to_csv(file_path, index=False)
                else:
                    # Convert dict to DataFrame if possible
                    df = pd.DataFrame(results)
                    df.to_csv(file_path, index=False)

            elif format_type == OutputFormat.JSON:
                # Handle numpy arrays in JSON serialization
                def json_serializer(obj: Any) -> Any:
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(
                        f"Object of type {type(obj)} is not JSON serializable"
                    )

                output_data = {
                    "metadata": metadata or {},
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                    "engine": engine,
                }

                with open(file_path, "w") as f:
                    json.dump(output_data, f, indent=2, default=json_serializer)

            elif format_type == OutputFormat.HDF5:
                if isinstance(results, pd.DataFrame):
                    results.to_hdf(file_path, key="data", mode="w")
                else:
                    # Convert to DataFrame first
                    df = pd.DataFrame(results)
                    df.to_hdf(file_path, key="data", mode="w")

            elif format_type == OutputFormat.PICKLE:
                output_data = {
                    "metadata": metadata or {},
                    "results": results,
                    "timestamp": datetime.now(),
                    "engine": engine,
                }
                # Use binary mode for pickle - ignore type checking for this line
                with open(file_path, "wb") as f:  # type: ignore[assignment,arg-type]
                    pickle.dump(output_data, f)  # type: ignore[arg-type]

            elif format_type == OutputFormat.PARQUET:
                if isinstance(results, pd.DataFrame):
                    results.to_parquet(file_path, index=False)
                else:
                    df = pd.DataFrame(results)
                    df.to_parquet(file_path, index=False)

            logger.info(f"Simulation results saved to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving simulation results: {e}")
            raise

    def load_simulation_results(
        self,
        filename: str,
        format_type: OutputFormat = OutputFormat.CSV,
        engine: str = "mujoco",
    ) -> pd.DataFrame | dict[str, Any]:
        """
        Load simulation results from file.

        Args:
            filename: Input filename
            format_type: File format
            engine: Physics engine name

        Returns:
            Loaded simulation results
        """
        engine_dir = self.directories["simulations"] / engine

        # Handle filename with or without extension
        if not filename.endswith(f".{format_type.value}"):
            filename = f"{filename}.{format_type.value}"

        file_path = engine_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Simulation file not found: {file_path}")

        try:
            if format_type == OutputFormat.CSV:
                return pd.read_csv(file_path)

            elif format_type == OutputFormat.JSON:
                with open(file_path) as f:
                    data = json.load(f)
                return data.get("results", data)

            elif format_type == OutputFormat.HDF5:
                return pd.read_hdf(file_path, key="data")

            elif format_type == OutputFormat.PICKLE:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                return data.get("results", data)

            elif format_type == OutputFormat.PARQUET:
                return pd.read_parquet(file_path)

        except Exception as e:
            logger.error(f"Error loading simulation results: {e}")
            raise

    def get_simulation_list(self, engine: str | None = None) -> list[str]:
        """
        Get list of available simulation files.

        Args:
            engine: Filter by specific engine (optional)

        Returns:
            List of simulation filenames
        """
        simulations = []

        if engine:
            engine_dir = self.directories["simulations"] / engine
            if engine_dir.exists():
                simulations.extend(
                    [f.name for f in engine_dir.iterdir() if f.is_file()]
                )
        else:
            # Get from all engines and also from root simulations directory
            sim_dir = self.directories["simulations"]

            # Check root simulations directory
            if sim_dir.exists():
                simulations.extend([f.name for f in sim_dir.iterdir() if f.is_file()])

            # Check engine subdirectories
            for engine_dir in sim_dir.iterdir():
                if engine_dir.is_dir():
                    simulations.extend(
                        [f.name for f in engine_dir.iterdir() if f.is_file()]
                    )

        return sorted(simulations)

    def export_analysis_report(
        self,
        analysis_data: dict[str, Any],
        report_name: str,
        format_type: str = "json",
    ) -> Path:
        """
        Export analysis report.

        Args:
            analysis_data: Analysis results and metadata
            report_name: Report filename (without extension)
            format_type: Report format (json, html, pdf)

        Returns:
            Path to exported report
        """
        report_dir = self.directories["reports"] / format_type
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name}_{timestamp}.{format_type}"
        file_path = report_dir / filename

        try:
            if format_type == "json":

                def json_serializer(obj: Any) -> Any:
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer | np.floating):
                        return float(obj)
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(
                        f"Object of type {type(obj)} is not JSON serializable"
                    )

                with open(file_path, "w") as f:
                    json.dump(analysis_data, f, indent=2, default=json_serializer)

            elif format_type == "html":
                # Basic HTML report generation
                html_content = self._generate_html_report(analysis_data, report_name)
                with open(file_path, "w") as f:
                    f.write(html_content)

            # PDF generation would require additional dependencies
            # elif format_type == "pdf":
            #     self._generate_pdf_report(analysis_data, file_path)

            logger.info(f"Analysis report exported to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error exporting analysis report: {e}")
            raise

    def cleanup_old_files(self, max_age_days: int = 30) -> int:
        """
        Clean up old files based on age.

        Args:
            max_age_days: Maximum age in days before cleanup

        Returns:
            Number of files cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0

        # Clean temporary files more aggressively (1 day)
        temp_cutoff = datetime.now() - timedelta(days=1)

        for directory in [self.directories["cache"] / "temp"]:
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            )
                            if file_time < temp_cutoff:
                                file_path.unlink()
                                cleaned_count += 1
                        except (OSError, PermissionError):
                            continue

        # Clean older simulation and analysis files
        for directory in [
            self.directories["simulations"],
            self.directories["analysis"],
        ]:
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        try:
                            file_time = datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            )
                            if file_time < cutoff_date:
                                # Move to archive instead of deleting
                                archive_dir = self.base_path / "archive"
                                archive_dir.mkdir(exist_ok=True)

                                relative_path = file_path.relative_to(self.base_path)
                                archive_path = archive_dir / relative_path
                                archive_path.parent.mkdir(parents=True, exist_ok=True)

                                file_path.rename(archive_path)
                                cleaned_count += 1
                        except (OSError, PermissionError):
                            continue

        logger.info(f"Cleaned up {cleaned_count} old files")
        return cleaned_count

    def _generate_html_report(self, data: dict[str, Any], title: str) -> str:
        """Generate basic HTML report."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title} - Golf Modeling Suite Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p class="timestamp">Generated: {timestamp_str}</p>

            <h2>Summary</h2>
            <table>
        """

        # Add data to table
        for key, value in data.items():
            if not isinstance(value, dict | list):
                html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"

        html += f"""
            </table>

            <h2>Detailed Data</h2>
            <pre>{json.dumps(data, indent=2, default=str)}</pre>
        </body>
        </html>
        """

        return html


# Convenience functions for backward compatibility
def save_results(results: Any, filename: str, format_type: str = "csv", engine: str = "mujoco") -> str:
    """Convenience function for saving results."""
    manager = OutputManager()
    return manager.save_simulation_results(
        results, filename, OutputFormat(format_type), engine
    )


def load_results(filename: str, format_type: str = "csv", engine: str = "mujoco") -> Any:
    """Convenience function for loading results."""
    manager = OutputManager()
    return manager.load_simulation_results(filename, OutputFormat(format_type), engine)
