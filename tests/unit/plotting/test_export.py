"""Tests for src.shared.python.plotting.export (Issues #1949, #1744)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from src.shared.python.plotting.export import (
    ExportConfig,
    export_figure,
    export_plot_data,
)
from src.shared.python.plotting.identity import PlotIdentity


class TestExportConfig:
    def test_export_default_construction(self) -> None:
        cfg = ExportConfig()
        assert cfg.image_format == "png"

    def test_default_dpi(self) -> None:
        assert ExportConfig().dpi == 300

    def test_default_vector_format(self) -> None:
        assert ExportConfig().vector_format == "pdf"

    def test_default_transparent_false(self) -> None:
        assert ExportConfig().transparent is False

    def test_default_include_metadata_true(self) -> None:
        assert ExportConfig().include_metadata is True

    def test_custom_output_dir(self) -> None:
        cfg = ExportConfig(output_dir="/tmp/test_exports")
        assert str(cfg.output_dir) == "/tmp/test_exports"


class TestExportFigure:
    def test_exports_png(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        config = ExportConfig(output_dir=str(tmp_path), dpi=72)
        paths = export_figure(fig, "test_plot", config=config, formats=["png"])
        plt.close("all")
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"

    def test_exports_multiple_formats(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        config = ExportConfig(output_dir=str(tmp_path), dpi=72)
        paths = export_figure(fig, "multi_test", config=config, formats=["png", "svg"])
        plt.close("all")
        assert len(paths) == 2
        exts = {p.suffix for p in paths}
        assert ".png" in exts
        assert ".svg" in exts

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        new_dir = tmp_path / "new_subdir"
        assert not new_dir.exists()
        fig, ax = plt.subplots()
        ax.plot([1])
        config = ExportConfig(output_dir=str(new_dir), dpi=72)
        export_figure(fig, "test", config=config, formats=["png"])
        plt.close("all")
        assert new_dir.exists()


class TestExportPlotData:
    def test_exports_json_format(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path))
        data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([1.0, 4.0, 9.0])}
        path = export_plot_data(data, "test_data", config=config, fmt="json")
        assert path.exists()
        assert path.suffix == ".json"

    def test_json_content_valid(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path))
        data = {"x": [1.0, 2.0], "y": [3.0, 4.0]}
        path = export_plot_data(data, "test", config=config, fmt="json")
        with open(path) as f:
            loaded = json.load(f)
        assert "x" in loaded
        assert loaded["x"] == [1.0, 2.0]

    def test_json_includes_metadata(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path), include_metadata=True)
        data = {"x": [1.0]}
        path = export_plot_data(data, "test", config=config, fmt="json")
        with open(path) as f:
            loaded = json.load(f)
        assert "_meta" in loaded

    def test_json_excludes_metadata_when_disabled(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path), include_metadata=False)
        data = {"x": [1.0]}
        path = export_plot_data(data, "test", config=config, fmt="json")
        with open(path) as f:
            loaded = json.load(f)
        assert "_meta" not in loaded

    def test_exports_csv_format(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path))
        data = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
        path = export_plot_data(data, "test_csv", config=config, fmt="csv")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_csv_has_header_and_rows(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path))
        data = {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}
        path = export_plot_data(data, "test", config=config, fmt="csv")
        lines = path.read_text().splitlines()
        assert len(lines) >= 4  # header + 3 data rows
        assert "x" in lines[0]
        assert "y" in lines[0]

    def test_numpy_array_serialized_correctly(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path))
        data = {"values": np.array([1.5, 2.5, 3.5])}
        path = export_plot_data(data, "arr_test", config=config, fmt="json")
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["values"] == [1.5, 2.5, 3.5]

    def test_export_unsupported_format_raises(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            export_plot_data({}, "test", config=config, fmt="xml")

    @pytest.mark.unit
    def test_json_metadata_includes_identity_fields(self, tmp_path: Path) -> None:
        """(c) JSON export carries identity fields, not just a static string."""
        config = ExportConfig(output_dir=str(tmp_path), include_metadata=True)
        identity = PlotIdentity(engine="mujoco", model="golfer_v3", run_id="run-9")
        data = {"x": [1.0]}
        path = export_plot_data(
            data, "test", config=config, fmt="json", identity=identity
        )
        with open(path) as f:
            loaded = json.load(f)
        meta = loaded["_meta"]
        assert meta["engine"] == "mujoco"
        assert meta["model"] == "golfer_v3"
        assert meta["run_id"] == "run-9"
        assert "exported_at" in meta
        assert meta["source"] == "UpstreamDrift"

    @pytest.mark.unit
    def test_json_metadata_omits_unknown_identity_fields(self, tmp_path: Path) -> None:
        config = ExportConfig(output_dir=str(tmp_path), include_metadata=True)
        identity = PlotIdentity(engine="drake")
        data = {"x": [1.0]}
        path = export_plot_data(
            data, "test", config=config, fmt="json", identity=identity
        )
        with open(path) as f:
            loaded = json.load(f)
        meta = loaded["_meta"]
        assert meta["engine"] == "drake"
        assert "model" not in meta
        assert "run_id" not in meta

    @pytest.mark.unit
    def test_json_metadata_without_identity_keeps_static_source(
        self, tmp_path: Path
    ) -> None:
        config = ExportConfig(output_dir=str(tmp_path), include_metadata=True)
        path = export_plot_data({"x": [1.0]}, "test", config=config, fmt="json")
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["_meta"]["source"] == "UpstreamDrift"
        assert "engine" not in loaded["_meta"]


class TestExportFigureMetadata:
    """(b) export_figure(include_metadata=True) embeds identity in PNG metadata."""

    @pytest.mark.unit
    def test_png_metadata_contains_identity_fields(self, tmp_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        config = ExportConfig(output_dir=str(tmp_path), dpi=72, include_metadata=True)
        identity = PlotIdentity(engine="mujoco", model="golfer_v3", run_id="run-9")
        paths = export_figure(
            fig, "meta_test", config=config, formats=["png"], identity=identity
        )
        plt.close("all")

        img = Image.open(paths[0])
        info = img.info
        assert info.get("engine") == "mujoco"
        assert info.get("model") == "golfer_v3"
        assert info.get("run_id") == "run-9"
        assert "Creation Time" in info
        assert info.get("Software") == "UpstreamDrift"

    @pytest.mark.unit
    def test_png_metadata_absent_when_include_metadata_false(
        self, tmp_path: Path
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        config = ExportConfig(output_dir=str(tmp_path), dpi=72, include_metadata=False)
        identity = PlotIdentity(engine="mujoco")
        paths = export_figure(
            fig, "no_meta_test", config=config, formats=["png"], identity=identity
        )
        plt.close("all")

        img = Image.open(paths[0])
        assert "engine" not in img.info

    @pytest.mark.unit
    def test_png_metadata_without_identity_still_has_timestamp(
        self, tmp_path: Path
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        config = ExportConfig(output_dir=str(tmp_path), dpi=72, include_metadata=True)
        paths = export_figure(fig, "identityless", config=config, formats=["png"])
        plt.close("all")

        img = Image.open(paths[0])
        assert "Creation Time" in img.info
        assert "engine" not in img.info

    @pytest.mark.unit
    def test_pdf_export_with_identity_does_not_raise(self, tmp_path: Path) -> None:
        """PDF's stricter metadata key vocabulary must not break exports."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        config = ExportConfig(output_dir=str(tmp_path), dpi=72, include_metadata=True)
        identity = PlotIdentity(engine="mujoco", model="golfer_v3")
        paths = export_figure(
            fig, "pdf_meta_test", config=config, formats=["pdf"], identity=identity
        )
        plt.close("all")
        assert paths[0].exists()
