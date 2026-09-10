"""Display contracts preserve gaps, provenance and dimensional separation."""

import csv
import io

import pytest

pytestmark = pytest.mark.unit

from src.shared.python.analysis.biomechanics_display import (
    biomechanics_csv,
    prepare_biomechanics_plot,
)


@pytest.fixture
def result():
    return {
        "times": [0, 0.1, 0.2],
        "channels": {
            "angle": {
                "values": [0, None, 3.141592653589793],
                "unit": "rad",
                "definition": "Signed Separation",
                "frame": "world",
            },
            "com": {
                "values": [[1, 2, 3], [None, None, None], [4, 5, 6]],
                "unit": "m",
                "definition": "Center of Mass",
                "frame": "world",
            },
        },
        "source": "measured",
        "unavailable": {"shaft_twist_velocity": "Missing Orientation"},
        "events": {"impact": 2},
    }


def test_flatten_convert_and_preserve_gaps(result):
    plot = prepare_biomechanics_plot(result)
    assert plot["channels"]["angle"]["values"] == [0, None, 180]
    assert plot["channels"]["angle"]["unit"] == "deg"
    assert plot["channels"]["com.x"]["values"] == [1, None, 4]
    assert plot["channels"]["com.x"]["frame"] == "world"
    assert plot["unavailable"] == result["unavailable"]
    assert plot["events"] == {"impact": 0.2}
    assert result["channels"]["angle"]["unit"] == "rad"


def test_selection_csv_has_units_and_empty_gaps(result):
    plot = prepare_biomechanics_plot(result, selected=["angle", "com.y"])
    rows = list(csv.reader(io.StringIO(biomechanics_csv(plot))))
    assert rows[0] == ["Time (s)", "angle (deg)", "com.y (m)"]
    assert rows[2] == ["0.1", "", ""]


def test_rejects_unknown_selection_and_misaligned_data(result):
    with pytest.raises(ValueError, match="Unknown"):
        prepare_biomechanics_plot(result, selected=["absent"])
    result["channels"]["angle"]["values"] = [1]
    with pytest.raises(ValueError, match="length"):
        prepare_biomechanics_plot(result)


def test_matplotlib_separates_units_preserves_gaps_and_exports(result, tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    from src.shared.python.analysis.biomechanics_display import (
        render_biomechanics_figure,
    )

    figure = Figure()
    render_biomechanics_figure(prepare_biomechanics_plot(result), figure)
    assert [axis.get_ylabel() for axis in figure.axes] == ["deg", "m"]
    assert len(figure.axes[0].lines) == 2  # channel and impact marker
    path = tmp_path / "metrics.svg"
    figure.savefig(path)
    assert "angle" in path.read_text(encoding="utf-8")


def test_desktop_result_has_every_channel_and_can_switch_units(result):
    try:
        from PyQt6.QtWidgets import QApplication
        from src.shared.python.dashboard.biomechanics_widget import BiomechanicsWidget
    except (ImportError, OSError) as error:
        pytest.skip(f"Optional GUI dependencies unavailable: {error}")
    application = QApplication.instance() or QApplication([])
    widget = BiomechanicsWidget()
    widget.set_result(result)
    assert widget.channels.count() == 4
    widget.angle_unit.setCurrentText("rad")
    assert widget._prepared()["channels"]["angle"]["unit"] == "rad"
    widget.close()
    assert application is not None
