"""Tests for :class:`DataChannelEditor`."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.plot_style.channels import DataChannel
from src.shared.python.plot_style.widgets.data_channel_editor import (
    DataChannelEditor,
)


def _channel(name: str, values: np.ndarray) -> DataChannel:
    return DataChannel(name=name, values=values)


@pytest.fixture
def channels() -> list[DataChannel]:
    return [
        _channel("speed", np.array([0.0, 1.0, 2.0, 3.0])),
        _channel("force", np.array([10.0, 20.0, 30.0, 40.0])),
        _channel("height", np.array([-1.0, 0.0, 1.0])),
    ]


def test_data_channel_editor_default_initial_value(channels: list[DataChannel]) -> None:
    widget = DataChannelEditor(channels)
    assert widget.value() is channels[0]
    vmin, vmax = widget.range_value()
    assert vmin == 0.0
    assert vmax == 3.0


def test_empty_channels_rejected() -> None:
    with pytest.raises(ValueError):
        DataChannelEditor([])


def test_non_sequence_channels_rejected() -> None:
    with pytest.raises(TypeError):
        DataChannelEditor("not-a-sequence")  # type: ignore[arg-type]


def test_non_channel_entry_rejected() -> None:
    with pytest.raises(TypeError):
        DataChannelEditor([object()])  # type: ignore[list-item]


def test_initial_not_in_list_rejected(channels: list[DataChannel]) -> None:
    other = _channel("other", np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        DataChannelEditor(channels, initial=other)


def test_set_value_emits_channel_changed(  # type: ignore[no-untyped-def]
    channels, qtbot
) -> None:
    widget = DataChannelEditor(channels)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.channelChanged, timeout=500) as blocker:
        widget.set_value(channels[1])
    assert blocker.args == [channels[1]]
    assert widget.value() is channels[1]


def test_data_channel_editor_set_value_no_emit_when_unchanged(  # type: ignore[no-untyped-def]
    channels, qtbot
) -> None:
    widget = DataChannelEditor(channels)
    qtbot.addWidget(widget)
    received: list[DataChannel] = []
    widget.channelChanged.connect(received.append)
    widget.set_value(channels[0])
    assert received == []


def test_set_value_rejects_unknown_channel(channels: list[DataChannel]) -> None:
    widget = DataChannelEditor(channels)
    other = _channel("other", np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        widget.set_value(other)


def test_set_value_rejects_wrong_type(channels: list[DataChannel]) -> None:
    widget = DataChannelEditor(channels)
    with pytest.raises(TypeError):
        widget.set_value("not-a-channel")  # type: ignore[arg-type]


def test_set_range_emits(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DataChannelEditor(channels)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.rangeChanged, timeout=500) as blocker:
        widget.set_range(-5.0, 5.0)
    assert blocker.args == [-5.0, 5.0]
    assert widget.range_value() == (-5.0, 5.0)


def test_set_range_validates_order(channels: list[DataChannel]) -> None:
    widget = DataChannelEditor(channels)
    with pytest.raises(ValueError):
        widget.set_range(1.0, 1.0)
    with pytest.raises(ValueError):
        widget.set_range(2.0, 1.0)


def test_set_range_validates_finite(channels: list[DataChannel]) -> None:
    widget = DataChannelEditor(channels)
    with pytest.raises(ValueError):
        widget.set_range(float("nan"), 1.0)


def test_set_range_round_trip(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DataChannelEditor(channels)
    qtbot.addWidget(widget)
    for vmin, vmax in [(0.0, 1.0), (-3.0, 7.0), (10.0, 20.0)]:
        widget.set_range(vmin, vmax)
        assert widget.range_value() == (vmin, vmax)


def test_combo_user_change_emits(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DataChannelEditor(channels)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.channelChanged, timeout=500) as blocker:
        widget._channel_combo.setCurrentIndex(1)
    assert blocker.args == [channels[1]]


def test_spin_user_change_emits(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DataChannelEditor(channels)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.rangeChanged, timeout=500):
        widget._vmax_spin.setValue(99.0)
    assert widget.range_value()[1] == 99.0


def test_fit_button_uses_default_fit(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    widget = DataChannelEditor(channels, initial_range=(-100.0, 100.0))
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.rangeChanged, timeout=500) as blocker:
        widget._fit_button.click()
    assert blocker.args == [0.0, 3.0]


def test_fit_button_uses_custom_fit(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    def custom_fit(channel: DataChannel) -> tuple[float, float]:
        return (-99.0, 99.0)

    widget = DataChannelEditor(channels, fit_fn=custom_fit, initial_range=(0.0, 1.0))
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.rangeChanged, timeout=500) as blocker:
        widget._fit_button.click()
    assert blocker.args == [-99.0, 99.0]


def test_fit_button_handles_invalid_fit(channels: list[DataChannel], qtbot) -> None:  # type: ignore[no-untyped-def]
    def bad_fit(channel: DataChannel) -> tuple[float, float]:
        return (5.0, 5.0)  # invalid: vmax not > vmin

    widget = DataChannelEditor(channels, fit_fn=bad_fit, initial_range=(0.0, 1.0))
    qtbot.addWidget(widget)
    received: list[tuple[float, float]] = []
    widget.rangeChanged.connect(lambda lo, hi: received.append((lo, hi)))
    widget._fit_button.click()
    assert received == []  # invalid fit silently ignored
    assert widget.range_value() == (0.0, 1.0)


def test_default_fit_falls_back_for_nan_channel(qtbot) -> None:  # type: ignore[no-untyped-def]
    nan_channel = _channel("nan_only", np.array([float("nan"), float("nan")]))
    widget = DataChannelEditor([nan_channel])
    qtbot.addWidget(widget)
    assert widget.range_value() == (0.0, 1.0)
