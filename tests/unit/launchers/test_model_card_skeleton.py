import pytest
from PyQt6.QtWidgets import QGraphicsOpacityEffect

from src.launchers.model_card import SkeletonCard


@pytest.mark.unit
def test_skeleton_card_initialization(qapp) -> None:
    """Test that SkeletonCard initializes with correct dimensions and effects."""
    card = SkeletonCard()

    assert card.objectName() == "SkeletonCard"
    assert card.minimumSize().width() == 180
    assert card.minimumSize().height() == 240

    # The pulse must be driven by a QGraphicsOpacityEffect installed on the
    # frame itself -- setWindowOpacity() is a no-op on a non top-level
    # widget (issue #8906).
    effect = card.graphicsEffect()
    assert isinstance(effect, QGraphicsOpacityEffect)


@pytest.mark.unit
def test_skeleton_card_animation_has_distinct_endpoints(qapp) -> None:
    """The pulse animation must actually move between two opacity values.

    Regression test for issue #8906: the original animation set both
    ``startValue`` and ``endValue`` to 0.3, making the "pulse" a
    mathematical no-op.
    """
    card = SkeletonCard()

    assert hasattr(card, "_anim")
    anim = card._anim
    assert anim.loopCount() == -1

    # Ping-ponging animation group: one leg growing, one leg shrinking,
    # with distinct, non-equal endpoints.
    assert anim.animationCount() == 2
    forward = anim.animationAt(0)
    backward = anim.animationAt(1)

    assert forward.propertyName() == b"pulseOpacity"
    assert backward.propertyName() == b"pulseOpacity"
    assert forward.startValue() != forward.endValue()
    assert forward.startValue() == pytest.approx(backward.endValue())
    assert forward.endValue() == pytest.approx(backward.startValue())


@pytest.mark.unit
def test_skeleton_card_pulse_opacity_setter_updates_effect(qapp) -> None:
    """The pulseOpacity property setter must write through to the
    QGraphicsOpacityEffect instead of calling the no-op setWindowOpacity().
    """
    card = SkeletonCard()

    card.pulseOpacity = 0.6

    assert card.pulseOpacity == pytest.approx(0.6)
    assert card.graphicsEffect().opacity() == pytest.approx(0.6)


@pytest.mark.unit
def test_skeleton_card_stops_animation_on_hide(qapp) -> None:
    """Hiding a skeleton card must stop its animation so it doesn't leak a
    forever-running timer once the card is torn down (issue #8906).
    """
    card = SkeletonCard()
    card.show()
    assert card._anim.state() == card._anim.State.Running

    card.hide()

    assert card._anim.state() == card._anim.State.Stopped
