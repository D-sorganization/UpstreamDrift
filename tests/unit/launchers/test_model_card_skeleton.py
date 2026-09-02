from src.launchers.model_card import SkeletonCard


def test_skeleton_card_initialization(qapp) -> None:
    """Test that SkeletonCard initializes with correct dimensions and effects."""
    card = SkeletonCard()

    assert card.objectName() == "SkeletonCard"
    assert card.minimumSize().width() == 180
    assert card.minimumSize().height() == 240
    assert card.graphicsEffect() is not None

    # Check Animation
    assert hasattr(card, "_anim")
    anim = card._anim
    assert anim.propertyName() == b"pulseOpacity"
    assert anim.duration() == 1000
    assert anim.startValue() == 0.3
    assert anim.endValue() == 0.3
    assert anim.loopCount() == -1
