from src.launchers.model_card import SkeletonCard


def test_skeleton_card_initialization(qapp):
    """Test that SkeletonCard initializes with correct dimensions and effects."""
    card = SkeletonCard()

    card.setObjectName.assert_called_with("SkeletonCard")

    if hasattr(card.setMinimumSize, "assert_called_with"):
        card.setMinimumSize.assert_called_with(180, 240)

    # Check Graphics Effect was applied
    if hasattr(card.setGraphicsEffect, "assert_called"):
        card.setGraphicsEffect.assert_called()

    # Check Animation
    assert hasattr(card, "_anim")
    anim = card._anim
    if hasattr(anim.setPropertyName, "assert_called_with"):
        anim.setPropertyName.assert_called_with(b"windowOpacity")
        anim.setDuration.assert_called_with(1000)
        anim.setStartValue.assert_called_with(0.5)
        anim.setEndValue.assert_called_with(1.0)
        anim.setLoopCount.assert_called_with(-1)

    if hasattr(anim.start, "assert_called"):
        anim.start.assert_called()
