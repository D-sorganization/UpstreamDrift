import pytest
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from PyQt6.QtCore import QPropertyAnimation
from src.launchers.model_card import SkeletonCard

def test_skeleton_card_initialization(qapp):
    """Test that SkeletonCard initializes with correct dimensions and effects."""
    card = SkeletonCard()
    
    assert card.objectName() == "SkeletonCard"
    assert card.minimumWidth() == 180
    assert card.minimumHeight() == 240
    
    # Check Graphics Effect
    effect = card.graphicsEffect()
    assert isinstance(effect, QGraphicsDropShadowEffect)
    assert effect.blurRadius() == 20
    
    # Check Animation
    assert hasattr(card, "_anim")
    anim = card._anim
    assert isinstance(anim, QPropertyAnimation)
    assert anim.propertyName() == b"windowOpacity"
    assert anim.duration() == 1000
    assert anim.startValue() == 0.5
    assert anim.endValue() == 1.0
    assert anim.loopCount() == -1
    
    # Check if animation is running
    from PyQt6.QtCore import QAbstractAnimation
    assert anim.state() == QAbstractAnimation.State.Running
