"""Built-in reference reviewers (Tools #2738).

Three reviewers — one per role — that demonstrate the protocol and serve
as drop-in defaults until a project supplies its own provider-backed
reviewers.
"""

from __future__ import annotations

from .advocate_reviewer import AdvocateReviewer
from .critic_reviewer import CriticReviewer
from .specialist_reviewer import SpecialistReviewer

__all__ = [
    "AdvocateReviewer",
    "CriticReviewer",
    "SpecialistReviewer",
]
