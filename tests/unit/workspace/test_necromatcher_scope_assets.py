"""Portable registration is mandatory before queue or persisted scope admission."""

from types import SimpleNamespace
import pytest
from test_necromatcher_scope_integration import scoped
from src.shared.python.workspace import necromatcher_fit as fit

pytestmark = pytest.mark.unit


def test_external_review_rejected_before_capture_provider(
    tmp_path, monkeypatch
) -> None:
    identity, scope = scoped(tmp_path)
    source = {
        "capture_id": identity.capture_id,
        "capture_hash": identity.capture_hash,
        "provenance": {},
    }
    calls = []
    monkeypatch.setattr(fit, "capture_identity", lambda *a: calls.append("capture"))

    def missing(_):
        raise KeyError("review")

    library = SimpleNamespace(load_source_scope_review=missing)
    with pytest.raises(ValueError, match="Registered"):
        fit.admit_refit_scope(library, source, (0, 2), requested=scope)
    assert calls == []


def test_registered_receipt_cannot_be_transplanted(tmp_path, monkeypatch) -> None:
    _, scope = scoped(tmp_path)
    _, other = scoped(tmp_path / "other", end=4)
    library = SimpleNamespace(load_source_scope_review=lambda _: other)
    source = {
        "capture_id": scope.capture_id,
        "capture_hash": scope.capture_hash,
        "provenance": {},
    }
    calls = []
    monkeypatch.setattr(fit, "capture_identity", lambda *a: calls.append("capture"))
    with pytest.raises(ValueError, match="registered"):
        fit.admit_refit_scope(library, source, (0, 2), requested=scope)
    assert calls == []
