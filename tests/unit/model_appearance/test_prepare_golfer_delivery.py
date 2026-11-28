import importlib
import pytest

from scripts import prepare_golfer_delivery as delivery


@pytest.mark.unit
def test_output_and_importer_fail_closed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = delivery._output(tmp_path / "delivery", repo, final=True)
    with pytest.raises(FileExistsError):
        delivery._output(target, repo, final=True)
    monkeypatch.setattr(
        importlib, "import_module", lambda _: (_ for _ in ()).throw(ImportError())
    )
    assert delivery._import_api(required=False) is None
    with pytest.raises(RuntimeError, match="#15957"):
        delivery._import_api(required=True)
