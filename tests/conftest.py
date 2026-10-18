import pytest


@pytest.fixture(autouse=True)
def isolate_outputs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
