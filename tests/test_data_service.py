from __future__ import annotations

import pandas as pd

from src.services import data_service


def test_load_dataframe_prefers_upload(monkeypatch):
    upload_df = pd.DataFrame({"label": [0], "TLD": ["com"], "URLLength": [10.0]})

    monkeypatch.setattr(data_service, "resolve_default_path", lambda: type("P", (), {"is_file": lambda self: True})())
    monkeypatch.setattr(data_service, "load_csv_from_upload", lambda _uploaded: upload_df)
    monkeypatch.setattr(data_service, "load_csv_from_path", lambda _path: pd.DataFrame())

    result = data_service.load_dataframe(uploaded=object())
    assert result["source"] == "upload"
    assert result["error"] is None
    assert result["dataframe"] is upload_df


def test_load_dataframe_uses_default_path(monkeypatch):
    default_df = pd.DataFrame({"label": [1], "TLD": ["org"], "URLLength": [99.0]})

    monkeypatch.setattr(data_service, "resolve_default_path", lambda: type("P", (), {"is_file": lambda self: True})())
    monkeypatch.setattr(data_service, "load_csv_from_path", lambda _path: default_df)

    result = data_service.load_dataframe(uploaded=None)
    assert result["source"] == "default"
    assert result["error"] is None
    assert result["dataframe"] is default_df


def test_load_dataframe_none_when_no_sources(monkeypatch):
    monkeypatch.setattr(data_service, "resolve_default_path", lambda: type("P", (), {"is_file": lambda self: False})())
    result = data_service.load_dataframe(uploaded=None)
    assert result == {"dataframe": None, "source": "none", "error": None}


def test_load_dataframe_returns_error_message(monkeypatch):
    monkeypatch.setattr(data_service, "resolve_default_path", lambda: type("P", (), {"is_file": lambda self: True})())
    monkeypatch.setattr(data_service, "load_csv_from_path", lambda _path: (_ for _ in ()).throw(ValueError("bad csv")))
    result = data_service.load_dataframe(uploaded=None)
    assert result["source"] == "error"
    assert result["dataframe"] is None
    assert "bad csv" in (result["error"] or "")

