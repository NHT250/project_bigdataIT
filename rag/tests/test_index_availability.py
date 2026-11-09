"""Machines without a GPU or the model skip summaries and keep running."""
import logging
import sys
import types

import psycopg
import torch

from rag import index, settings


def install_model(root):
    folder = root / "models" / "summarizer"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text("{}")


def fake_summarizer_module(monkeypatch, loaded):
    class Summarizer:
        def __init__(self):
            loaded.append(True)

    monkeypatch.setitem(sys.modules, "rag.summarizer", types.SimpleNamespace(Summarizer=Summarizer))


def test_no_gpu_skips_without_loading_the_model(monkeypatch, tmp_path, caplog, conn):
    install_model(tmp_path)
    loaded = []
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fake_summarizer_module(monkeypatch, loaded)

    with caplog.at_level(logging.WARNING):
        assert index.summarize_if_possible(conn) == 0

    assert loaded == []
    assert "GPU" in caplog.text


def test_missing_model_skips_without_loading_it(monkeypatch, tmp_path, caplog, conn):
    loaded = []
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_summarizer_module(monkeypatch, loaded)

    with caplog.at_level(logging.WARNING):
        assert index.summarize_if_possible(conn) == 0

    assert loaded == []
    assert "models/summarizer" in caplog.text


def test_gpu_and_model_run_the_summarizer(monkeypatch, tmp_path, conn):
    install_model(tmp_path)
    loaded = []
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_summarizer_module(monkeypatch, loaded)

    assert index.summarize_if_possible(conn) == 0  # Nothing to summarize in an empty schema.
    assert loaded == [True]


def test_prepare_schema_builds_every_table_on_an_empty_database():
    with psycopg.connect(settings.database_url(), autocommit=True) as connection:
        schema = "test_prepare_schema"
        connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(f'SET search_path TO "{schema}", public')
        try:
            index.prepare_schema(connection)
            index.prepare_schema(connection)  # Running it twice is harmless.
            tables = connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", (schema,)
            ).fetchall()
        finally:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
    assert {name for (name,) in tables} == {"articles", "article_chunks", "article_summaries"}
