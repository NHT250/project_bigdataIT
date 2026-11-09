"""Where the database address comes from on the host and in a container."""
from rag import settings


def test_database_url_from_the_environment_wins(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@postgres:5432/db")
    assert settings.database_url() == "postgresql://u:p@postgres:5432/db"


def test_database_url_defaults_to_the_local_container(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss")
    assert settings.database_url() == "postgresql://techpulse:p%40ss@127.0.0.1:5433/techpulse"
