"""The command line reports a missing key cleanly, before loading any model."""
from rag import ask, settings


def test_missing_key_prints_a_clear_message_and_exits_2(monkeypatch, capsys):
    monkeypatch.setattr(settings, "get", lambda name, default="": "")

    code = ask.main(["What is new?"])

    assert code == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
