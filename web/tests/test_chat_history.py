"""The chat endpoint carries earlier turns so follow-up questions can refer to them."""
import pytest

from rag.answering import CitedSource, Turn

EARLIER = {
    "question": "What is new in graph neural networks?",
    "answer": "Sheaf-MPNN is a library [1].",
    "sources": [{"number": 1, "title": "Sheaf-MPNN"}],
}


def test_chat_without_history_passes_no_turns(client, assistant):
    client.post("/api/chat", json={"question": "What?"})
    assert assistant.histories == [[]]


def test_chat_passes_earlier_turns_to_the_assistant(client, assistant):
    client.post("/api/chat", json={"question": "Explain the first one", "history": [EARLIER]})
    assert assistant.questions == ["Explain the first one"]
    assert assistant.histories == [[Turn(
        question="What is new in graph neural networks?",
        answer="Sheaf-MPNN is a library [1].",
        sources=[CitedSource(1, "Sheaf-MPNN")],
    )]]


def test_chat_turn_without_sources_is_accepted(client, assistant):
    turn = {"question": "Q", "answer": "I don't have information about that in the archive."}
    assert client.post("/api/chat", json={"question": "And?", "history": [turn]}).status_code == 200
    assert assistant.histories[0][0].sources == []


def test_chat_returns_what_was_searched_for(client, assistant):
    assistant.outcome.searched_for = "How does Sheaf-MPNN work?"
    body = client.post("/api/chat", json={"question": "Explain it", "history": [EARLIER]}).json()
    assert body["searched_for"] == "How does Sheaf-MPNN work?"


@pytest.mark.parametrize("history", [
    [EARLIER] * 11,
    [{**EARLIER, "question": "x" * 1001}],
    [{**EARLIER, "answer": "x" * 8001}],
    [{**EARLIER, "sources": [{"number": 1, "title": "t"}] * 21}],
    [{"question": "Q"}],
])
def test_chat_rejects_unusable_history_without_calling_the_llm(client, assistant, history):
    response = client.post("/api/chat", json={"question": "And?", "history": history})
    assert response.status_code == 422
    assert assistant.questions == []
