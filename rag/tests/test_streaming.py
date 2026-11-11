"""Streamed answers: what was searched for, the answer piece by piece, then the checked result."""
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from rag import answering
from rag.answering import AnswerError, GeminiGenerator, IncompleteReply, Turn, answer_events


def row(number, text="Chunk text"):
    return {
        "source": "arxiv", "source_id": f"id{number}", "title": f"Paper {number}",
        "url": f"https://arxiv.org/abs/id{number}", "text": f"{text} {number}", "summary": None, "score": 0.8,
    }


class Scripted:
    """Replies in order; a list reply is streamed piece by piece, an exception is raised at the end."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0)

    def stream(self, system, user):
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        pieces = reply if isinstance(reply, list) else [reply]
        for piece in pieces:
            if isinstance(piece, Exception):
                raise piece
            yield piece


def events(question, *replies, history=(), retrieve=None):
    model = Scripted(*replies)
    retrieve = retrieve or (lambda query: [row(1), row(2)])
    found = list(answer_events(question, retrieve=retrieve, generate=model.generate, stream=model.stream,
                               searched_articles=5, api_key="key", history=history))
    return found, model


def kinds(found):
    return [event["type"] for event in found]


def streamed_text(found):
    return "".join(event["text"] for event in found if event["type"] == "delta")


def test_the_answer_arrives_in_pieces_and_ends_with_the_checked_result():
    found, _ = events("What is it?", ["It is ", "a parser ", "[1]."])
    assert kinds(found) == ["searching", "delta", "delta", "delta", "done"]
    assert streamed_text(found) == "It is a parser [1]."
    final = found[-1]["answer"]
    assert (final.status, final.text) == ("answered", "It is a parser [1].")
    assert [s.number for s in final.sources] == [1]


def test_a_first_question_is_searched_as_asked_without_a_rewrite():
    found, model = events("What is Sheaf-MPNN?", ["A library [1]."])
    assert found[0] == {"type": "searching", "searched_for": "What is Sheaf-MPNN?"}
    assert len(model.calls) == 1


def test_a_follow_up_shows_the_rewritten_search_before_the_answer():
    earlier = Turn("What is new in graph neural networks?", "Sheaf-MPNN [1] and FiLSim [2].")
    found, _ = events("Tell me more about the second one", "Tell me more about FiLSim graph similarity learning",
                      ["FiLSim scales graph search [1]."], history=[earlier])
    assert found[0] == {"type": "searching", "searched_for": "Tell me more about FiLSim graph similarity learning"}
    assert found[-1]["answer"].searched_for == "Tell me more about FiLSim graph similarity learning"


def test_the_no_information_marker_is_never_streamed_even_in_pieces():
    found, _ = events("What is it?", ["NO_", "INFOR", "MATION"], answering.NO_INFORMATION, answering.NO_INFORMATION)
    assert "delta" not in kinds(found)
    assert found[-1]["answer"].status == "no_information"


def test_a_missed_question_is_searched_again_and_only_the_second_answer_is_streamed():
    def retrieve(query):
        return [row(9, "Artificial intelligence is the field")] if "artificial" in query else [row(1)]

    found, _ = events("what is AI", [answering.NO_INFORMATION], "artificial intelligence (AI) definition",
                      ["AI is a field ", "of computer science [1]."], retrieve=retrieve)
    searches = [event["searched_for"] for event in found if event["type"] == "searching"]
    assert searches == ["what is AI", "artificial intelligence (AI) definition"]
    assert streamed_text(found) == "AI is a field of computer science [1]."
    assert found[-1]["answer"].status == "answered"


def test_a_streamed_answer_with_a_bad_citation_ends_rejected():
    found, _ = events("What is it?", ["It is a parser [7]."])
    assert found[-1]["answer"].status == "rejected"
    assert found[-1]["answer"].text == answering.REJECTED_MESSAGE


def test_a_reply_cut_off_mid_stream_ends_rejected():
    found, _ = events("What is it?", ["It is a parser [1]", IncompleteReply()])
    assert found[-1]["answer"].status == "rejected"


def test_nothing_retrieved_ends_without_streaming_or_calling_the_llm():
    found, model = events("What is it?", retrieve=lambda query: [])
    assert kinds(found) == ["searching", "done"]
    assert found[-1]["answer"].status == "no_information"
    assert model.calls == []


# The Gemini side of streaming


def chunk(text, finish_reason=None, thought=None):
    parts = [types.Part(text=thought, thought=True)] if thought else []
    parts.append(types.Part(text=text))
    candidate = types.Candidate(content=types.Content(parts=parts), finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate])


class FakeStreamingModels:
    """Plays back one outcome per call: an exception, or a list of chunks (an exception inside is raised there)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content_stream(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        def play():
            for item in outcome:
                if isinstance(item, Exception):
                    raise item
                yield item
        return play()


def gemini(*outcomes):
    models = FakeStreamingModels(*outcomes)
    waits = []
    return GeminiGenerator(SimpleNamespace(models=models), model="gemini-test", sleep=waits.append), models, waits


def rate_limited():
    return errors.ClientError(429, {"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED"}})


def test_gemini_stream_yields_the_answer_without_the_reasoning():
    generator, models, _ = gemini([chunk("It is ", thought="thinking"), chunk("a parser [1].", "STOP")])
    assert list(generator.stream("system", "user")) == ["It is ", "a parser [1]."]
    assert models.calls[0]["contents"] == "user"
    assert models.calls[0]["config"].system_instruction == "system"


def test_gemini_stream_retries_a_rate_limit_before_anything_was_sent():
    generator, _, waits = gemini(rate_limited(), [chunk("Hi [1].", "STOP")])
    assert list(generator.stream("s", "u")) == ["Hi [1]."]
    assert len(waits) == 1


def test_gemini_stream_reports_a_cut_off_reply():
    generator, _, _ = gemini([chunk("It is a ", None), chunk("par", "MAX_TOKENS")])
    with pytest.raises(IncompleteReply):
        list(generator.stream("s", "u"))


def test_gemini_stream_failure_after_the_first_piece_is_a_readable_error():
    generator, _, _ = gemini([chunk("It is "), errors.ServerError(500, {"error": {"code": 500, "message": "boom"}})])
    with pytest.raises(AnswerError, match="500"):
        list(generator.stream("s", "u"))
