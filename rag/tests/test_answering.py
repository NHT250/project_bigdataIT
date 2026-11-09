"""The answer pipeline: grounding rules, citation checks and error reporting."""
from types import SimpleNamespace

import httpx2
import pytest
from google.genai import errors, types

from rag import answering
from rag.answering import GeminiGenerator, MissingApiKey, answer


def source(number=1, **overrides):
    row = {
        "source": "arxiv", "source_id": f"id{number}", "title": f"Paper {number}",
        "url": f"https://arxiv.org/abs/id{number}", "text": f"Chunk text {number}",
        "summary": f"Summary {number}", "score": 0.9,
    }
    return {**row, **overrides}


class Generator:
    def __init__(self, reply=""):
        self.reply = reply
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.reply


def ask(reply, sources=None, **kwargs):
    generator = Generator(reply)
    rows = [source(1), source(2)] if sources is None else sources
    result = answer(
        "What is it?", retrieve=lambda question: rows, generate=generator,
        searched_articles=42, api_key="key", **kwargs,
    )
    return result, generator


def test_cited_answer_is_returned():
    result, _ = ask("It is a parser [1].")
    assert (result.status, result.text) == ("answered", "It is a parser [1].")


def test_only_cited_sources_are_returned_with_link_and_summary():
    result, _ = ask("It is a parser [2].")
    assert [(s.number, s.url, s.summary) for s in result.sources] == [
        (2, "https://arxiv.org/abs/id2", "Summary 2")
    ]


def test_result_reports_how_many_articles_were_searched():
    result, _ = ask("It is a parser [1].")
    assert result.searched_articles == 42


def test_grouped_citations_count_as_citations():
    result, _ = ask("It is a parser [1, 2].")
    assert [s.number for s in result.sources] == [1, 2]


def test_citation_beyond_the_sources_rejects_the_answer():
    result, _ = ask("It is a parser [3].")
    assert (result.status, result.sources) == ("rejected", [])
    assert "[3]" not in result.text


def test_citation_zero_rejects_the_answer():
    result, _ = ask("It is a parser [0].")
    assert result.status == "rejected"


def test_one_bad_citation_among_good_ones_rejects_the_answer():
    result, _ = ask("It is a parser [1] and a lexer [9].")
    assert result.status == "rejected"


def test_answer_without_citations_is_rejected():
    result, _ = ask("It is a parser.")
    assert result.status == "rejected"


def test_empty_reply_is_rejected():
    result, _ = ask("")
    assert result.status == "rejected"


def test_no_information_marker_says_the_archive_lacks_it():
    result, _ = ask("NO_INFORMATION")
    assert (result.status, result.text, result.sources) == (
        "no_information", answering.NO_INFORMATION_MESSAGE, []
    )


def test_nothing_retrieved_says_no_information_without_calling_the_llm():
    result, generator = ask("It is a parser [1].", sources=[])
    assert result.status == "no_information"
    assert generator.calls == []


def test_missing_api_key_is_a_clear_error():
    with pytest.raises(MissingApiKey, match="GEMINI_API_KEY"):
        answer("Q", retrieve=lambda q: [], generate=Generator(), searched_articles=0, api_key="")


def test_sources_are_numbered_in_the_prompt():
    _, generator = ask("It is a parser [1].")
    user = generator.calls[0][1]
    assert '<source id="1"' in user and "Chunk text 1" in user
    assert '<source id="2"' in user and "Chunk text 2" in user
    assert "What is it?" in user


def test_system_prompt_states_the_three_rules():
    _, generator = ask("It is a parser [1].")
    system = generator.calls[0][0]
    assert "only" in system.lower()
    assert "[1]" in system
    assert answering.NO_INFORMATION in system


def test_hn_source_without_a_url_links_to_the_discussion():
    result, _ = ask("A story [1].", sources=[source(1, source="hn", source_id="77", url=None)])
    assert result.sources[0].url == "https://news.ycombinator.com/item?id=77"


def test_arxiv_source_without_a_url_links_to_the_abstract():
    result, _ = ask("A paper [1].", sources=[source(1, source_id="2401.1", url=None)])
    assert result.sources[0].url == "https://arxiv.org/abs/2401.1"


def response(text, finish_reason="STOP"):
    parts = [types.Part(text="hidden reasoning", thought=True), types.Part(text=text)]
    candidate = types.Candidate(content=types.Content(parts=parts), finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate])


class FakeModels:
    """Plays back one outcome per call: an exception to raise or a response to return."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def gemini(*outcomes):
    models = FakeModels(*outcomes)
    waits = []
    generator = GeminiGenerator(SimpleNamespace(models=models), model="gemini-test", sleep=waits.append)
    return generator, models, waits


def api_error(code, message="boom", status="ERROR", details=None):
    error = {"code": code, "message": message, "status": status, "details": details or []}
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": error})


def rate_limited(retry_delay=None):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}] if retry_delay else []
    return api_error(429, "quota exceeded", "RESOURCE_EXHAUSTED", details)


def test_generator_returns_the_answer_without_the_reasoning():
    generator, _, _ = gemini(response("Hello [1]."))
    assert generator("system", "user") == "Hello [1]."


def test_generator_sends_system_and_question():
    generator, models, _ = gemini(response("x"))
    generator("the system", "the user")
    assert models.calls[0]["model"] == "gemini-test"
    assert models.calls[0]["contents"] == "the user"
    assert models.calls[0]["config"].system_instruction == "the system"


def test_generator_treats_a_blocked_reply_as_no_reply():
    generator, _, _ = gemini(response("partial", finish_reason="SAFETY"))
    assert generator("s", "u") == ""


def test_generator_treats_a_truncated_reply_as_no_reply():
    generator, _, _ = gemini(response("partial [1]", finish_reason="MAX_TOKENS"))
    assert generator("s", "u") == ""


def test_generator_treats_a_response_without_candidates_as_no_reply():
    generator, _, _ = gemini(types.GenerateContentResponse(candidates=[]))
    assert generator("s", "u") == ""


@pytest.mark.parametrize("error,expected", [
    (api_error(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"), "GEMINI_API_KEY"),
    (api_error(403, "forbidden", "PERMISSION_DENIED"), "GEMINI_API_KEY"),
    (api_error(404, "model not found", "NOT_FOUND"), "gemini-test"),
    (api_error(400, "bad request", "INVALID_ARGUMENT"), "400"),
    (api_error(500, "server error", "INTERNAL"), "500"),
])
def test_generator_turns_api_failures_into_readable_errors(error, expected):
    generator, _, _ = gemini(error)
    with pytest.raises(answering.AnswerError, match=expected):
        generator("s", "u")


def test_generator_reports_a_network_failure_as_a_readable_error():
    generator, _, _ = gemini(httpx2.ConnectError("no route"))
    with pytest.raises(answering.AnswerError, match="network"):
        generator("s", "u")


def test_rate_limit_is_retried_until_it_succeeds():
    generator, models, _ = gemini(rate_limited(), rate_limited(), response("Hello [1]."))
    assert generator("s", "u") == "Hello [1]."
    assert len(models.calls) == 3


def test_rate_limit_waits_grow_exponentially():
    generator, _, waits = gemini(rate_limited(), rate_limited(), rate_limited(), response("x"))
    generator("s", "u")
    assert waits == [2, 4, 8]


def test_rate_limit_waits_as_long_as_the_api_asks():
    generator, _, waits = gemini(rate_limited("34.5s"), response("x"))
    generator("s", "u")
    assert waits == [34.5]


def test_rate_limit_wait_is_capped():
    generator, _, waits = gemini(rate_limited("86400s"), response("x"))
    generator("s", "u")
    assert waits == [answering.MAX_RETRY_DELAY_SECONDS]


def test_rate_limit_that_never_clears_is_a_readable_error():
    generator, models, _ = gemini(*[rate_limited() for _ in range(answering.MAX_ATTEMPTS)])
    with pytest.raises(answering.AnswerError, match="rate limit"):
        generator("s", "u")
    assert len(models.calls) == answering.MAX_ATTEMPTS


def test_other_api_failures_are_not_retried():
    generator, models, waits = gemini(api_error(500, "server error", "INTERNAL"), response("x"))
    with pytest.raises(answering.AnswerError):
        generator("s", "u")
    assert (len(models.calls), waits) == (1, [])
