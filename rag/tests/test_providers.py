"""Choosing the LLM (Groq by default, Gemini as the fallback) and falling back when one fails."""
import pytest

from rag import answering, providers
from rag.answering import AnswerError, IncompleteReply, MissingApiKey
from rag.groq_client import ProviderUnavailable


class Fake:
    """A provider that replies, or fails with the given error, optionally after some pieces."""

    def __init__(self, name, reply="Answer [1].", fail=None, pieces_before_failing=()):
        self.name = name
        self.reply = reply
        self.fail = fail
        self.pieces_before_failing = list(pieces_before_failing)
        self.calls = 0

    def __call__(self, system, user):
        self.calls += 1
        if self.fail:
            raise self.fail
        return self.reply

    def stream(self, system, user):
        self.calls += 1
        yield from self.pieces_before_failing
        if self.fail:
            raise self.fail
        yield self.reply


def settings_with(**values):
    return lambda name, default="": values.get(name, default)


def build(**values):
    made = []

    def make(name):
        def factory(key, model):
            made.append((name, key, model))
            return Fake(name)
        return factory

    chain = providers.build_generator(get=settings_with(**values), make_groq=make("groq"), make_gemini=make("gemini"))
    return chain, made


# Which provider answers


def test_groq_answers_first_by_default_with_gemini_behind_it():
    chain, made = build(GROQ_API_KEY="g", GEMINI_API_KEY="m")
    assert [p.name for p in chain.providers] == ["groq", "gemini"]
    assert made[0] == ("groq", "g", providers.DEFAULT_GROQ_MODEL)


def test_llm_provider_gemini_puts_gemini_first():
    chain, _ = build(LLM_PROVIDER="gemini", GROQ_API_KEY="g", GEMINI_API_KEY="m")
    assert [p.name for p in chain.providers] == ["gemini", "groq"]


def test_only_providers_with_a_key_are_used():
    chain, _ = build(GROQ_API_KEY="g")
    assert [p.name for p in chain.providers] == ["groq"]
    chain, _ = build(GEMINI_API_KEY="m")
    assert [p.name for p in chain.providers] == ["gemini"]


def test_model_names_come_from_the_environment_when_set():
    _, made = build(GROQ_API_KEY="g", GROQ_MODEL="qwen/qwen3.8-27b", GEMINI_API_KEY="m", GEMINI_MODEL="gemini-x")
    assert ("groq", "g", "qwen/qwen3.8-27b") in made
    assert ("gemini", "m", "gemini-x") in made


def test_the_default_groq_model_is_one_groq_listed_as_active():
    # Listed by GET /openai/v1/models in 11/2026. Qwen cited its sources every time and kept to the
    # question's language; gpt-oss-20b, as fast, left out citations and answered an English
    # question in Thai when a retrieved passage was Thai.
    assert providers.DEFAULT_GROQ_MODEL == "qwen/qwen3.8-27b"


def test_no_key_at_all_names_both_settings():
    with pytest.raises(MissingApiKey, match="GROQ_API_KEY.*GEMINI_API_KEY"):
        build()


def test_an_unknown_provider_name_is_a_clear_error():
    with pytest.raises(MissingApiKey, match="LLM_PROVIDER"):
        build(LLM_PROVIDER="openai", GROQ_API_KEY="g")


# Falling back


def chain_of(*fakes):
    return providers.FallbackGenerator(list(fakes))


def test_a_failed_groq_reply_is_answered_by_gemini():
    groq, gemini = Fake("groq", fail=ProviderUnavailable("Groq returned 503")), Fake("gemini", "From Gemini [1].")
    assert chain_of(groq, gemini)("s", "u") == "From Gemini [1]."


def test_a_working_groq_never_calls_gemini():
    groq, gemini = Fake("groq", "From Groq [1]."), Fake("gemini")
    assert chain_of(groq, gemini)("s", "u") == "From Groq [1]."
    assert gemini.calls == 0


def test_a_gemini_error_also_falls_through_to_the_next_provider():
    first, second = Fake("gemini", fail=AnswerError("The Gemini API returned 500")), Fake("groq", "Groq [1].")
    assert chain_of(first, second)("s", "u") == "Groq [1]."


def test_when_every_provider_fails_the_message_is_friendly_and_suggests_trying_again():
    chain = chain_of(Fake("groq", fail=ProviderUnavailable("Groq returned 503")),
                     Fake("gemini", fail=AnswerError("The Gemini API returned 500: boom")))
    with pytest.raises(AnswerError) as caught:
        chain("s", "u")
    message = str(caught.value)
    assert message == providers.ALL_FAILED_MESSAGE
    assert "try again" in message.lower()
    assert "500" not in message


def test_stream_falls_back_before_anything_was_shown():
    groq, gemini = Fake("groq", fail=ProviderUnavailable("busy")), Fake("gemini", "Gemini [1].")
    assert list(chain_of(groq, gemini).stream("s", "u")) == ["Gemini [1]."]


def test_stream_that_fails_midway_starts_the_answer_over_with_the_next_provider():
    groq = Fake("groq", fail=ProviderUnavailable("busy"), pieces_before_failing=["Half an "])
    gemini = Fake("gemini", "Whole answer [1].")
    assert list(chain_of(groq, gemini).stream("s", "u")) == ["Half an ", providers.RESTART, "Whole answer [1]."]


def test_stream_when_every_provider_fails_is_the_friendly_error():
    chain = chain_of(Fake("groq", fail=ProviderUnavailable("busy")), Fake("gemini", fail=AnswerError("down")))
    with pytest.raises(AnswerError, match="try again"):
        list(chain.stream("s", "u"))


def test_a_cut_off_reply_is_not_a_provider_failure():
    groq, gemini = Fake("groq", fail=IncompleteReply()), Fake("gemini")
    with pytest.raises(IncompleteReply):
        list(chain_of(groq, gemini).stream("s", "u"))
    assert gemini.calls == 0


# The restart reaches the page as its own event


def test_a_restarted_stream_tells_the_page_to_drop_what_it_showed():
    def stream(system, user):
        yield from ["Half an ", providers.RESTART, "Whole answer ", "[1]."]

    rows = [{"source": "arxiv", "source_id": "1", "title": "P", "url": "https://arxiv.org/abs/1",
             "text": "t", "summary": None, "score": 0.9}]
    events = list(answering.answer_events("Q?", retrieve=lambda q: rows, generate=None, stream=stream,
                                          searched_articles=1, api_key="k"))
    kinds = [e["type"] for e in events]
    assert kinds == ["searching", "delta", "restart", "delta", "delta", "done"]
    assert events[-1]["answer"].text == "Whole answer [1]."
