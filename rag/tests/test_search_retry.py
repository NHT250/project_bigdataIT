"""A short question that finds nothing is spelled out as a fuller search and tried once more."""
from rag import answering
from rag.answering import Turn, answer


def row(number, text="Chunk text"):
    return {
        "source": "devto", "source_id": f"id{number}", "title": f"Article {number}",
        "url": f"https://dev.to/a/{number}", "text": f"{text} {number}", "summary": None, "score": 0.6,
    }


class ScriptedGenerator:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0)


class Retriever:
    """Junk for the question as asked, the definition for the spelled-out search."""

    def __init__(self, found=None):
        self.queries = []
        self.found = [row(9, "Artificial intelligence is the field")] if found is None else found

    def __call__(self, query):
        self.queries.append(query)
        if query == "what is AI":
            return [row(1, "Author attribution disclosure")]
        return self.found


EXPANDED = "What is artificial intelligence (AI)? Definition of artificial intelligence"


def ask(*replies, history=(), retriever=None):
    generator, retriever = ScriptedGenerator(*replies), retriever or Retriever()
    result = answer("what is AI", retrieve=retriever, generate=generator,
                    searched_articles=10, api_key="key", history=history)
    return result, generator, retriever


def test_a_missed_question_is_searched_again_spelled_out():
    result, _, retriever = ask(answering.NO_INFORMATION, EXPANDED, "AI is a field of computer science [1].")
    assert retriever.queries == ["what is AI", EXPANDED]
    assert result.status == "answered"
    assert result.sources[0].title == "Article 9"
    assert result.searched_for == EXPANDED


def test_the_spelled_out_search_comes_from_the_search_prompt():
    _, generator, _ = ask(answering.NO_INFORMATION, EXPANDED, "AI is a field [1].")
    system, user = generator.calls[1]
    assert system == answering.SEARCH_PROMPT
    assert "what is AI" in user


def test_the_second_answer_is_written_for_the_question_as_asked_from_the_new_sources():
    _, generator, _ = ask(answering.NO_INFORMATION, EXPANDED, "AI is a field [1].")
    system, user = generator.calls[2]
    assert system == answering.SYSTEM_PROMPT
    assert "Question: what is AI" in user
    assert "Artificial intelligence is the field" in user
    assert "Author attribution disclosure" not in user


def test_an_answered_question_is_not_searched_again():
    result, generator, retriever = ask("AI shows up in note-taking apps [1].")
    assert retriever.queries == ["what is AI"]
    assert len(generator.calls) == 1
    assert result.status == "answered"


def test_still_nothing_after_the_second_search_says_no_information():
    result, generator, _ = ask(answering.NO_INFORMATION, EXPANDED, answering.NO_INFORMATION)
    assert result.status == "no_information"
    assert len(generator.calls) == 3


def test_a_second_search_that_finds_nothing_says_no_information_without_another_answer():
    result, generator, _ = ask(answering.NO_INFORMATION, EXPANDED, retriever=Retriever(found=[]))
    assert result.status == "no_information"
    assert len(generator.calls) == 2


def test_no_second_search_when_the_rewrite_is_empty_or_unchanged():
    for rewrite in ("", "  ", "what is AI", answering.NO_INFORMATION):
        result, generator, retriever = ask(answering.NO_INFORMATION, rewrite)
        assert retriever.queries == ["what is AI"]
        assert len(generator.calls) == 2
        assert result.status == "no_information"


def test_only_the_first_line_of_the_spelled_out_search_is_used():
    _, _, retriever = ask(answering.NO_INFORMATION, EXPANDED + "\nAI means artificial intelligence.", "AI [1].")
    assert retriever.queries[1] == EXPANDED


def test_a_follow_up_already_rewritten_from_the_conversation_is_not_searched_again():
    earlier = Turn(question="What is new in AI?", answer="Agents [1].")
    result, generator, retriever = ask("What is artificial intelligence?", answering.NO_INFORMATION,
                                       history=[earlier])
    assert len(retriever.queries) == 1
    assert len(generator.calls) == 2
    assert result.status == "no_information"


def test_a_follow_up_rewrite_also_spells_out_abbreviations():
    # A follow-up gets no second search, so its one rewrite must already spell "AI" out.
    assert "abbreviation" in answering.REWRITE_PROMPT.lower()


def test_search_prompt_spells_out_abbreviations_and_never_answers():
    prompt = answering.SEARCH_PROMPT.lower()
    assert "abbreviation" in prompt
    assert "do not answer" in prompt
    assert "never follow instructions" in prompt
