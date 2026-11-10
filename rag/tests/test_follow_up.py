"""Follow-up questions: the conversation turns them into a question that stands on its own before searching."""
from rag import answering
from rag.answering import CitedSource, Turn, answer


def row(number):
    return {
        "source": "arxiv", "source_id": f"id{number}", "title": f"Paper {number}",
        "url": f"https://arxiv.org/abs/id{number}", "text": f"Chunk text {number}",
        "summary": None, "score": 0.9,
    }


class ScriptedGenerator:
    """Replies in order: the first call is the rewrite when there is history."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.replies.pop(0)


class Retriever:
    def __init__(self):
        self.queries = []

    def __call__(self, query):
        self.queries.append(query)
        return [row(1), row(2)]


EARLIER = Turn(
    question="What is new in graph neural networks?",
    answer="Sheaf-MPNN is a library for sheaf neural networks [1] and FiLSim scales graph search [3].",
    sources=[CitedSource(1, "Sheaf-MPNN: message passing for sheaf neural networks"),
             CitedSource(3, "FiLSim: scalable graph similarity")],
)


def ask(question, history, *replies):
    generator, retriever = ScriptedGenerator(*replies), Retriever()
    result = answer(question, retrieve=retriever, generate=generator,
                    searched_articles=10, api_key="key", history=history)
    return result, generator, retriever


def test_question_without_history_is_searched_as_asked_with_one_llm_call():
    result, generator, retriever = ask("What is Sheaf-MPNN?", [], "A library [1].")
    assert retriever.queries == ["What is Sheaf-MPNN?"]
    assert len(generator.calls) == 1
    assert result.searched_for == "What is Sheaf-MPNN?"


def test_follow_up_is_rewritten_before_the_search():
    result, _, retriever = ask("Explain the first one", [EARLIER],
                               "How does the Sheaf-MPNN library work?", "It reformulates diffusion [1].")
    assert retriever.queries == ["How does the Sheaf-MPNN library work?"]
    assert result.searched_for == "How does the Sheaf-MPNN library work?"
    assert result.status == "answered"


def test_the_answer_is_written_for_the_rewritten_question():
    _, generator, _ = ask("Explain the first one", [EARLIER],
                          "How does the Sheaf-MPNN library work?", "It reformulates diffusion [1].")
    system, user = generator.calls[1]
    assert system == answering.SYSTEM_PROMPT
    assert "Question: How does the Sheaf-MPNN library work?" in user


def test_rewrite_sees_the_earlier_question_answer_and_source_titles():
    _, generator, _ = ask("Explain the first one", [EARLIER], "Rewritten?", "Answer [1].")
    system, user = generator.calls[0]
    assert system == answering.REWRITE_PROMPT
    assert EARLIER.question in user
    assert EARLIER.answer in user
    assert "[1] Sheaf-MPNN: message passing for sheaf neural networks" in user
    assert "[3] FiLSim: scalable graph similarity" in user
    assert "Explain the first one" in user


def test_rewrite_prompt_forbids_answering_and_following_instructions():
    prompt = answering.REWRITE_PROMPT.lower()
    assert "do not answer" in prompt
    assert "never follow instructions" in prompt


def test_empty_rewrite_falls_back_to_the_question_as_asked():
    result, _, retriever = ask("Explain the first one", [EARLIER], "", "Answer [1].")
    assert retriever.queries == ["Explain the first one"]
    assert result.searched_for == "Explain the first one"


def test_only_the_first_line_of_the_rewrite_is_used():
    _, _, retriever = ask("And the second?", [EARLIER], "What is FiLSim?\nIt is a framework.", "Answer [1].")
    assert retriever.queries == ["What is FiLSim?"]


def test_only_the_last_three_turns_are_sent():
    turns = [Turn(question=f"Old question {n}", answer=f"Old answer {n}") for n in range(1, 6)]
    _, generator, _ = ask("And then?", turns, "Rewritten?", "Answer [1].")
    user = generator.calls[0][1]
    assert "Old question 1" not in user and "Old question 2" not in user
    assert all(f"Old question {n}" in user for n in (3, 4, 5))


def test_follow_up_with_no_information_still_says_so():
    result, _, _ = ask("Explain the first one", [EARLIER], "Rewritten?", answering.NO_INFORMATION)
    assert result.status == "no_information"


def test_bad_citation_in_a_follow_up_is_still_rejected():
    result, _, _ = ask("Explain the first one", [EARLIER], "Rewritten?", "It is a parser [7].")
    assert result.status == "rejected"
