"""Which LLM writes the answers: Groq first by default, Gemini behind it, chosen with LLM_PROVIDER."""
import logging

from rag import answering
from rag.answering import AnswerError, MissingApiKey

log = logging.getLogger(__name__)

# Listed as active by GET https://api.groq.com/openai/v1/models in 11/2026. Of Groq's chat models
# it cited sources every time and answered in the question's language at about the same speed
# as gpt-oss-20b, which often left out citations and once answered in Thai. GROQ_MODEL picks another.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
PROVIDERS = ("groq", "gemini")
RESTART = answering.RESTART
ALL_FAILED_MESSAGE = "The answer service is busy right now. Please try again in a minute."


def make_groq(key, model):
    import httpx

    from rag.groq_client import GroqGenerator
    return GroqGenerator(httpx.Client(), api_key=key, model=model)


def make_gemini(key, model):
    from google import genai

    return answering.GeminiGenerator(genai.Client(api_key=key), model=model)


class FallbackGenerator:
    """Ask each provider in turn until one answers; a reader only sees the last failure as a friendly note."""

    def __init__(self, providers):
        self.providers = providers

    def __call__(self, system, user):
        for provider in self.providers:
            try:
                return provider(system, user)
            except AnswerError as error:
                log.warning("LLM %s failed, trying the next one: %s", provider.name, error)
        raise AnswerError(ALL_FAILED_MESSAGE)

    def stream(self, system, user):
        shown = False
        for provider in self.providers:
            try:
                for piece in provider.stream(system, user):
                    shown = True
                    yield piece
                return
            except AnswerError as error:
                log.warning("LLM %s failed while streaming, trying the next one: %s", provider.name, error)
                if shown:
                    # Part of an answer is on screen; the next provider writes it again from the start.
                    yield RESTART
                    shown = False
        raise AnswerError(ALL_FAILED_MESSAGE)


def build_generator(get, make_groq=make_groq, make_gemini=make_gemini):
    first = (get("LLM_PROVIDER") or "groq").strip().lower()
    if first not in PROVIDERS:
        raise MissingApiKey(f'LLM_PROVIDER must be "groq" or "gemini", not "{first}".')
    keys = {"groq": get("GROQ_API_KEY"), "gemini": get("GEMINI_API_KEY")}
    makers = {
        "groq": lambda: make_groq(keys["groq"], get("GROQ_MODEL") or DEFAULT_GROQ_MODEL),
        "gemini": lambda: make_gemini(keys["gemini"], get("GEMINI_MODEL") or answering.DEFAULT_MODEL),
    }
    order = [first] + [name for name in PROVIDERS if name != first]
    chain = [makers[name]() for name in order if keys[name]]
    if not chain:
        raise MissingApiKey(
            "No LLM key is set. Add GROQ_API_KEY or GEMINI_API_KEY to the .env file or the environment, then try again."
        )
    return FallbackGenerator(chain)
