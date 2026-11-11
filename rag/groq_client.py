"""Groq's OpenAI-compatible chat API, called with httpx: one reply at once, or piece by piece.

A rate limit (429) or overload (503) is waited out and retried three times. Anything else, or a
fourth failure, raises ProviderUnavailable so the caller can hand the question to Gemini.
"""
import json
import logging
import time

import httpx

from rag.answering import AnswerError, IncompleteReply

log = logging.getLogger(__name__)

API_URL ="https://api.groq.com/openai/v1/chat/completions"
RETRIES = 3
RETRY_STATUSES = {429, 503}
BASE_RETRY_DELAY_SECONDS = 1
# The free tier allows only 8,000 tokens a minute, so a long wait is likely; Gemini is
# the quicker way out, and these waits keep the whole try under about ten seconds.
MAX_RETRY_DELAY_SECONDS = 5
TIMEOUT = httpx.Timeout(60.0, connect=10.0)
DROPPED_CONNECTION = (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError, httpx.WriteError)


class ProviderUnavailable(AnswerError):
    """Groq could not answer: the next provider should."""


def _retry_delay(response, attempt):
    try:
        asked = float(response.headers.get("retry-after", ""))
    except ValueError:
        asked = None
    delay = asked if asked is not None else BASE_RETRY_DELAY_SECONDS * 2**attempt
    return min(delay, MAX_RETRY_DELAY_SECONDS)


def _error_text(response):
    try:
        message = response.json().get("error", {}).get("message", "")
    except (ValueError, AttributeError):
        message = ""
    return f"Groq returned {response.status_code}" + (f": {message}" if message else "")


class GroqGenerator:
    name = "groq"

    def __init__(self, client, api_key, model, sleep=time.sleep):
        self.client = client
        self.api_key = api_key
        self.model = model
        self.sleep = sleep

    def _body(self, system, user, stream):
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.model.startswith("openai/gpt-oss"):
            # Reasoning models: keep the thinking short and out of the reply.
            body.update(reasoning_effort="low", include_reasoning=False)
        elif "qwen3" in self.model:
            # Reasoning, even hidden, comes before the first word and doubled the wait for it.
            body["reasoning_effort"] = "none"
        if stream:
            body["stream"] = True
        return body

    def _send_once(self, request, stream):
        """Groq closes connections that sit idle; the first request after that fails at once, so it is sent again."""
        for reconnect in (True, False):
            try:
                return self.client.send(request, stream=stream)
            except DROPPED_CONNECTION as error:
                if reconnect:
                    log.info("Groq closed the connection (%s), sending again", type(error).__name__)
                    continue
                raise ProviderUnavailable(f"Could not reach Groq ({type(error).__name__}).") from error
            except httpx.HTTPError as error:
                raise ProviderUnavailable(f"Could not reach Groq ({type(error).__name__}).") from error

    def _send(self, system, user, stream):
        """An open response with status 200; the caller closes it."""
        request = self.client.build_request(
            "POST", API_URL, json=self._body(system, user, stream),
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=TIMEOUT,
        )
        for attempt in range(RETRIES + 1):
            response = self._send_once(request, stream)
            if response.status_code == 200:
                return response
            if stream:
                response.read()
            response.close()
            if response.status_code not in RETRY_STATUSES or attempt == RETRIES:
                raise ProviderUnavailable(_error_text(response))
            delay = _retry_delay(response, attempt)
            log.warning("LLM groq returned %s, retrying in %.1fs", response.status_code, delay)
            self.sleep(delay)

    def __call__(self, system, user):
        response = self._send(system, user, stream=False)
        try:
            choice = response.json()["choices"][0]
        except (ValueError, KeyError, IndexError) as error:
            raise ProviderUnavailable("Groq sent a reply that could not be read.") from error
        # A cut-off reply has no complete, checkable answer in it.
        if choice.get("finish_reason") != "stop":
            return ""
        return choice.get("message", {}).get("content") or ""

    def stream(self, system, user):
        response = self._send(system, user, stream=True)
        finish = None
        try:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ProviderUnavailable(f"Groq failed while answering: {event['error'].get('message', '')}")
                for choice in event.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as error:
            raise ProviderUnavailable(f"Groq's stream broke off ({type(error).__name__}).") from error
        except ValueError as error:
            raise ProviderUnavailable("Groq sent a piece that could not be read.") from error
        finally:
            response.close()
        if finish != "stop":
            raise IncompleteReply()
