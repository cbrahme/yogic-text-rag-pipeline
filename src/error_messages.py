"""
User-facing error messages shared by the Streamlit app and the terminal chat.

Maps OpenAI SDK exceptions (and a few project errors) to short, actionable
messages, so users see what to do instead of a stack trace.
"""

import openai

from chroma_manager import IndexNotFoundError

# Fail fast instead of the OpenAI SDK's 10-minute default timeout
OPENAI_TIMEOUT_SECONDS = 60


class EmptyResponseError(RuntimeError):
    """Raised when the model returns no text."""


def friendly_error_message(exc: Exception) -> str:
    """Returns a short, actionable message describing `exc`."""
    if isinstance(exc, openai.AuthenticationError):
        return "The OpenAI API key was rejected. Check `OPENAI_API_KEY` in your `.env` file."
    if isinstance(exc, openai.PermissionDeniedError):
        return "This OpenAI API key does not have access to the required model."
    if isinstance(exc, openai.RateLimitError):
        if "insufficient_quota" in str(exc):
            return "The OpenAI account has run out of credits. Add billing/credits and try again."
        return "OpenAI is rate-limiting requests right now. Please wait a few seconds and try again."
    if isinstance(exc, openai.APITimeoutError):
        return "OpenAI took too long to respond. Please try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach OpenAI. Check your internet connection and try again."
    if isinstance(exc, openai.NotFoundError):
        return "OpenAI could not find the requested model. Check the model name."
    if isinstance(exc, openai.BadRequestError):
        return "OpenAI rejected the request (it may be too long). Try a shorter question or clear the chat memory."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI returned an error (HTTP {exc.status_code}). Please try again shortly."
    if isinstance(exc, EmptyResponseError):
        return "The model returned an empty answer. Please try again."
    if isinstance(exc, IndexNotFoundError):
        return f"The scripture index is not available. {exc}"
    return "Something went wrong while answering. Please try again."
