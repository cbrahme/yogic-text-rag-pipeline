#!/usr/bin/env python3
"""Unit tests for user-facing error messages shared by the app and CLI."""

import os
import sys
import unittest

import httpx
import openai

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chroma_manager import IndexNotFoundError
from error_messages import friendly_error_message, EmptyResponseError

_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, code, message="error"):
    return cls(message, response=httpx.Response(code, request=_REQ), body=None)


class TestFriendlyErrorMessages(unittest.TestCase):

    def test_openai_errors_map_to_actionable_messages(self):
        cases = [
            (_status_error(openai.AuthenticationError, 401), "API key was rejected"),
            (_status_error(openai.PermissionDeniedError, 403), "does not have access"),
            (_status_error(openai.RateLimitError, 429), "rate-limiting"),
            (_status_error(openai.RateLimitError, 429, "insufficient_quota"), "run out of credits"),
            (openai.APITimeoutError(request=_REQ), "took too long"),
            (openai.APIConnectionError(request=_REQ), "Could not reach OpenAI"),
            (_status_error(openai.NotFoundError, 404), "could not find the requested model"),
            (_status_error(openai.BadRequestError, 400), "rejected the request"),
            (_status_error(openai.InternalServerError, 500), "HTTP 500"),
        ]
        for exc, expected in cases:
            with self.subTest(type(exc).__name__):
                self.assertIn(expected, friendly_error_message(exc))

    def test_project_and_unknown_errors(self):
        self.assertIn("empty answer", friendly_error_message(EmptyResponseError("x")))
        self.assertIn("index is not available", friendly_error_message(IndexNotFoundError("No index at /p")))
        self.assertIn("/p", friendly_error_message(IndexNotFoundError("No index at /p")))
        self.assertIn("Something went wrong", friendly_error_message(ValueError("boom")))


if __name__ == "__main__":
    unittest.main()
