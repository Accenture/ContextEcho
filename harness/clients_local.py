"""Explicit connection to a self-hosted OpenAI-compatible chat endpoint."""

import os

from openai import OpenAI


def make_local_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["LOCAL_OPENAI_BASE_URL"],
        api_key=os.environ.get("LOCAL_OPENAI_API_KEY", "local"),
    )
