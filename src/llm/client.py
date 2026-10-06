"""
Thin wrapper around the SSP Cloud LLM lab endpoint.

Mirrors the calling convention used in
https://github.com/SSPHub/tchap_bot_llm/blob/main/src/listeners/llm_tchat.py :
an OpenAI-compatible client pointed at https://llm.lab.sspcloud.fr/api, the
LLM_LAB_API_KEY environment variable, and a chat-completion call.

Endpoint and model can be overridden through environment variables so the
script keeps working if the lab changes its default model:
    - LLM_LAB_API_KEY   : API key (required)
    - LLM_LAB_ENDPOINT  : base url   (default https://llm.lab.sspcloud.fr/api)
    - LLM_MODEL_NAME    : model name (default qwen3-8-27b)
"""

import contextlib
import json
import os
from typing import Any

from openai import OpenAI

DEFAULT_ENDPOINT = "https://llm.lab.sspcloud.fr/api"
DEFAULT_MODEL = "qwen3-8-27b"


def langfuse_enabled() -> bool:
    """Langfuse n'est utilisé que pour le tracing automatique : sans clés il
    s'initialise, affiche un avertissement puis se désactive. On évite donc
    l'import (~1 s) et le message quand il n'est pas configuré.
    """

    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _client_class() -> type[OpenAI]:
    """Wrapper Langfuse si les clés sont configurées, client OpenAI sinon."""

    if langfuse_enabled():
        from langfuse.openai import OpenAI as LangfuseOpenAI

        return LangfuseOpenAI
    return OpenAI


def trace_attributes(**attributes: Any) -> contextlib.AbstractContextManager:
    """Étiqueter les traces Langfuse des appels faits dans ce contexte.

    Arguments de `langfuse.propagate_attributes` (session_id, tags, metadata,
    trace_name…), valeurs US-ASCII de 200 caractères au plus ; les valeurs None
    sont ignorées. Sans Langfuse configuré, contexte vide.
    """

    if not langfuse_enabled():
        return contextlib.nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(
        **{name: value for name, value in attributes.items() if value is not None}
    )


def flush_traces() -> None:
    """Envoyer les traces Langfuse en attente (fin de batch)."""

    if langfuse_enabled():
        from langfuse import get_client as get_langfuse_client

        get_langfuse_client().flush()


def get_client() -> OpenAI:
    return _client_class()(
        base_url=os.environ.get("LLM_LAB_ENDPOINT", DEFAULT_ENDPOINT),
        api_key=os.environ.get("LLM_LAB_API_KEY", ""),
        max_retries=5
    )


def get_model_name() -> str:
    return os.environ.get("LLM_MODEL_NAME", DEFAULT_MODEL)


def ask(
    messages: list,
    client: OpenAI | None = None,
    reasoning: bool = True,
    **kwargs,
) -> str:
    """
    Send a list of chat messages and return the assistant's text answer.

    Args:
        messages: list of {"role": ..., "content": ...} dicts.
        client: an optional pre-built OpenAI client (handy for tests / reuse).
        reasoning: True keeps the model's default behaviour (qwen3 reasons
            before answering); False sends reasoning_effort="none", which the
            lab honours for qwen3-8-27b and gemma4-26b-moe.
        kwargs: forwarded to chat.completions.create (e.g. temperature).

    Returns:
        the model answer as a string.
    """
    if not reasoning:
        kwargs.setdefault("reasoning_effort", "none")
    client = client or get_client()
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content


def parse_json_answer(raw: str) -> dict:
    """
    Parse a JSON object out of a model answer, tolerating ```json fences and
    surrounding prose. Returns {} if no valid object can be recovered.

    Example:
        >>> parse_json_answer('```json\\n{"a": 1}\\n```')
        {'a': 1}
    """
    if not raw:
        return {}

    cleaned = raw.strip()
    # Drop Markdown code fences if the model wrapped its answer.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove an eventual leading "json" language hint
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} block.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start: end + 1])
            except json.JSONDecodeError:
                return {}
        return {}


def ask_json(messages: list, client: OpenAI | None = None, **kwargs) -> dict:
    """Convenience: ask() then parse_json_answer()."""
    return parse_json_answer(ask(messages, client=client, **kwargs))
