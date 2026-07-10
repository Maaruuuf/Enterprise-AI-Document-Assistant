"""
LLM answer generation via Groq, with multi-model fallback.

If the primary model fails (rate limit, timeout, decommissioned, etc.), the
service automatically retries with each fallback model in order before
giving up. This is a resilience layer, not a quality layer — all models in
the chain are instructed identically via the same system/user prompt.
"""

import logging
from typing import List, Optional, Tuple

from groq import Groq, GroqError

from app.core.config import settings
from app.models.schemas import ConversationTurn, RetrievedChunk
from app.prompts.templates import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Raised when ALL models in the fallback chain fail to produce an answer."""


_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    """Return a shared Groq client instance."""
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def _build_messages(
    question: str,
    reranked_chunks: List[RetrievedChunk],
    history: Optional[List[ConversationTurn]] = None,
) -> list:
    """Assemble the full message list: system prompt, recent history, then
    the current question with retrieved context.

    Args:
        question: Current user question.
        reranked_chunks: Retrieved context chunks for the current question.
        history: Recent prior turns in this session, for follow-up resolution.

    Returns:
        List of {role, content} dicts for the Groq chat completion API.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in (history or []):
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer})

    user_prompt = build_user_prompt(question, reranked_chunks)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_answer(
    question: str,
    reranked_chunks: List[RetrievedChunk],
    history: Optional[List[ConversationTurn]] = None,
) -> Tuple[str, str]:
    """Generate a grounded answer, trying each model in the fallback chain
    in order until one succeeds.

    Args:
        question: The user's raw question.
        reranked_chunks: Top reranked context chunks.
        history: Recent conversation turns for this session (optional).

    Returns:
        Tuple of (answer_text, model_name_used).

    Raises:
        LLMGenerationError: If every model in the fallback chain fails.
    """
    client = get_groq_client()
    messages = _build_messages(question, reranked_chunks, history)
    model_chain = [settings.GROQ_MODEL_PRIMARY, *settings.GROQ_MODEL_FALLBACKS]

    last_error: Optional[Exception] = None

    for model_name in model_chain:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            if model_name != settings.GROQ_MODEL_PRIMARY:
                logger.warning(f"Primary model failed earlier — answered using fallback model '{model_name}'")
            return answer, model_name

        except GroqError as e:
            logger.warning(f"Model '{model_name}' failed: {e}. Trying next in fallback chain...")
            last_error = e
            continue
        except Exception as e:
            # Non-Groq errors (network, serialization) — still try next model rather than crash
            logger.error(f"Unexpected error with model '{model_name}': {e}")
            last_error = e
            continue

    raise LLMGenerationError(
        f"All {len(model_chain)} model(s) in the fallback chain failed. Last error: {last_error}"
    )