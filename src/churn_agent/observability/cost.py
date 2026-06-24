"""Estimacion de tokens y coste por sesion.

Precios de Claude Haiku 4.5 (claude-haiku-4-5-20251001):
  Input:  $0.80 / 1M tokens
  Output: $4.00 / 1M tokens

La estimacion usa ~4 caracteres por token, que es una aproximacion
conservadora para texto en espanol (tiende a subestimar ligeramente).
"""

from __future__ import annotations

from typing import Any

_INPUT_USD_PER_TOKEN: float = 0.80 / 1_000_000
_OUTPUT_USD_PER_TOKEN: float = 4.00 / 1_000_000
_CHARS_PER_TOKEN: int = 4


def _message_chars(msg: Any) -> tuple[int, int]:
    """Devuelve (input_chars, output_chars) para un mensaje LangChain."""
    content = getattr(msg, "content", "") or ""
    chars = len(content) if isinstance(content, str) else 0

    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        if isinstance(tc, dict):
            chars += len(str(tc.get("args", "")))

    if "AI" in type(msg).__name__:
        return 0, chars
    return chars, 0


def estimate_session_cost(messages: list[Any]) -> tuple[int, int, float]:
    """Estima tokens y coste para los mensajes de una sesion del agente.

    Returns:
        (input_tokens, output_tokens, cost_usd)
    """
    input_chars = 0
    output_chars = 0
    for msg in messages:
        ic, oc = _message_chars(msg)
        input_chars += ic
        output_chars += oc

    input_tokens = max(1, input_chars // _CHARS_PER_TOKEN)
    output_tokens = max(1, output_chars // _CHARS_PER_TOKEN)
    cost_usd = (
        input_tokens * _INPUT_USD_PER_TOKEN + output_tokens * _OUTPUT_USD_PER_TOKEN
    )
    return input_tokens, output_tokens, round(cost_usd, 8)
