"""Rough token estimation without hard tiktoken dependency."""

from __future__ import annotations

from typing import Any


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return 0
    # ~4 chars per token heuristic
    return max(1, (len(text) + 3) // 4)
