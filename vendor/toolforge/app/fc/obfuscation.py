"""Optional tool-name obfuscation for prompt FC (Phase 3)."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..models.canonical import ToolCall, ToolDef

# Client-facing name → safe prompt name
SAFE_ALIASES: Dict[str, str] = {
    "Read": "fs_open_file",
    "Write": "fs_put_file",
    "Edit": "fs_patch_file",
    "Bash": "shell_run",
    "Grep": "text_search",
    "Glob": "path_find",
    "NotebookEdit": "notebook_patch",
    "WebFetch": "http_get_url",
    "WebSearch": "web_query",
}

REVERSE_ALIASES: Dict[str, str] = {v: k for k, v in SAFE_ALIASES.items()}

_SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")


def to_safe_name(name: str) -> str:
    if name in SAFE_ALIASES:
        return SAFE_ALIASES[name]
    if name in REVERSE_ALIASES:
        return name
    cleaned = _SAFE_RE.sub("_", name).strip("_")
    if not cleaned:
        cleaned = "tool"
    if cleaned[0].isupper() or any(ch.isupper() for ch in cleaned[1:]):
        return f"u_{cleaned}"
    return cleaned


def from_safe_name(name: str, original_names: Dict[str, str] | None = None) -> str:
    if original_names and name in original_names:
        return original_names[name]
    if name in REVERSE_ALIASES:
        return REVERSE_ALIASES[name]
    if name.startswith("u_") and len(name) > 2:
        return name[2:]
    return name


def obfuscate_tools(tools: List[ToolDef]) -> Tuple[List[ToolDef], Dict[str, str]]:
    """Return tools with safe names + map safe→original."""
    mapping: Dict[str, str] = {}
    out: List[ToolDef] = []
    for tool in tools:
        safe = to_safe_name(tool.name)
        mapping[safe] = tool.name
        out.append(
            ToolDef(
                name=safe,
                description=tool.description,
                parameters=tool.parameters,
                raw={},
            )
        )
    return out, mapping


def deobfuscate_calls(calls: List[ToolCall], mapping: Dict[str, str]) -> List[ToolCall]:
    restored: List[ToolCall] = []
    for call in calls:
        restored.append(
            ToolCall(
                id=call.id,
                name=from_safe_name(call.name, mapping),
                arguments=call.arguments,
            )
        )
    return restored
