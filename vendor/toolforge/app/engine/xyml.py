"""Protocol-agnostic helpers for LLM tool calling.

The implementation intentionally depends only on the Python standard library so
applications can use it in lightweight clients, workers, and serverless jobs.
"""

from __future__ import annotations

import inspect
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


DEFAULT_RAW_STRING_PARAMS: Set[str] = {
    "content",
    "command",
    "cmd",
    "script",
    "code",
    "prompt",
    "file_content",
    "old_string",
    "new_string",
    "insert_text",
    "patch",
    "pattern",
    "text",
    "query",
    "url",
    "path",
    "file_path",
}

DEFAULT_TOOL_ALIASES: Dict[str, str] = {
    "fs_open_file": "Read",
    "fs_put_file": "Write",
    "fs_patch_file": "Edit",
    "shell_run": "Bash",
    "text_search": "Grep",
    "path_find": "Glob",
    "notebook_patch": "NotebookEdit",
    "http_get_url": "WebFetch",
    "web_query": "WebSearch",
}

SAFE_TOOL_ALIASES: Dict[str, str] = {
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

MARKUP_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    ("＜", "<"),
    ("＞", ">"),
    ("／", "/"),
    ("∕", "/"),
    ("⁄", "/"),
    ("＝", "="),
    ("｜", "|"),
    ("│", "|"),
    ("┃", "|"),
    ("▏", "|"),
    ("▕", "|"),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
    ("﹤", "<"),
    ("﹥", ">"),
)


def random_id(length: int = 12) -> str:
    """Return a compact lowercase identifier suitable for local call IDs."""

    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def random_call_id() -> str:
    return "call_{}".format(random_id(12))


@dataclass
class ParsedToolCall:
    """A normalized function call extracted from model output."""

    name: str
    input: Any = field(default_factory=dict)
    id: str = field(default_factory=random_call_id)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = random_call_id()
        if self.input is None:
            self.input = {}


ToolCall = ParsedToolCall


class ProtocolSpec:
    """Names and tags for one markup tool-call protocol."""

    def __init__(
        self,
        name: str,
        parse_only: bool = False,
        tags: Optional[Mapping[str, str]] = None,
        **options: Any,
    ) -> None:
        if "parseOnly" in options:
            parse_only = bool(options.pop("parseOnly"))
        if not isinstance(name, str) or not name.strip():
            raise TypeError("ProtocolSpec name must be a non-empty string")
        supplied_tags = dict(tags or options.pop("tags", {}) or {})
        self.name = name.strip()
        self.parse_only = bool(parse_only)
        self.tags = {
            "root": supplied_tags.get("root", "tool_calls"),
            "invoke": supplied_tags.get("invoke", "invoke"),
            "parameter": supplied_tags.get("parameter", "parameter"),
        }


class ToolCallConfig:
    """Parser, compatibility, and validation policies for tool calls.

    Snake-case option names are preferred in Python. JavaScript SDK option names
    are also accepted so a shared JSON configuration can be reused unchanged.
    """

    def __init__(self, options: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        values: Dict[str, Any] = dict(options or {})
        values.update(kwargs)

        self.emit_protocol = str(
            _take_option(
                values,
                "emit_protocol",
                "emitProtocol",
                "default_protocol",
                "defaultProtocol",
                default="XYML",
            )
            or "XYML"
        ).strip()
        protocols = _take_option(values, "parse_protocols", "parseProtocols", "protocols")
        if protocols is None:
            protocols = [
                ProtocolSpec(self.emit_protocol),
                ProtocolSpec("QNML", parse_only=True),
            ]
        self.parse_protocols = _normalize_protocol_specs(protocols, self.emit_protocol)
        self.strict = bool(_take_option(values, "strict", default=False))
        self.unknown_tool = _take_option(values, "unknown_tool", "unknownTool", default="drop")
        self.missing_required = _take_option(
            values, "missing_required", "missingRequired", default="drop"
        )
        self.enable_markup = bool(
            _take_option(values, "enable_markup", "enableMarkup", default=True)
        )
        self.enable_xml = bool(_take_option(values, "enable_xml", "enableXml", default=True))
        self.enable_json = bool(_take_option(values, "enable_json", "enableJson", default=True))
        self.enable_text_kv = bool(
            _take_option(values, "enable_text_kv", "enableTextKV", default=True)
        )
        self.enable_coercion = bool(
            _take_option(values, "enable_coercion", "enableCoercion", default=True)
        )
        self.enable_dedupe = bool(
            _take_option(values, "enable_dedupe", "enableDedupe", default=True)
        )
        self.prompt_style = _take_option(values, "prompt_style", "promptStyle", default="standard")
        self.tool_aliases = dict(DEFAULT_TOOL_ALIASES)
        self.tool_aliases.update(
            dict(_take_option(values, "tool_aliases", "toolAliases", default={}) or {})
        )
        self.argument_aliases = dict(
            _take_option(values, "argument_aliases", "argumentAliases", default={}) or {}
        )
        custom_raw = _take_option(values, "raw_string_params", "rawStringParams", default=[]) or []
        self.raw_string_params = set(DEFAULT_RAW_STRING_PARAMS)
        self.raw_string_params.update(str(value).lower() for value in custom_raw)
        self.id_factory = _take_option(values, "id_factory", "idFactory", default=random_call_id)
        if not callable(self.id_factory):
            raise TypeError("id_factory must be callable")

    @classmethod
    def default(cls) -> "ToolCallConfig":
        return cls()

    def with_overrides(self, overrides: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> "ToolCallConfig":
        values: Dict[str, Any] = {
            "emit_protocol": self.emit_protocol,
            "parse_protocols": self.parse_protocols,
            "strict": self.strict,
            "unknown_tool": self.unknown_tool,
            "missing_required": self.missing_required,
            "enable_markup": self.enable_markup,
            "enable_xml": self.enable_xml,
            "enable_json": self.enable_json,
            "enable_text_kv": self.enable_text_kv,
            "enable_coercion": self.enable_coercion,
            "enable_dedupe": self.enable_dedupe,
            "prompt_style": self.prompt_style,
            "tool_aliases": self.tool_aliases,
            "argument_aliases": self.argument_aliases,
            "raw_string_params": list(self.raw_string_params),
            "id_factory": self.id_factory,
        }
        values.update(dict(overrides or {}))
        values.update(kwargs)
        return ToolCallConfig(values)

    with_ = with_overrides


class ToolCallEngine:
    """Stateful facade around the standalone parser and renderer functions."""

    def __init__(
        self,
        options: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
        *,
        config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            source: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = config
        elif isinstance(options, Mapping) and "config" in options:
            source = options["config"]
        else:
            source = options
        if kwargs:
            merged = dict(source or {}) if isinstance(source, Mapping) else {}
            merged.update(kwargs)
            source = merged
        self.config = _resolve_config(source)

    @classmethod
    def default(cls) -> "ToolCallEngine":
        return cls()

    @classmethod
    def with_protocols(
        cls,
        emit: str = "XYML",
        parse: Optional[Sequence[Union[str, ProtocolSpec, Mapping[str, Any]]]] = None,
        **options: Any,
    ) -> "ToolCallEngine":
        parse = parse or ["XYML", "QNML"]
        protocols: List[ProtocolSpec] = []
        for item in parse:
            if isinstance(item, ProtocolSpec):
                protocols.append(item)
            else:
                protocols.append(ProtocolSpec(str(item), parse_only=str(item) != emit))
        options["emit_protocol"] = emit
        options["parse_protocols"] = protocols
        return cls(options)

    def normalize_tools(self, tools: Any) -> List[Dict[str, Any]]:
        return normalize_tools(tools)

    def build_instructions(
        self,
        tools: Any,
        *,
        config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
        protocol: Optional[Union[str, ProtocolSpec, Mapping[str, Any]]] = None,
    ) -> str:
        return build_tool_instructions(tools, config=config or self.config, protocol=protocol)

    def parse(
        self,
        text: Any,
        tools: Any,
        *,
        config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
    ) -> List[ParsedToolCall]:
        return parse_tool_calls(text, tools, config=config or self.config)

    def render(
        self,
        call_or_name: Union[ParsedToolCall, Mapping[str, Any], str],
        input: Any = None,
        *,
        config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
        protocol: Optional[Union[str, ProtocolSpec, Mapping[str, Any]]] = None,
    ) -> str:
        active_config = config or self.config
        if isinstance(call_or_name, str):
            return render_tool_call(call_or_name, input, config=active_config, protocol=protocol)
        return render_tool_call(
            _call_value(call_or_name, "name"),
            _call_value(call_or_name, "input", {}),
            config=active_config,
            protocol=protocol,
        )

    def create_sieve(
        self,
        tools: Any,
        *,
        hold_length: int = 96,
        config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
    ) -> "ToolSieve":
        return ToolSieve(tools, config=config or self.config, hold_length=hold_length)


def normalize_tools(value: Any) -> List[Dict[str, Any]]:
    """Normalize OpenAI function specifications and plain tool specifications."""

    out: List[Dict[str, Any]] = []
    for raw in _as_list(value):
        if not _is_mapping(raw):
            continue
        if raw.get("type") == "function" and _is_mapping(raw.get("function")):
            out.append(dict(raw["function"]))
        elif isinstance(raw.get("name"), str) and raw["name"].strip():
            out.append(dict(raw))
    return out


def build_tool_instructions(
    tools: Any,
    *,
    config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
    protocol: Optional[Union[str, ProtocolSpec, Mapping[str, Any]]] = None,
) -> str:
    config = _resolve_config(config)
    active_protocol = _normalize_protocol_spec(protocol or config.emit_protocol)
    normalized_tools = normalize_tools(tools)
    safe_tools = [dict(tool, name=_safe_tool_name(tool.get("name"))) for tool in normalized_tools]
    names = [tool["name"] for tool in safe_tools if tool.get("name")]
    schemas = []
    for tool in safe_tools:
        parameters = tool.get("parameters") or tool.get("input_schema") or {}
        schemas.append(
            "\n".join(
                (
                    "Action name: {}".format(tool["name"]),
                    "Description: {}".format(_clip(tool.get("description", ""), 240)),
                    "Parameters: {}".format(_summarize_schema(parameters)),
                )
            )
        )
    example_tools = safe_tools[:2] or [
        {
            "name": "TOOL_NAME",
            "parameters": {
                "type": "object",
                "properties": {"ARG": {"type": "string"}},
            },
        }
    ]
    examples = "\n\n".join(
        render_tool_call(
            tool["name"],
            _example_input_from_tool(tool),
            config=config,
            protocol=active_protocol,
        )
        for tool in example_tools
    )
    accepted = ", ".join(spec.name for spec in config.parse_protocols)
    schema_block = "You have access to these tools:\n\n{}\n\n".format("\n\n".join(schemas)) if schemas else ""
    defensive_rules = ""
    if config.prompt_style != "minimal":
        defensive_rules = """
RULES:
1. If a tool is needed, output a parseable {name} tool-call block. If no tool is needed, answer normally.
2. Use exact action names and parameter names from the schema.
3. Strings should use <![CDATA[...]]>; objects may use JSON or nested XML-like values; arrays may use JSON arrays or repeated <item> nodes.
4. Never emit empty required parameters. Ask normally if required information is unknown.
5. After a tool result, call another tool only if needed; otherwise answer normally.
6. Path-like parameters must contain only the path string, not prose or protocol fragments.
""".format(name=active_protocol.name)
    rendered_format = render_tool_call(
        "TOOL_NAME", {"ARG": "value"}, config=config, protocol=active_protocol
    )
    return """=== {name} TOOL CALL PROTOCOL ===
{schema_block}Default protocol for new tool calls: {name}
Accepted parse protocols by this client: {accepted}
Available action names: {names}

FORMAT:
{rendered_format}
{defensive_rules}
CORRECT EXAMPLES:

{examples}

Remember: the preferred tool-call form is <|{name}|tool_calls>...</|{name}|tool_calls>.
=== END {name} TOOL INSTRUCTIONS ===""".format(
        name=active_protocol.name,
        schema_block=schema_block,
        accepted=accepted,
        names=", ".join(names),
        rendered_format=rendered_format,
        defensive_rules=defensive_rules,
        examples=examples,
    )


def render_tool_call(
    name: Any,
    input: Any = None,
    *,
    config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
    protocol: Optional[Union[str, ProtocolSpec, Mapping[str, Any]]] = None,
) -> str:
    """Render one call in the configured markup protocol."""

    config = _resolve_config(config)
    active_protocol = _normalize_protocol_spec(protocol or config.emit_protocol)
    call_name = str(name or "").strip()
    if not call_name:
        return ""
    arguments = dict(input) if _is_mapping(input) else {"input": input}
    protocol_name = active_protocol.name
    root = active_protocol.tags["root"]
    invoke = active_protocol.tags["invoke"]
    parameter = active_protocol.tags["parameter"]
    lines = [
        "<|{}|{}>".format(protocol_name, root),
        '  <|{}|{} name="{}">'.format(protocol_name, invoke, _escape_xml(call_name)),
    ]
    for key in sorted(arguments, key=lambda item: str(item)):
        lines.append(
            '    <|{}|{} name="{}">{}</|{}|{}>'.format(
                protocol_name,
                parameter,
                _escape_xml(key),
                _render_markup_value(arguments[key]),
                protocol_name,
                parameter,
            )
        )
    lines.extend(
        (
            "  </|{}|{}>".format(protocol_name, invoke),
            "</|{}|{}>".format(protocol_name, root),
        )
    )
    return "\n".join(lines)


def render_tool_calls(
    calls: Any,
    *,
    config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
    protocol: Optional[Union[str, ProtocolSpec, Mapping[str, Any]]] = None,
) -> str:
    return "\n\n".join(
        rendered
        for rendered in (
            render_tool_call(
                _call_value(call, "name"),
                _call_value(call, "input", {}),
                config=config,
                protocol=protocol,
            )
            for call in _as_list(calls)
        )
        if rendered
    )


def parse_tool_calls(
    text: Any,
    tools: Any = None,
    *,
    config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
) -> List[ParsedToolCall]:
    """Extract normalized calls from markup, JSON, XML, or text key/value output."""

    config = _resolve_config(config)
    normalized_tools = normalize_tools(tools)
    if not str(text or "").strip() or not normalized_tools:
        return []
    allowed = _build_allowed_tool_map(normalized_tools, config)
    calls: List[ParsedToolCall] = []
    if config.enable_markup:
        for protocol in config.parse_protocols:
            calls.extend(_parse_protocol_markup(text, protocol, allowed, normalized_tools, config))
    if config.enable_xml:
        calls.extend(_parse_xml_tool_calls(text, allowed, config))
    if config.enable_json:
        _for_each_json_fragment(
            text,
            lambda value: calls.extend(_parse_json_tool_calls(value, allowed, config)),
        )
    if config.enable_text_kv:
        calls.extend(_parse_text_kv_tool_calls(text, allowed, normalized_tools, config))
    fixed = calls
    if config.enable_coercion:
        fixed = [
            parsed
            for parsed in (_coerce_parsed_call(call, normalized_tools, config) for call in calls)
            if parsed is not None
        ]
    return _dedupe_tool_calls(fixed) if config.enable_dedupe else fixed


def parse_markup_tool_calls(
    text: Any,
    tools: Any = None,
    *,
    config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
    protocols: Optional[Sequence[Union[str, ProtocolSpec, Mapping[str, Any]]]] = None,
) -> List[ParsedToolCall]:
    config = _resolve_config(config)
    normalized_tools = normalize_tools(tools)
    allowed = _build_allowed_tool_map(normalized_tools, config)
    active_protocols = (
        _normalize_protocol_specs(protocols, config.emit_protocol)
        if protocols is not None
        else config.parse_protocols
    )
    calls: List[ParsedToolCall] = []
    for protocol in active_protocols:
        calls.extend(_parse_protocol_markup(text, protocol, allowed, normalized_tools, config))
    fixed = [
        parsed
        for parsed in (_coerce_parsed_call(call, normalized_tools, config) for call in calls)
        if parsed is not None
    ]
    return _dedupe_tool_calls(fixed) if config.enable_dedupe else fixed


def coerce_tool_input(
    name: str,
    input: Any,
    tools: Any = None,
    *,
    config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
) -> Any:
    """Apply schema coercion and common aliases to a parsed call input."""

    config = _resolve_config(config)
    fixed = _coerce_tool_input_by_schema(name, input, normalize_tools(tools))
    if not _is_mapping(fixed):
        return fixed
    fixed = dict(fixed)
    aliases = config.argument_aliases.get(name, {}) if _is_mapping(config.argument_aliases) else {}
    for canonical, alternate_names in dict(aliases).items():
        _rename_first_present(fixed, canonical, *_as_list(alternate_names))
    if name == "AskUserQuestion":
        if fixed.get("question") is not None and fixed.get("questions") is None:
            fixed["questions"] = [
                {
                    "question": fixed["question"],
                    "header": "Question",
                    "multiSelect": False,
                    "options": [
                        {"label": "Yes", "description": "Confirm"},
                        {"label": "No", "description": "Decline"},
                    ],
                }
            ]
            fixed.pop("question", None)
        if fixed.get("questions") is not None and not isinstance(fixed["questions"], list):
            fixed["questions"] = [fixed["questions"]]
    elif name == "Agent":
        fixed.setdefault("description", "Execute sub-task")
        fixed.setdefault("prompt", fixed["description"])
    elif name == "Read":
        _rename_first_present(fixed, "file_path", "path", "filename", "file")
    elif name == "Write":
        _rename_first_present(fixed, "file_path", "path", "target_file", "filename", "file")
        _rename_first_present(
            fixed,
            "content",
            "text",
            "body",
            "data",
            "file_content",
            "contents",
            "value",
        )
    elif name == "Edit":
        _rename_first_present(fixed, "file_path", "path", "target_file", "filename", "file")
    elif name in {"Bash", "PowerShell"}:
        _rename_first_present(fixed, "command", "cmd", "script")
    elif fixed.get("query") is None and fixed.get("queries") is not None and _tool_accepts_field(
        name, tools, "query"
    ):
        queries = fixed.pop("queries")
        if isinstance(queries, list):
            fixed["query"] = "\n".join(str(value) for value in queries if str(value))
        else:
            fixed["query"] = str(queries).strip()
    return fixed


def openai_tool_calls(calls: Any) -> List[Dict[str, Any]]:
    return [
        {
            "id": _call_value(call, "id"),
            "type": "function",
            "function": {
                "name": _call_value(call, "name"),
                "arguments": _arguments_string(_call_value(call, "input", {})),
            },
        }
        for call in _as_list(calls)
    ]


def responses_tool_items(calls: Any) -> List[Dict[str, Any]]:
    return [
        {
            "id": "fc_{}".format(random_id(12)),
            "type": "function_call",
            "status": "completed",
            "call_id": _call_value(call, "id"),
            "name": _call_value(call, "name"),
            "arguments": _arguments_string(_call_value(call, "input", {})),
        }
        for call in _as_list(calls)
    ]


def anthropic_tool_use_blocks(calls: Any) -> List[Dict[str, Any]]:
    return [
        {
            "type": "tool_use",
            "id": _call_value(call, "id"),
            "name": _call_value(call, "name"),
            "input": _call_value(call, "input", {}),
        }
        for call in _as_list(calls)
    ]


class ToolSieve:
    """Split streamed content from textual tool-call envelopes."""

    def __init__(
        self,
        tools: Any = None,
        *,
        config: Optional[Union[ToolCallConfig, Mapping[str, Any]]] = None,
        hold_length: int = 96,
    ) -> None:
        self.config = _resolve_config(config)
        self.tools = normalize_tools(tools)
        self.pending = ""
        self.capture = ""
        self.capturing = False
        self.hold_length = hold_length

    def process_chunk(self, chunk: Any) -> List[Dict[str, Any]]:
        if not chunk:
            return []
        self.pending += str(chunk)
        events: List[Dict[str, Any]] = []
        if self.capturing:
            self.capture += self.pending
            self.pending = ""
            consumed = self._consume_capture(force=False)
            if consumed:
                events.extend(consumed)
            return events
        start = _first_tool_marker_index(self.pending, self.config)
        if start >= 0:
            prefix = self.pending[:start]
            if prefix:
                events.append({"type": "content", "text": prefix})
            self.capture = self.pending[start:]
            self.pending = ""
            self.capturing = True
            consumed = self._consume_capture(force=False)
            if consumed:
                events.extend(consumed)
            return events
        if len(self.pending) <= self.hold_length:
            return events
        safe = self.pending[: -self.hold_length]
        self.pending = self.pending[-self.hold_length :]
        if safe:
            events.append({"type": "content", "text": safe})
        return events

    def flush(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self.capturing and self.capture:
            consumed = self._consume_capture(force=True)
            if consumed:
                events.extend(consumed)
            else:
                events.append({"type": "content", "text": self.capture})
            self.capture = ""
            self.capturing = False
        if self.pending:
            events.append({"type": "content", "text": self.pending})
            self.pending = ""
        return events

    def _consume_capture(self, force: bool) -> Optional[List[Dict[str, Any]]]:
        if (
            not force
            and _has_open_protocol_block(self.capture, self.config)
            and not _looks_structurally_closed(self.capture, self.config)
        ):
            return None
        calls = parse_tool_calls(self.capture, self.tools, config=self.config)
        if not calls:
            return None
        self.capture = ""
        self.capturing = False
        return [{"type": "tool_calls", "calls": calls}]

    processChunk = process_chunk


class ToolRuntime:
    """Explicit registry for safely dispatching previously parsed tool calls."""

    def __init__(self, tools: Optional[Mapping[str, Callable[..., Any]]] = None) -> None:
        self.tools: Dict[str, Callable[..., Any]] = {}
        for name, handler in dict(tools or {}).items():
            self.register(name, handler)

    def register(self, name: str, handler: Callable[..., Any]) -> "ToolRuntime":
        if not name or not callable(handler):
            raise TypeError("ToolRuntime.register(name, handler) requires a tool name and function")
        self.tools[name] = handler
        return self

    def tool(
        self,
        name: str,
        handler: Optional[Callable[..., Any]] = None,
    ) -> Union["ToolRuntime", Callable[[Callable[..., Any]], Callable[..., Any]]]:
        if handler is not None:
            return self.register(name, handler)

        def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            self.register(name, function)
            return function

        return decorator

    async def execute(
        self,
        calls: Any,
        context: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        runtime_context = dict(context or {})
        for call in _as_list(calls):
            name = _call_value(call, "name")
            call_id = _call_value(call, "id")
            handler = self.tools.get(name)
            if handler is None:
                results.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "name": name,
                        "error": "No handler registered for tool {}".format(name),
                    }
                )
                continue
            try:
                output = handler(
                    _call_value(call, "input", {}),
                    {"call": call, "context": runtime_context},
                )
                if inspect.isawaitable(output):
                    output = await output
                results.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "name": name,
                        "output": output,
                    }
                )
            except Exception as error:  # Runtime handlers are application-defined.
                results.append(
                    {
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "name": name,
                        "error": str(error),
                    }
                )
        return results


def _parse_protocol_markup(
    text: Any,
    protocol: ProtocolSpec,
    allowed: Mapping[str, str],
    tools: Sequence[Mapping[str, Any]],
    config: ToolCallConfig,
) -> List[ParsedToolCall]:
    canonical = _canonicalize_markup(_strip_markdown_fences(str(text or "")))
    calls: List[ParsedToolCall] = []
    for candidate in _extract_protocol_candidates(canonical, protocol):
        for match in _protocol_tag_block_re(protocol, protocol.tags["invoke"]).finditer(candidate):
            name = _canonical_tool_name(_extract_name_attr(match.group(1)), allowed, config)
            if not name:
                continue
            input = _parse_protocol_parameters(match.group(2), protocol, config)
            calls.append(ParsedToolCall(id=config.id_factory(), name=name, input=input))
    if not calls:
        calls.extend(_parse_loose_protocol_calls(canonical, protocol, allowed, tools, config))
    return calls


def _extract_protocol_candidates(text: str, protocol: ProtocolSpec) -> List[str]:
    candidates = [
        match.group(2)
        for match in _protocol_tag_block_re(protocol, protocol.tags["root"]).finditer(text)
    ]
    if candidates:
        return candidates
    match = _protocol_open_tag_re(protocol, protocol.tags["invoke"]).search(text)
    return [text[match.start() :]] if match else []


def _parse_protocol_parameters(body: str, protocol: ProtocolSpec, config: ToolCallConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for match in _protocol_tag_block_re(protocol, protocol.tags["parameter"]).finditer(body):
        name = _extract_name_attr(match.group(1))
        if name:
            out[name] = _decode_markup_value(match.group(2), name, config)
    return out or _parse_text_kv_input(body)


def _parse_loose_protocol_calls(
    text: str,
    protocol: ProtocolSpec,
    allowed: Mapping[str, str],
    tools: Sequence[Mapping[str, Any]],
    config: ToolCallConfig,
) -> List[ParsedToolCall]:
    if not re.search(r"\b{}\b".format(re.escape(protocol.name)), text, re.IGNORECASE):
        return []
    attr_re = re.compile(
        r"\b(?:name|parameter)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([A-Za-z0-9_.:-]+))",
        re.IGNORECASE | re.DOTALL,
    )
    attributes: List[Tuple[str, str, bool, int]] = []
    for match in attr_re.finditer(text):
        raw = _html_unescape(next((value for value in match.groups() if value is not None), "").strip())
        if not raw:
            continue
        name = _canonical_tool_name(raw, allowed, config)
        attributes.append((raw, name or raw, bool(name), match.start()))
    calls: List[ParsedToolCall] = []
    for index, attribute in enumerate(attributes):
        raw, name, is_tool, position = attribute
        if not is_tool:
            continue
        next_tool = next(
            (item[3] for item in attributes[index + 1 :] if item[2]), len(text)
        )
        input: Dict[str, Any] = {}
        for field_raw, _, field_is_tool, field_position in attributes[index + 1 :]:
            if field_position >= next_tool or field_is_tool:
                break
            cdata = re.search(r"<!\[CDATA\[([\s\S]*?)\]\]>", text[field_position:next_tool], re.IGNORECASE)
            if cdata:
                input[field_raw] = _decode_markup_value(cdata.group(1), field_raw, config)
        filtered = _filter_input_for_tool(name, input, tools)
        if not filtered and _required_tool_args(name, tools):
            continue
        calls.append(ParsedToolCall(id=config.id_factory(), name=name, input=filtered))
    return calls


def _parse_xml_tool_calls(
    text: Any,
    allowed: Mapping[str, str],
    config: ToolCallConfig,
) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    raw_text = str(text or "")
    for match in re.finditer(r"<tool_call\b[^>]*>\s*([\s\S]*?)\s*</tool_call\s*>", raw_text, re.IGNORECASE):
        body = match.group(1).strip()
        parsed = _try_json(body)
        calls.extend(
            _parse_json_tool_calls(parsed[1] if parsed[0] else _parse_tool_input(body), allowed, config)
        )
    for expression in (
        r"<tool_use\b([^>]*)>([\s\S]*?)</tool_use>",
        r"<tool_call\b([^>]*)>([\s\S]*?)</tool_call>",
        r"<function\b([^>]*)>([\s\S]*?)</function>",
        r"<invoke\b([^>]*)>([\s\S]*?)</invoke>",
    ):
        for match in re.finditer(expression, raw_text, re.IGNORECASE):
            name = _canonical_tool_name(_extract_name_attr(match.group(1)), allowed, config)
            if name:
                calls.append(
                    ParsedToolCall(
                        id=config.id_factory(),
                        name=name,
                        input=_parse_tool_input(match.group(2).strip()),
                    )
                )
    return calls


def _parse_json_tool_calls(
    value: Any,
    allowed: Mapping[str, str],
    config: ToolCallConfig,
) -> List[ParsedToolCall]:
    calls: List[ParsedToolCall] = []
    if isinstance(value, list):
        for item in value:
            calls.extend(_parse_json_tool_calls(item, allowed, config))
        return calls
    if not _is_mapping(value):
        return calls
    for key in ("tool_calls", "tools"):
        if isinstance(value.get(key), list):
            for item in value[key]:
                calls.extend(_parse_json_tool_calls(item, allowed, config))
    name = _first_string(value.get("name"), value.get("tool"), value.get("tool_name"), value.get("function_name"))
    input = _first_defined(value.get("input"), value.get("arguments"), value.get("args"), value.get("parameters"))
    function = value.get("function")
    if _is_mapping(function):
        name = name or _first_string(function.get("name"))
        if input is None:
            input = _first_defined(function.get("arguments"), function.get("input"), function.get("parameters"))
    canonical_name = _canonical_tool_name(name, allowed, config)
    if canonical_name:
        calls.append(
            ParsedToolCall(
                id=_first_string(value.get("id"), value.get("call_id")) or config.id_factory(),
                name=canonical_name,
                input=_normalize_tool_input(input),
            )
        )
    return calls


def _parse_text_kv_tool_calls(
    text: Any,
    allowed: Mapping[str, str],
    tools: Sequence[Mapping[str, Any]],
    config: ToolCallConfig,
) -> List[ParsedToolCall]:
    values: Dict[str, List[str]] = {"name": [], "arguments": []}
    current = ""
    aliases = {
        "function.name": "name",
        "name": "name",
        "tool": "name",
        "tool.name": "name",
        "tool_name": "name",
        "function.arguments": "arguments",
        "arguments": "arguments",
        "args": "arguments",
        "input": "arguments",
        "tool_input": "arguments",
        "parameters": "arguments",
    }
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^([A-Za-z_.-][A-Za-z0-9_.-]*)\s*:\s*(.*)$", line, re.DOTALL)
        if match and match.group(1).lower() in aliases:
            current = aliases[match.group(1).lower()]
            values[current].append(match.group(2).strip())
            continue
        if current:
            values[current].append(raw_line)
    if not values["name"]:
        return []
    raw_name = "\n".join(values["name"]).splitlines()[0].strip().strip("'\"")
    name = _canonical_tool_name(raw_name, allowed, config)
    if not name:
        return []
    input = _normalize_tool_input("\n".join(values["arguments"]).strip())
    call = _coerce_parsed_call(
        ParsedToolCall(id=config.id_factory(), name=name, input=input), tools, config
    )
    return [call] if call is not None else []


def _coerce_parsed_call(
    call: ParsedToolCall,
    tools: Sequence[Mapping[str, Any]],
    config: ToolCallConfig,
) -> Optional[ParsedToolCall]:
    input = coerce_tool_input(call.name, call.input, tools, config=config)
    if config.unknown_tool == "error" and _tool_schema(call.name, tools) is None:
        raise ValueError("Unknown tool: {}".format(call.name))
    if _missing_required_args(call.name, input, tools):
        if config.missing_required == "error" or config.strict:
            raise ValueError("Missing required arguments for tool: {}".format(call.name))
        if config.missing_required == "drop":
            return None
    if _invalid_tool_args(input):
        return None
    return ParsedToolCall(id=call.id, name=call.name, input=input)


def _coerce_tool_input_by_schema(name: str, input: Any, tools: Sequence[Mapping[str, Any]]) -> Any:
    if not _is_mapping(input):
        return input
    properties = _schema_properties(_tool_schema(name, tools))
    if not properties:
        return input
    fixed = dict(input)
    for key, value in fixed.items():
        if _is_mapping(properties.get(key)):
            fixed[key] = _coerce_value_by_schema(value, properties[key])
    return fixed


def _coerce_value_by_schema(value: Any, schema: Mapping[str, Any]) -> Any:
    types = _schema_types(schema)
    if isinstance(value, str) and ("array" in types or "object" in types):
        parsed, changed = _parse_json_string_for_schema(value, "array" in types, "object" in types)
        if changed:
            value = parsed
    if "array" in types:
        if _is_mapping(value):
            value = [value]
        if isinstance(value, list) and _is_mapping(schema.get("items")):
            return [_coerce_value_by_schema(item, schema["items"]) for item in value]
        return value
    if "object" in types and _is_mapping(value):
        properties = _schema_properties(schema)
        if not properties:
            return value
        fixed = dict(value)
        for key, child in properties.items():
            if key in fixed and _is_mapping(child):
                fixed[key] = _coerce_value_by_schema(fixed[key], child)
        return fixed
    return value


def _parse_json_string_for_schema(value: str, want_array: bool, want_object: bool) -> Tuple[Any, bool]:
    stripped = value.strip()
    if not stripped:
        return value, False
    candidates = [stripped]
    if want_array and not stripped.startswith("["):
        candidates.append("[{}]".format(stripped))
    for candidate in candidates:
        ok, parsed = _try_json(candidate)
        if not ok:
            continue
        if want_array and isinstance(parsed, list):
            return parsed, True
        if want_array and _is_mapping(parsed):
            return [parsed], True
        if want_object and _is_mapping(parsed):
            return parsed, True
    return value, False


def _normalize_tool_input(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return {}
        ok, parsed = _try_json(trimmed)
        if ok:
            return _normalize_tool_input(parsed)
        key_values = _parse_text_kv_input(trimmed)
        return key_values if key_values else value
    return value


def _parse_tool_input(text: str) -> Any:
    if not text:
        return {}
    ok, parsed = _try_json(text)
    if ok:
        return _normalize_tool_input(parsed)
    parameters: Dict[str, Any] = {}
    for match in re.finditer(
        r"<([A-Za-z_][A-Za-z0-9_.:-]*)\b[^>]*>([\s\S]*?)</\1>", text
    ):
        parameters[match.group(1)] = _decode_markup_value(
            match.group(2), match.group(1), ToolCallConfig.default()
        )
    if parameters:
        return parameters
    key_values = _parse_text_kv_input(text)
    return key_values if key_values else {"input": text}


def _parse_text_kv_input(text: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        equals = line.find("=")
        colon = line.find(":")
        separator = colon if equals < 0 or (colon >= 0 and colon < equals) else equals
        if separator <= 0:
            continue
        key = line[:separator].strip()
        value = line[separator + 1 :].strip().strip("'\"")
        if key:
            out[key] = value
    return out


def _for_each_json_fragment(text: Any, visit: Callable[[Any], None]) -> None:
    normalized = _strip_json_fence(str(text or ""))
    for candidate in (
        normalized,
        _repair_loose_json(normalized),
        _recover_json_like(normalized),
    ):
        ok, parsed = _try_json(candidate)
        if ok:
            visit(parsed)
    starts = [index for index, char in enumerate(normalized) if char in "{"]
    starts.extend(index for index, char in enumerate(normalized) if char == "[")
    for start in starts:
        for end in range(len(normalized), start, -1):
            fragment = normalized[start:end]
            ok, parsed = _try_json(fragment)
            if ok:
                visit(parsed)
                break
            ok, parsed = _try_json(_repair_loose_json(fragment))
            if ok:
                visit(parsed)
                break


def _build_allowed_tool_map(
    tools: Sequence[Mapping[str, Any]], config: ToolCallConfig
) -> Dict[str, str]:
    allowed: Dict[str, str] = {}
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        allowed[_tool_alias_key(name)] = name
        alias = SAFE_TOOL_ALIASES.get(name)
        if alias:
            allowed[_tool_alias_key(alias)] = name
    for alias, canonical in config.tool_aliases.items():
        real = allowed.get(_tool_alias_key(canonical), canonical)
        allowed[_tool_alias_key(alias)] = real
    return allowed


def _canonical_tool_name(name: Any, allowed: Mapping[str, str], config: ToolCallConfig) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    direct = allowed.get(_tool_alias_key(raw))
    if direct:
        return direct
    configured = config.tool_aliases.get(raw) or config.tool_aliases.get(raw.lower())
    if configured and allowed.get(_tool_alias_key(configured)):
        return allowed[_tool_alias_key(configured)]
    if raw.startswith("u_"):
        return allowed.get(_tool_alias_key(raw[2:]), "")
    return "" if config.unknown_tool == "drop" else raw


def _dedupe_tool_calls(calls: Iterable[ParsedToolCall]) -> List[ParsedToolCall]:
    seen: Set[str] = set()
    out: List[ParsedToolCall] = []
    for call in calls:
        key = "{}\0{}".format(_tool_alias_key(call.name), _stable_stringify(call.input))
        if not call.name or key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def _tool_schema(name: str, tools: Any) -> Optional[Mapping[str, Any]]:
    for tool in normalize_tools(tools):
        if tool.get("name") == name and _is_mapping(tool.get("parameters") or tool.get("input_schema")):
            return tool.get("parameters") or tool.get("input_schema")
    return None


def _schema_properties(schema: Any) -> Optional[Mapping[str, Any]]:
    return schema.get("properties") if _is_mapping(schema) and _is_mapping(schema.get("properties")) else None


def _schema_types(schema: Any) -> Set[str]:
    types: Set[str] = set()
    if not _is_mapping(schema):
        return types
    kind = schema.get("type")
    if isinstance(kind, str):
        types.add(kind)
    elif isinstance(kind, list):
        types.update(item for item in kind if isinstance(item, str))
    if schema.get("properties") is not None:
        types.add("object")
    if schema.get("items") is not None:
        types.add("array")
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            for variant in schema[key]:
                types.update(_schema_types(variant))
    return types


def _required_tool_args(name: str, tools: Any) -> List[str]:
    seen: Set[str] = set()
    required: List[str] = []

    def add(*keys: Any) -> None:
        for key in keys:
            if isinstance(key, str) and key and key not in seen:
                seen.add(key)
                required.append(key)

    schema = _tool_schema(name, tools)
    if _is_mapping(schema) and isinstance(schema.get("required"), list):
        add(*schema["required"])
    if name == "Read":
        add("file_path")
    elif name == "Write":
        add("file_path", "content")
    elif name == "Edit":
        add("file_path")
    elif name in {"Bash", "PowerShell"}:
        add("command")
    return required


def _missing_required_args(name: str, input: Any, tools: Any) -> bool:
    if not _is_mapping(input):
        return False
    for key in _required_tool_args(name, tools):
        value = input.get(key)
        if value is None:
            return True
        if isinstance(value, str) and not value.strip() and not _required_arg_allows_empty_string(name, key):
            return True
    return False


def _required_arg_allows_empty_string(tool_name: str, argument_name: str) -> bool:
    return _tool_alias_key(tool_name) in {"write", "writefile", "createfile"} and _tool_alias_key(
        argument_name
    ) in {"content", "text", "body", "data", "value", "contents", "filecontent"}


def _invalid_tool_args(input: Any) -> bool:
    if not _is_mapping(input):
        return False
    return any(
        _is_path_like_arg_name(key) and _path_like_arg_looks_polluted(str(value or ""))
        for key, value in input.items()
    )


def _is_path_like_arg_name(name: Any) -> bool:
    return _tool_alias_key(name) in {
        "path",
        "filepath",
        "filename",
        "targetfile",
        "file",
        "dir",
        "directory",
        "cwd",
        "workdir",
        "workingdirectory",
    }


def _path_like_arg_looks_polluted(value: str) -> bool:
    trimmed = value.strip()
    if not trimmed or "\0" in trimmed or re.search(r"[\r\n<>]", trimmed):
        return True
    lowered = trimmed.lower()
    markers = (
        "<![cdata[",
        "]]>",
        "xyml|",
        "qnml|",
        "tool_calls",
        "invoke name=",
        "parameter name=",
        "</parameter",
        "</invoke",
        "function.name:",
        "function.arguments:",
    )
    return any(marker in lowered for marker in markers)


def _filter_input_for_tool(
    name: str, input: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    properties = _schema_properties(_tool_schema(name, tools))
    return dict(input) if not properties else {key: value for key, value in input.items() if key in properties}


def _tool_accepts_field(name: str, tools: Any, field: str) -> bool:
    properties = _schema_properties(_tool_schema(name, tools))
    return bool(properties and field in properties)


def _rename_first_present(object: Dict[str, Any], canonical: str, *aliases: Any) -> None:
    if object.get(canonical) is not None:
        return
    for alias in aliases:
        if object.get(alias) is not None:
            object[canonical] = object.pop(alias)
            return


def _normalize_protocol_specs(values: Any, emit_protocol: str) -> List[ProtocolSpec]:
    out: List[ProtocolSpec] = []
    seen: Set[str] = set()
    for value in _as_list(values):
        spec = _normalize_protocol_spec(value)
        key = spec.name.lower()
        if key not in seen:
            seen.add(key)
            out.append(spec)
    if str(emit_protocol).lower() not in seen:
        out.insert(0, ProtocolSpec(emit_protocol))
    return out


def _normalize_protocol_spec(value: Union[str, ProtocolSpec, Mapping[str, Any]]) -> ProtocolSpec:
    if isinstance(value, ProtocolSpec):
        return value
    if isinstance(value, str):
        return ProtocolSpec(value)
    if _is_mapping(value):
        options = dict(value)
        name = options.pop("name", None)
        return ProtocolSpec(name, **options)
    raise TypeError("Invalid protocol spec")


def _protocol_open_tag_re(protocol: ProtocolSpec, tag: str) -> re.Pattern[str]:
    return re.compile(
        r"<\s*\|\s*{}\s*\|\s*{}\b[^>]*>".format(
            re.escape(protocol.name), re.escape(tag)
        ),
        re.IGNORECASE,
    )


def _protocol_tag_block_re(protocol: ProtocolSpec, tag: str) -> re.Pattern[str]:
    escaped_protocol = re.escape(protocol.name)
    escaped_tag = re.escape(tag)
    return re.compile(
        r"<\s*\|\s*{}\s*\|\s*{}\b([^>]*)>([\s\S]*?)<\s*/\s*\|\s*{}\s*\|\s*{}\s*>".format(
            escaped_protocol,
            escaped_tag,
            escaped_protocol,
            escaped_tag,
        ),
        re.IGNORECASE,
    )


def _extract_name_attr(attributes: Any) -> str:
    match = re.search(
        r"(?:^|[\s|])name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s|/>]+))",
        str(attributes or ""),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return _html_unescape(next((value for value in match.groups() if value is not None), "").strip())


def _decode_markup_value(raw: Any, parameter_name: Any, config: ToolCallConfig) -> Any:
    cdata_matches = re.findall(r"<!\[CDATA\[([\s\S]*?)\]\]>", str(raw or ""), re.IGNORECASE)
    raw_string = str(parameter_name or "").lower() in config.raw_string_params
    if cdata_matches:
        joined = "".join(cdata_matches)
        return joined if raw_string else _coerce_markup_scalar(joined, raw_string=False)
    if not raw_string:
        parsed, nested = _parse_nested_markup_value(str(raw or ""), config)
        if parsed:
            return nested
    return _coerce_markup_scalar(raw, raw_string=raw_string)


def _parse_nested_markup_value(raw: str, config: ToolCallConfig) -> Tuple[bool, Any]:
    text = raw.strip()
    if not text or "<" not in text:
        return False, None
    matches = list(
        re.finditer(r"<([A-Za-z_][A-Za-z0-9_.:-]*)\b[^>]*>([\s\S]*?)</\1>", text)
    )
    if not matches:
        return False, None
    names = [match.group(1) for match in matches]
    values = [_decode_markup_value(match.group(2), match.group(1), config) for match in matches]
    if all(name.lower() == "item" for name in names):
        return True, values
    out: Dict[str, Any] = {}
    for name, value in zip(names, values):
        if name not in out:
            out[name] = value
        elif isinstance(out[name], list):
            out[name].append(value)
        else:
            out[name] = [out[name], value]
    return True, out


def _coerce_markup_scalar(raw: Any, raw_string: bool) -> Any:
    value = _html_unescape(str(raw or "").strip())
    if raw_string:
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null":
        return None
    ok, parsed = _try_json(value)
    return _normalize_tool_input(parsed) if ok else value


def _render_markup_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return "<![CDATA[{}]]>".format(value.replace("]]>", "]]]]><![CDATA[>"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return _json_dumps(value)


def _first_tool_marker_index(text: str, config: ToolCallConfig) -> int:
    indexes: List[int] = []
    for protocol in config.parse_protocols:
        for tag in (protocol.tags["root"], protocol.tags["invoke"]):
            match = _protocol_open_tag_re(protocol, tag).search(text)
            if match:
                indexes.append(match.start())
    for expression in (r'^\s*\{\s*"tool_calls"', r"function\.name\s*:"):
        match = re.search(expression, text, re.IGNORECASE | re.DOTALL)
        if match:
            indexes.append(match.start())
    return min(indexes) if indexes else -1


def _has_open_protocol_block(text: str, config: ToolCallConfig) -> bool:
    return any(
        _protocol_open_tag_re(protocol, protocol.tags["root"]).search(text)
        or _protocol_open_tag_re(protocol, protocol.tags["invoke"]).search(text)
        for protocol in config.parse_protocols
    )


def _looks_structurally_closed(text: str, config: ToolCallConfig) -> bool:
    if re.search(r"\n\s*[\]}]\s*$", text):
        return True
    for protocol in config.parse_protocols:
        expression = r"<\s*/\s*\|\s*{}\s*\|\s*{}\s*>".format(
            re.escape(protocol.name), re.escape(protocol.tags["root"])
        )
        if re.search(expression, text, re.IGNORECASE):
            return True
    return False


def _canonicalize_markup(text: str) -> str:
    for old, new in MARKUP_REPLACEMENTS:
        text = text.replace(old, new)
    return text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "").replace("\u3000", " ").replace("\u00a0", " ")


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)```", r"\1", text)


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)```", text.strip(), re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _repair_loose_json(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r'"name="\s*', '"name": "', repaired, flags=re.IGNORECASE | re.DOTALL)
    repaired = re.sub(r'"name=([^",}\s]+)"', r'"name": "\1"', repaired, flags=re.IGNORECASE | re.DOTALL)
    repaired = re.sub(
        r'"(name|input|arguments|args|parameters|tool|tool_name|function_name)"\s*=\s*',
        r'"\1": ',
        repaired,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return re.sub(
        r'([{,]\s*)(name|input|arguments|args|parameters|tool|tool_name|function_name)\s*:',
        r'\1"\2":',
        repaired,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _recover_json_like(text: str) -> str:
    repaired = text.strip()
    unclosed_braces = repaired.count("{") - repaired.count("}")
    unclosed_brackets = repaired.count("[") - repaired.count("]")
    return repaired + ("]" * max(unclosed_brackets, 0)) + ("}" * max(unclosed_braces, 0))


def _try_json(text: Any) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, None


def _resolve_config(config: Optional[Union[ToolCallConfig, Mapping[str, Any]]]) -> ToolCallConfig:
    if isinstance(config, ToolCallConfig):
        return config
    if config is None:
        return ToolCallConfig()
    if _is_mapping(config):
        return ToolCallConfig(config)
    raise TypeError("config must be a ToolCallConfig or mapping")


def _safe_tool_name(name: Any) -> str:
    trimmed = str(name or "").strip()
    if not trimmed:
        return ""
    if trimmed in SAFE_TOOL_ALIASES:
        return SAFE_TOOL_ALIASES[trimmed]
    if any(alias.lower() == trimmed.lower() for alias in SAFE_TOOL_ALIASES.values()):
        return trimmed
    return trimmed if trimmed.startswith("u_") else "u_{}".format(trimmed)


def _example_input_from_tool(tool: Mapping[str, Any]) -> Dict[str, Any]:
    properties = _schema_properties(tool.get("parameters") or tool.get("input_schema"))
    if not properties:
        return {"ARG": "value"}
    example = {key: _example_value(schema) for key, schema in list(properties.items())[:3]}
    return example or {"ARG": "value"}


def _example_value(schema: Any) -> Any:
    kinds = _schema_types(schema)
    if "array" in kinds:
        return []
    if "object" in kinds:
        return {}
    if "boolean" in kinds:
        return True
    if "number" in kinds or "integer" in kinds:
        return 1
    return "value"


def _summarize_schema(schema: Any) -> str:
    return "{}" if not schema else _json_dumps(schema)


def _clip(text: Any, maximum: int) -> str:
    value = str(text or "").strip()
    return "{}...".format(value[:maximum]) if len(value) > maximum else value


def _stable_stringify(value: Any) -> str:
    if isinstance(value, list):
        return "[{}]".format(",".join(_stable_stringify(item) for item in value))
    if _is_mapping(value):
        return "{{{}}}".format(
            ",".join(
                "{}:{}".format(_json_dumps(str(key)), _stable_stringify(value[key]))
                for key in sorted(value, key=lambda item: str(item))
            )
        )
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _escape_xml(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _html_unescape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )


def _tool_alias_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _first_string(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_defined(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _take_option(values: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in values:
            return values.pop(name)
    return default


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _call_value(call: Any, key: str, default: Any = None) -> Any:
    if _is_mapping(call):
        return call.get(key, default)
    return getattr(call, key, default)


def _arguments_string(value: Any) -> str:
    return value if isinstance(value, str) else _json_dumps({} if value is None else value)


# JavaScript-compatible aliases make shared migration examples straightforward.
normalizeTools = normalize_tools
buildToolInstructions = build_tool_instructions
renderToolCall = render_tool_call
renderToolCalls = render_tool_calls
parseToolCalls = parse_tool_calls
parseMarkupToolCalls = parse_markup_tool_calls
coerceToolInput = coerce_tool_input
openAIToolCalls = openai_tool_calls
responsesToolItems = responses_tool_items
anthropicToolUseBlocks = anthropic_tool_use_blocks
