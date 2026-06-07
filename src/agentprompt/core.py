"""Core types: Prompt + Messages builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

import jinja2


# Match the Rust crate's serde lowercase serialization so the rendered
# output is wire-compatible with Anthropic's and OpenAI's messages arrays.
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(TypedDict):
    """One rendered turn, ready to drop into a provider SDK payload."""

    role: str
    content: str


class MissingVariableError(KeyError):
    """Raised when a template references a variable not present in the context.

    Strict mode is on by default so we fail loud rather than ship a prompt
    with a literal ``{{q}}`` in it — the silent-send footgun is the whole
    reason this library exists.

    ``name`` is the missing variable name; ``str(exc)`` is the human message.
    """

    def __init__(self, name: str, *, template_label: str = "<inline>") -> None:
        super().__init__(name)
        self.name = name
        self.template_label = template_label

    def __str__(self) -> str:  # type: ignore[override]
        return f"template {self.template_label!r} references undefined variable {self.name!r}"


def _build_env() -> jinja2.Environment:
    """One module-wide Environment. ``StrictUndefined`` makes missing
    variables raise instead of silently rendering empty string."""
    return jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        # Templates are LLM prompts, not HTML — autoescape would mangle quote
        # characters in user-facing strings.
        autoescape=False,
        # ``keep_trailing_newline=True`` matches what people expect when
        # writing multi-line system prompts: the final newline they typed
        # stays in the rendered output.
        keep_trailing_newline=True,
    )


_ENV = _build_env()


def _undefined_name(msg: str) -> str:
    """Best-effort extraction of the offending name from a Jinja2
    ``UndefinedError`` message.

    Jinja2's messages are stable across the two shapes we care about:

    * a bare undefined variable -> ``"'q' is undefined"`` (we want ``q``)
    * a missing attribute/key on a defined value ->
      ``"'dict object' has no attribute 'name'"`` (we want ``name``)

    Anything we don't recognise falls back to ``"<unknown>"`` rather than
    raising, so a render failure never turns into a parser failure.
    """
    is_undefined = "' is undefined"
    if msg.startswith("'") and msg.endswith(is_undefined):
        return msg[1 : len(msg) - len(is_undefined)]
    no_attr = " has no attribute '"
    if no_attr in msg and msg.endswith("'"):
        # The attribute name is the final quoted segment.
        return msg[msg.rfind(no_attr) + len(no_attr) : -1]
    return "<unknown>"


@dataclass
class Prompt:
    """A single template; render() returns a plain string.

    Compiles the template lazily on first ``render()`` so cheap construction
    is fine. The cached ``_template`` makes repeated renders of the same
    Prompt fast.
    """

    source: str
    label: str = "<inline>"
    # The compiled template is an internal cache populated lazily on first
    # render. Exclude it from ``__init__``, ``__repr__`` and ``__eq__`` so two
    # Prompts with the same source/label stay equal regardless of whether
    # either has been rendered yet.
    _template: jinja2.Template | None = field(default=None, init=False, repr=False, compare=False)

    def render(self, context: Mapping[str, Any] | None = None) -> str:
        if self._template is None:
            self._template = _ENV.from_string(self.source)
        ctx = dict(context or {})
        try:
            return self._template.render(**ctx)
        except jinja2.UndefinedError as exc:
            raise MissingVariableError(
                _undefined_name(str(exc)), template_label=self.label
            ) from None


@dataclass
class _Turn:
    role: Role
    template: Prompt


class Messages:
    """Chainable builder for a sequence of role-tagged templates.

    Each ``.system(...)`` / ``.user(...)`` / ``.assistant(...)`` appends a
    new turn. ``.render(context)`` returns the list of dict-shaped messages
    ready for an SDK payload. The builder is immutable on render — call it
    repeatedly with different contexts to fan out the same template set.

    Construction is cheap; templates compile on first render and cache.
    """

    def __init__(self) -> None:
        self._turns: list[_Turn] = []

    # ---- builder methods ----
    def system(self, source: str, *, label: str | None = None) -> Messages:
        self._turns.append(_Turn(Role.SYSTEM, Prompt(source, label=label or "system")))
        return self

    def user(self, source: str, *, label: str | None = None) -> Messages:
        self._turns.append(_Turn(Role.USER, Prompt(source, label=label or "user")))
        return self

    def assistant(self, source: str, *, label: str | None = None) -> Messages:
        self._turns.append(_Turn(Role.ASSISTANT, Prompt(source, label=label or "assistant")))
        return self

    def tool(self, source: str, *, label: str | None = None) -> Messages:
        """A `tool` role turn — useful when the provider wants tool-call
        results to round-trip back to the model. Anthropic's API uses a
        slightly different shape for tools; this just produces ``role: "tool"``
        and ``content: <rendered>``, which most SDKs accept as a string."""
        self._turns.append(_Turn(Role.TOOL, Prompt(source, label=label or "tool")))
        return self

    # ---- render ----
    def render(self, context: Mapping[str, Any] | None = None) -> list[Message]:
        return [
            {"role": turn.role.value, "content": turn.template.render(context)}
            for turn in self._turns
        ]

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return f"Messages(turns={[turn.role.value for turn in self._turns]})"
