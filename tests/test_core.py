"""Tests for the Prompt + Messages builder."""

from __future__ import annotations

import jinja2
import pytest

from agentprompt import Messages, MissingVariableError, Prompt, Role

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_prompt_renders_basic_substitution():
    p = Prompt("Hello, {{name}}!")
    assert p.render({"name": "Mukunda"}) == "Hello, Mukunda!"


def test_prompt_renders_filters_and_conditionals():
    p = Prompt(
        "{% if items %}You have {{items|length}} todos.{% else %}Nothing pending.{% endif %}"
    )
    assert p.render({"items": ["a", "b"]}) == "You have 2 todos."
    assert p.render({"items": []}) == "Nothing pending."


def test_prompt_strict_missing_raises():
    p = Prompt("Question: {{q}}")
    with pytest.raises(MissingVariableError) as exc:
        p.render({})
    assert exc.value.name == "q"


def test_prompt_missing_carries_template_label():
    p = Prompt("Question: {{q}}", label="welcome-prompt")
    with pytest.raises(MissingVariableError) as exc:
        p.render({})
    assert "welcome-prompt" in str(exc.value)


def test_prompt_caches_compiled_template():
    """Repeated renders of the same Prompt reuse the compiled template."""
    p = Prompt("{{x}}")
    p.render({"x": 1})
    first_template_id = id(p._template)
    p.render({"x": 2})
    assert id(p._template) == first_template_id


def test_prompt_no_autoescape_for_html_like_content():
    """Templates aren't HTML — angle brackets must round-trip literally."""
    p = Prompt("{{q}}")
    assert p.render({"q": "<script>alert(1)</script>"}) == "<script>alert(1)</script>"


def test_prompt_keeps_trailing_newline():
    """System prompts written with a trailing \\n should retain it."""
    p = Prompt("You are concise.\n")
    assert p.render({}) == "You are concise.\n"


def test_prompt_equality_unaffected_by_render_cache():
    """The lazily-compiled template is an internal cache; rendering one of two
    otherwise-identical Prompts must not make them compare unequal."""
    a = Prompt("{{x}}")
    b = Prompt("{{x}}")
    assert a == b
    a.render({"x": 1})
    assert a == b


def test_prompt_missing_attribute_reports_attribute_name():
    """A missing attribute on a defined value should surface the attribute
    name, not '<unknown>'."""
    p = Prompt("{{user.name}}")
    with pytest.raises(MissingVariableError) as exc:
        p.render({"user": {}})
    assert exc.value.name == "name"


def test_prompt_missing_key_reports_key_name():
    """Same for subscript access on a defined mapping that lacks the key."""
    p = Prompt("{{user['name']}}")
    with pytest.raises(MissingVariableError) as exc:
        p.render({"user": {}})
    assert exc.value.name == "name"


def test_missing_variable_error_is_catchable_as_key_error():
    """The README/docstring document ``MissingVariableError(KeyError)`` so callers
    can catch the broad ``KeyError`` — guard that subclassing contract."""
    with pytest.raises(KeyError) as exc:
        Prompt("{{q}}").render({})
    assert isinstance(exc.value, MissingVariableError)


def test_prompt_default_template_label_is_inline():
    """An unlabelled Prompt reports the ``<inline>`` sentinel in its error."""
    with pytest.raises(MissingVariableError) as exc:
        Prompt("{{q}}").render({})
    assert exc.value.template_label == "<inline>"
    assert "<inline>" in str(exc.value)


def test_prompt_syntax_error_propagates_as_jinja_error():
    """A malformed template is a programmer error and should surface as Jinja2's
    ``TemplateSyntaxError`` rather than being masked as a MissingVariableError."""
    with pytest.raises(jinja2.TemplateSyntaxError):
        Prompt("{% if %}").render({})


def test_prompt_renders_non_string_context_values():
    """Non-string context values are coerced via Jinja2's str rendering."""
    assert Prompt("{{n}}").render({"n": 42}) == "42"


# ---------------------------------------------------------------------------
# Messages builder
# ---------------------------------------------------------------------------


def test_messages_renders_role_tagged_list():
    out = (
        Messages()
        .system("You are a {{adj}} assistant.")
        .user("Q: {{q}}")
        .render({"adj": "concise", "q": "what is 2+2?"})
    )
    assert out == [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Q: what is 2+2?"},
    ]


def test_messages_supports_assistant_and_tool_roles():
    out = Messages().system("sys").user("u").assistant("a").tool("t").render()
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]


def test_messages_render_with_no_context_when_no_vars():
    out = Messages().system("hi").user("hello").render()
    assert out[0]["content"] == "hi"
    assert out[1]["content"] == "hello"


def test_messages_missing_variable_surfaces_label():
    msgs = Messages().system("ok", label="sys-msg").user("Q: {{q}}", label="user-msg")
    with pytest.raises(MissingVariableError) as exc:
        msgs.render({})
    assert "user-msg" in str(exc.value)
    assert exc.value.name == "q"


def test_messages_can_be_rendered_multiple_times_with_different_context():
    msgs = Messages().system("You translate to {{lang}}.").user("Translate: {{text}}")
    a = msgs.render({"lang": "fr", "text": "hello"})
    b = msgs.render({"lang": "es", "text": "hello"})
    assert a[0]["content"] == "You translate to fr."
    assert b[0]["content"] == "You translate to es."


def test_messages_len_reflects_turn_count():
    m = Messages().system("a").user("b").assistant("c")
    assert len(m) == 3


def test_messages_returns_dict_shape_compatible_with_provider_sdks():
    """Each message must be a plain dict with 'role' (lowercase string) and
    'content'. That's the shape both Anthropic and OpenAI accept."""
    out = Messages().system("s").user("u").render()
    for msg in out:
        assert isinstance(msg, dict)
        assert set(msg.keys()) == {"role", "content"}
        assert isinstance(msg["role"], str)
        assert msg["role"].islower()
        assert isinstance(msg["content"], str)


def test_messages_repr_lists_turn_roles_in_order():
    """``repr`` should reflect the role sequence for quick debugging."""
    m = Messages().system("a").user("b").assistant("c")
    assert repr(m) == "Messages(turns=['system', 'user', 'assistant'])"


def test_messages_tool_role_uses_lowercase_tool_value():
    """The ``tool`` turn serialises to the wire-compatible ``role: 'tool'``."""
    out = Messages().tool("result").render()
    assert out == [{"role": "tool", "content": "result"}]


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


def test_role_values_match_provider_wire_format():
    """Role values are lowercase to stay wire-compatible with provider SDKs."""
    assert [r.value for r in Role] == ["system", "user", "assistant", "tool"]
    assert Role.SYSTEM == "system"
