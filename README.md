# agentprompt-py

[![PyPI](https://img.shields.io/pypi/v/agentprompt-py.svg)](https://pypi.org/project/agentprompt-py/)
[![Python](https://img.shields.io/pypi/pyversions/agentprompt-py.svg)](https://pypi.org/project/agentprompt-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**LLM prompt templates with Jinja2 syntax. Role-aware Messages builder
produces a typed message list ready for Anthropic or OpenAI SDKs.**

Sibling to the Rust crate
[`agentprompt`](https://crates.io/crates/agentprompt).

## Install

```bash
pip install agentprompt-py
```

## Use

```python
from agentprompt import Messages

messages = (
    Messages()
    .system("You are a {{adjective}} assistant.")
    .user("Question: {{q}}")
    .render({"adjective": "concise", "q": "what is 2+2?"})
)
# messages = [
#   {"role": "system", "content": "You are a concise assistant."},
#   {"role": "user",   "content": "Question: what is 2+2?"},
# ]
```

Drop straight into the Anthropic SDK:

```python
import anthropic

client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=messages,        # already shaped right
)
```

Same for OpenAI:

```python
import openai

resp = openai.chat.completions.create(
    model="gpt-5.4",
    messages=messages,
)
```

## Pure string templating

If you don't need role splitting, `Prompt` is the leaner primitive:

```python
from agentprompt import Prompt

p = Prompt("{% if items %}You have {{items|length}} todos.{% else %}Nothing pending.{% endif %}")
p.render({"items": ["A", "B"]})   # "You have 2 todos."
p.render({"items": []})           # "Nothing pending."
```

## Strictness

Missing variables fail loud. We'd rather break tests than silently send
the model a prompt with a literal `{{q}}` in it.

```python
Prompt("Question: {{q}}").render({})
# → MissingVariableError: template '<inline>' references undefined variable 'q'
```

`MissingVariableError.name` carries the missing variable; `.template_label`
carries the label you optionally passed to `Prompt(source, label="...")` or
`Messages().system(source, label="...")`. Use them when wiring agent code
that wants structured logging instead of free-form exception text.

## Re-rendering

`Messages` is immutable on `render` — call it as many times as you want
with different contexts:

```python
greeter = Messages().system("You greet in {{lang}}.").user("Hello!")

en = greeter.render({"lang": "English"})
fr = greeter.render({"lang": "French"})
```

The compiled template is cached on first render, so subsequent renders
are fast even on large templates.

## API surface

- `Prompt(source, label="<inline>")` — single Jinja2 template; `.render(context)` returns `str`.
- `Messages()` — chainable builder.
  - `.system(source, label=None) -> Messages`
  - `.user(source, label=None) -> Messages`
  - `.assistant(source, label=None) -> Messages`
  - `.tool(source, label=None) -> Messages`
  - `.render(context) -> list[Message]` where `Message = TypedDict(role: str, content: str)`.
- `Role` enum — `SYSTEM | USER | ASSISTANT | TOOL`.
- `MissingVariableError(KeyError)` — raised by strict mode; `.name`, `.template_label`.

## What it doesn't do

- No `{% include %}` from disk in v0.1. Read your template files yourself and pass the content to `Prompt(...)`.
- No HTML autoescape (these aren't HTML — autoescape would mangle quotes in prompts).
- No provider-specific message shapes. You get `list[Message]`; map to the SDK type yourself.

## Tests

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

## License

MIT
