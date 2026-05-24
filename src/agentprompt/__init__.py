"""agentprompt — LLM prompt templates with Jinja2 syntax.

Smallest possible primitive: Jinja2 under the hood, role-aware on top.
Sibling to the Rust crate ``agentprompt`` (https://crates.io/crates/agentprompt).

Usage:

    from agentprompt import Messages, Prompt

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

    # Pure string templating:
    prompt = Prompt("{% if items %}{{items|length}} todos{% else %}empty{% endif %}")
    prompt.render({"items": ["a", "b"]})  # → "2 todos"
"""

from .core import Message, Messages, MissingVariableError, Prompt, Role

__all__ = [
    "Message",
    "Messages",
    "MissingVariableError",
    "Prompt",
    "Role",
]

__version__ = "0.1.0"
