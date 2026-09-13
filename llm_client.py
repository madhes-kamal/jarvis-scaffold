"""
llm_client.py

The single choke point every other file goes through to talk to the LLM.

Why this exists: conversation.py and calendar_manager.py used to each
call ollama.chat(...) directly, with the model name hardcoded in each
file. Wanting to try a different model, or point at your PC instead of
the Pi, meant hunting through multiple files. Now there's exactly one
place that knows how to talk to the model -- everything else calls
chat_completion() and doesn't know or care what's on the other end.
Swapping backends later (a different model, a different machine, a
cloud API) means editing this file only.

This also centralizes a Qwen3-specific quirk: Ollama's `think=False`
doesn't always fully suppress the model's <think>...</think> reasoning
block when tools are involved (a known issue). Every response gets
cleaned here once, instead of every caller needing to remember to do it.
"""

import ollama

# Single source of truth for which model the whole app uses. Change
# this one line to point everything at a different model.
MODEL = "qwen3:1.7b"


def _strip_thinking(text):
    """Remove a <think>...</think> block if the model included one, even
    when we asked it not to."""
    if text and "</think>" in text:
        return text.split("</think>")[-1].strip()
    return (text or "").strip()


def chat_completion(messages, tools=None, temperature=0.2):
    """
    The one function every other file calls to talk to the LLM.

    messages: standard list of {"role": ..., "content": ...} dicts
    tools: optional list of Python functions to expose as tools
    temperature: sampling temperature (lower = more consistent/less creative)

    Returns a plain dict: {"role": "assistant", "content": ..., "tool_calls": ...}
    -- shaped so callers can read .content/.tool_calls via dict keys AND
    append this same dict straight back into `messages` for a follow-up
    call, since Ollama expects conversation history as plain role/content
    entries.
    """
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "options": {"temperature": temperature, "think": False},
    }
    if tools:
        kwargs["tools"] = tools

    response = ollama.chat(**kwargs)

    return {
        "role": "assistant",
        "content": _strip_thinking(response.message.content),
        "tool_calls": response.message.tool_calls,
    }
