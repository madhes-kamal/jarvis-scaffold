"""
conversation.py

Handles one back-and-forth turn with the local LLM, including tool
calling: this is what lets you say "add a 30 minute break at 3" and have
the model actually call create_event() instead of just saying it did.

Ollama lets you hand it plain Python functions (type hints + docstring)
as tools and it works out the JSON schema for you -- no manual schema
writing needed, which is why calendar_service.py's functions have full
docstrings and type hints.
"""

import datetime

import ollama

from calendar_service import get_todays_events, create_event

# Tool calling is noticeably more reliable on 7b than 1.5b -- a small
# model will often *say* it added an event without actually calling the
# tool. If you're stuck on 1.5b for speed/RAM reasons, double check
# results land on your actual calendar, don't just trust the reply.
MODEL = "qwen2.5:1.5b"

AVAILABLE_TOOLS = {
    "get_todays_events": get_todays_events,
    "create_event": create_event,
}


def _system_prompt():
    now = datetime.datetime.now()
    return f"""You are a casual, friendly personal assistant (like JARVIS,
but relaxed, not formal). You can read and modify the user's Google
Calendar using the tools you're given.

Today's date and time: {now.strftime("%A, %B %d, %Y, %I:%M %p")}.

CRITICAL: if the user asks you to add, schedule, move, or create
anything, you MUST call the create_event tool in this same turn. Do not
respond with text saying you've added something -- if you haven't
actually called the tool, nothing happened. Only reply in plain text
once the tool result confirms it worked.

When calling create_event, pass a natural-language description close to
what the user actually said (e.g. "Dinner today from 7:15pm to 8pm") --
do NOT convert it to ISO format or compute a timezone offset yourself,
Google parses the natural description for you.

Example: user says "add a 10 minute break at 3pm" -> you call
create_event(event_description="Break today from 3pm to 3:10pm"). You do
NOT just say "Sure, I've added a break."

Keep spoken replies short and natural, like you're talking, not writing."""


def run_turn(user_message, history):
    """
    Run one conversation turn.

    `history` is a list of prior messages -- keep it between calls so the
    assistant remembers the conversation.

    Returns (reply_text, updated_history).
    """
    messages = [{"role": "system", "content": _system_prompt()}] + history
    messages.append({"role": "user", "content": user_message})

    response = ollama.chat(
        model=MODEL,
        messages=messages,
        tools=[get_todays_events, create_event],
        options={"temperature": 0.2},
    )

    messages.append(response.message)

    tool_calls = response.message.tool_calls
    print(f"  [debug: tool_calls = {tool_calls}]")  # remove once this works reliably
    if tool_calls:
        for call in tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [calling tool: {name}({args})]")
            func = AVAILABLE_TOOLS[name]
            result = func(**args)
            messages.append({"role": "tool", "content": str(result), "name": name})

        # Hand the tool's result back to the model so it can turn it into
        # a natural spoken reply instead of raw data.
        follow_up = ollama.chat(model=MODEL, messages=messages)
        messages.append(follow_up.message)
        reply = follow_up.message.content
    else:
        reply = response.message.content

    return reply, messages[1:]  # drop the system prompt from saved history
