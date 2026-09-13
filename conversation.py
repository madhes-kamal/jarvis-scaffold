"""
conversation.py

General chat fallback for anything calendar_manager.py decides is NOT a
calendar request, plus extract_event_phrase() which calendar_manager.py
reuses for turning messy speech into a clean create-event phrase.

All LLM calls go through llm_client.chat_completion() -- see that file
for why. Nothing here calls ollama directly or hardcodes a model name.
"""

import datetime

from calendar_service import get_todays_events, create_event
from llm_client import chat_completion

AVAILABLE_TOOLS = {
    "get_todays_events": get_todays_events,
    "create_event": create_event,
}


def extract_event_phrase(user_text, events_context=None):
    """
    Turn spoken/messy phrasing into a clean quick-add-style phrase, e.g.
    "Yo can you add a break from 715 to 8" -> "Break today from 7:15pm to 8pm".

    This is plain text rewriting, NOT tool calling -- Python has already
    decided (via calendar_manager's classification) that an event needs
    to be created.

    events_context: optional list of today's events so relative
    references like "after chemistry" get resolved against the real
    schedule instead of guessed.
    """
    now = datetime.datetime.now()

    context_block = ""
    if events_context:
        schedule_lines = "\n".join(
            f"- {e['summary']}: {e['start']} to {e['end']}" for e in events_context
        )
        context_block = f"\n\nToday's actual schedule (use this to resolve relative references like 'after X' or 'before X'):\n{schedule_lines}"

    message = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    f"Today is {now.strftime('%A, %B %d, %Y, %I:%M %p')}. "
                    "Rewrite the user's message into a short calendar "
                    "quick-add phrase: event title, then the date/time. "
                    "Resolve relative times ('tomorrow', 'in an hour') "
                    "into an actual day/time using today's date above. "
                    "If the request references another event ('after "
                    "chemistry'), look up that event in the schedule "
                    "below and use its actual time -- never guess a time "
                    "that isn't grounded in the schedule or the message "
                    "itself. Output ONLY the phrase itself -- no "
                    "commentary, no quotes, no explanation. "
                    "Example output: 'Break today from 7:15pm to 8pm'"
                    f"{context_block}"
                ),
            },
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,
    )
    return message["content"]


def _system_prompt():
    now = datetime.datetime.now()
    return f"""You are a casual, friendly personal assistant (like JARVIS,
but relaxed, not formal). You can read and modify the user's Google
Calendar using the tools you're given.

Today's date and time: {now.strftime("%A, %B %d, %Y, %I:%M %p")}.

Keep spoken replies short and natural, like you're talking, not writing."""


def run_turn(user_message, history):
    """
    Run one conversation turn for general chat (calendar_manager.py has
    already ruled out this being a calendar request).

    `history` is a list of prior messages -- keep it between calls so
    the assistant remembers the conversation.

    Returns (reply_text, updated_history).
    """
    messages = [{"role": "system", "content": _system_prompt()}] + history
    messages.append({"role": "user", "content": user_message})

    message = chat_completion(messages, tools=[get_todays_events, create_event])
    messages.append(message)

    tool_calls = message["tool_calls"]
    if tool_calls:
        for call in tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [calling tool: {name}({args})]")
            func = AVAILABLE_TOOLS[name]
            result = func() if name == "get_todays_events" else func(**args)
            messages.append({"role": "tool", "content": str(result), "name": name})

        follow_up = chat_completion(messages)
        messages.append(follow_up)
        reply = follow_up["content"]
    else:
        reply = message["content"]

    return reply, messages[1:]  # drop the system prompt from saved history
