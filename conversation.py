"""
conversation.py

General chat fallback for anything calendar_manager.py decides is NOT a
calendar request, plus extract_event_phrase() which calendar_manager.py
reuses for turning messy speech into a clean create-event phrase.

All LLM calls go through llm_client.chat_completion() -- see that file
for why. Nothing here calls ollama directly or hardcodes a model name.
"""

import datetime

from calendar_service import get_upcoming_events
from llm_client import chat_completion

# Read-only on purpose. Adding, moving, resizing, deleting and recoloring events
# are all done by calendar_manager.py, which works out dates and times itself;
# letting this small model create events from a string it wrote had it
# guessing dates ("next Friday" came out as a Sunday).
AVAILABLE_TOOLS = {
    "get_upcoming_events": get_upcoming_events,
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
                    "If the user doesn't specify how long the event lasts, "
                    "default to a 30 minute duration and state both a start "
                    "and end time explicitly -- never leave the phrase "
                    "incomplete or trail off with '...'. "
                    "Example output: 'Break today from 7:15pm to 8pm'"
                    f"{context_block}"
                ),
            },
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,
    )
    return message["content"]


def extract_event_title(user_text):
    """Just the TITLE of the event being created: "Schedule a chemistry test
    next Friday at 2.30 p.m." -> "Chemistry test". Dates and times are worked
    out in Python (calendar_manager._parse_new_event); a small model given
    the whole job turned "next Friday" into a Sunday, so it only names the
    event now. Returns "" if the reply looks unusable."""
    message = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the title of the calendar event the user wants "
                    "to create. Reply with ONLY the title: 1 to 4 words, no "
                    "quotes, no final punctuation. Leave out the command "
                    "(add, schedule, create), dates, times and durations.\n\n"
                    "Examples:\n"
                    "'Schedule a chemistry test next Friday at 2.30 p.m.' -> Chemistry test\n"
                    "'add a 5 minute scrolling break starting now' -> Scrolling break\n"
                    "'put dinner with grandma tomorrow from 7 to 8pm' -> Dinner with grandma\n"
                    "'create a study session for physics on Thursday at 5' -> Physics study session"
                ),
            },
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    lines = message["content"].strip().splitlines()
    title = lines[0].strip().strip("\"'").rstrip(".!") if lines else ""
    return title if 1 <= len(title.split()) <= 8 else ""


def extract_search_target(user_text):
    """The specific thing (if any) a calendar READ is asking about --
    "chemistry test" from "do I have a chemistry test this week", or ""
    when the message is really just a generic "what's on my calendar" with
    nothing specific named.

    calendar_manager.py used to answer this with plain stopword-removal
    over the whole raw transcript, which had no way to tell a real search
    term apart from chit-chat or a misheard wake word ("Google, Google, how
    are you doing bro, what do I have scheduled after today" was leaving
    behind {'after', 'bro', 'doing', 'google'} as "search terms", so nothing
    ever matched). This is the same kind of narrow, single-phrase job as
    extract_event_title -- reading intent out of messy speech -- just for
    reading instead of creating."""
    message = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "The user is asking about their calendar. If they're asking "
                    "whether a SPECIFIC named event, appointment, test, or activity "
                    "is scheduled, reply with ONLY that thing, 1-4 words. If they're "
                    "just asking generally what's on their schedule/calendar/day/"
                    "week, with nothing specific named, reply with exactly NONE. "
                    "Ignore any greeting, small talk, or filler in the message -- "
                    "it is never part of what to search for.\n\n"
                    "Examples:\n"
                    "'do I have a chemistry test this week' -> chemistry test\n"
                    "'is there a dentist appointment tomorrow' -> dentist appointment\n"
                    "'any meetings this week' -> meetings\n"
                    "'what do I have today' -> NONE\n"
                    "'what's on my calendar' -> NONE\n"
                    "'how does my week look' -> NONE\n"
                    "'hey buddy what do I have scheduled after today' -> NONE\n"
                    "'how many things do I have going on friday' -> NONE\n\n"
                    "Reply with ONLY the phrase or NONE, nothing else."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    text = message["content"].strip().strip("\"'").rstrip(".!")
    if not text or text.upper() == "NONE" or len(text.split()) > 5:
        return ""
    return text


def _system_prompt():
    now = datetime.datetime.now()
    return f"""You are a casual, friendly personal assistant (like JARVIS,
but relaxed, not formal), talking out loud to the user.

You are connected to the user's Google Calendar. Never say you can't access
it. If they ask what's on their schedule, call get_upcoming_events right away
(days=1 is the rest of today, days=7 the next week) -- never offer to check
or ask permission first.

Changing the calendar is handled by other parts of this app, and only works
when the request is specific. If the user asks you to change something, do
NOT refuse and do not pretend you did it. Ask them to say it plainly, for
example: "add a break at 3pm", "move my break to 4pm", "make my break 30
minutes long", "rename my break to lunch", "delete the 7pm one", or "make it red". The app can add,
move, resize, rename, delete and recolor events, and plan study blocks. It
can't invite people or set reminders yet -- if asked for one of those, say
plainly that it can't do that yet.

Today's date and time: {now.strftime("%A, %B %d, %Y, %I:%M %p")}.

Keep spoken replies short and natural, like you're talking, not writing.
Do not use emojis in any response."""


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

    message = chat_completion(messages, tools=[get_upcoming_events])
    messages.append(message)

    tool_calls = message["tool_calls"]
    if tool_calls:
        for call in tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [calling tool: {name}({args})]")
            func = AVAILABLE_TOOLS[name]
            result = func(**args)
            messages.append({"role": "tool", "content": str(result), "name": name})

        follow_up = chat_completion(messages)
        messages.append(follow_up)
        reply = follow_up["content"]
    else:
        reply = message["content"]

    return reply, messages[1:]  # drop the system prompt from saved history
