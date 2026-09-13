"""
calendar_manager.py

Sits between the voice/conversation layer and calendar_service.py. Owns
the two decisions that are too risky to hand to a small local model:

1. What kind of request is this? (classify_intent -- a one-word
   classification, much easier for a small model than deciding whether
   to call a function and building correct arguments)
2. Which real event does the user mean? (_resolve_target -- deterministic
   matching against actual event IDs, never a guess. If it's ambiguous
   or unclear, we ask the user rather than picking for them -- especially
   important for delete, which is destructive and can't be undone here.)

The LLM is used for two narrow jobs only: classifying intent, and (via
extract_event_phrase) cleaning up a create-event phrase. It never picks
an event ID out of thin air.
"""

import datetime
import re

import ollama
import dateparser

from calendar_service import get_todays_events, create_event, delete_event, update_event
from conversation import extract_event_phrase, MODEL

VALID_INTENTS = {"CREATE", "READ", "UPDATE", "DELETE", "NONE"}


def classify_intent(user_text):
    """One-word classification -- a much easier task for a small model
    than tool calling, since it's picking a label, not generating
    structured arguments."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's message into exactly one word: "
                    "CREATE (adding a new event), READ (asking what's "
                    "scheduled), UPDATE (changing or moving an existing "
                    "event), DELETE (cancelling or removing an event), "
                    "or NONE (not calendar-related at all). "
                    "Reply with ONLY that single word, nothing else."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        options={"temperature": 0.0},
    )
    label = response.message.content.strip().upper()
    for intent in VALID_INTENTS:
        if intent in label:
            return intent
    return "NONE"


def _find_candidates(user_text, events):
    """Simple, deterministic matching: does the message share a word
    with the event's title, or mention its hour? No guessing -- if
    nothing matches clearly, we return nothing rather than a best-effort
    pick. This is intentionally simple; it's a reasonable starting point,
    not a full NLP matcher."""
    lowered = user_text.lower()
    text_words = set(re.findall(r"[a-z]+", lowered))
    time_match = re.search(r"\b(\d{1,2})(:\d{2})?\s*(am|pm)?\b", lowered)

    candidates = []
    for e in events:
        title_words = set(re.findall(r"[a-z]+", e["summary"].lower()))
        title_overlap = bool(title_words & text_words)

        time_overlap = False
        if time_match:
            hour = time_match.group(1)
            time_overlap = hour == e["start"].lstrip("0").split(":")[0]

        if title_overlap or time_overlap:
            candidates.append(e)

    return candidates


def _resolve_target(user_text, events, context):
    """Returns ("NONE", None) / ("ONE", event) / ("MANY", [events])."""
    candidates = _find_candidates(user_text, events)

    if not candidates:
        # "that"/"it" can refer back to whatever was last discussed.
        lowered = user_text.lower()
        if ("that" in lowered or " it " in f" {lowered} ") and context.get("last_event"):
            return "ONE", context["last_event"]
        return "NONE", None

    if len(candidates) == 1:
        return "ONE", candidates[0]

    return "MANY", candidates


def _local_iso(dt):
    return dt.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo).isoformat()


def handle_calendar_request(user_text, context, skip_read=False):
    """
    Main entry point. Returns (reply_text, updated_context).

    `context` is a plain dict carried across turns -- currently just
    {"last_event": <event dict or None>}, enough to resolve "move that
    to 4" after a previous read/create/update/delete.

    `reply_text` is None when this wasn't a calendar request at all
    (intent NONE) -- the caller should fall through to general
    conversation in that case. `skip_read=True` also returns None for a
    READ request, for when you've already spoken a briefing this turn
    and don't want a second, less natural readout of the same data.
    """
    intent = classify_intent(user_text)
    print(f"  [calendar intent: {intent}]")

    if intent == "NONE":
        return None, context

    if intent == "READ" and skip_read:
        return None, context

    events = get_todays_events()

    if intent == "READ":
        if not events:
            return "You've got nothing on the calendar today.", context
        lines = [f"{e['summary']} at {e['start']}" for e in events]
        context["last_event"] = events[0]
        return "Today: " + "; ".join(lines) + ".", context

    if intent == "CREATE":
        phrase = extract_event_phrase(user_text)
        print(f"  [quick-add phrase: {phrase}]")
        confirmation = create_event(phrase)
        return confirmation, context

    # UPDATE and DELETE both need a specific real event, resolved by ID
    # -- never by re-describing it and hoping the API matches the right one.
    status, result = _resolve_target(user_text, events, context)

    if status == "NONE":
        return (
            "I couldn't find an event matching that -- "
            "can you tell me the title or time?"
        ), context

    if status == "MANY":
        options = ", ".join(f"{e['summary']} at {e['start']}" for e in result)
        return f"I found a few things that could match: {options}. Which one did you mean?", context

    target = result  # status == "ONE"

    if intent == "DELETE":
        delete_event(target["id"])
        context["last_event"] = None
        return f"Deleted {target['summary']} at {target['start']}.", context

    if intent == "UPDATE":
        base_date = datetime.datetime.fromisoformat(target["start_iso"])
        # dateparser, not the LLM, extracts the new time -- a parsing
        # library is more reliable here than asking a small model to
        # output an exact time format.
        new_start = dateparser.parse(
            user_text,
            settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base_date},
        )
        if not new_start:
            return (
                f"I found {target['summary']}, but couldn't tell what new time you want -- "
                "try saying something like 'move it to 4pm'."
            ), context

        duration = datetime.datetime.fromisoformat(target["end_iso"]) - base_date
        new_end = new_start + duration

        update_event(
            target["id"],
            new_start_iso=_local_iso(new_start),
            new_end_iso=_local_iso(new_end),
        )
        context["last_event"] = target
        return f"Moved {target['summary']} to {new_start.strftime('%I:%M %p')}.", context

    return "Something went wrong handling that calendar request.", context
