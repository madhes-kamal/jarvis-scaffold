"""
calendar_manager.py

Sits between the voice/conversation layer and calendar_service.py. Owns
the decisions that are too risky (or too easy to get wrong) for a small
local model to handle alone:

1. What kind of request is this? (classify_intent)
2. Which real event does the user mean? (_resolve_target)
3. If we just asked a clarifying question, does THIS message answer it?
   (the `pending` mechanism below) -- without this, a short answer like
   "the 7:15 one" gets classified fresh as if it were an unrelated new
   command, which is exactly what was going wrong before.

The LLM is used for narrow jobs only: classifying intent, and (via
extract_event_phrase) cleaning up a create-event phrase. It never picks
an event ID out of thin air, and it never decides a destructive action
happened without Python resolving a real target first.
"""

import datetime
import re

from dateparser.search import search_dates

from calendar_service import get_todays_events, create_event, delete_event, update_event
from conversation import extract_event_phrase
from llm_client import chat_completion

VALID_INTENTS = {"CREATE", "READ", "UPDATE", "DELETE", "NONE"}


def classify_intent(user_text):
    """One-word classification. Few-shot examples matter a lot here --
    abstract instructions alone weren't enough to stop a small model
    from treating a complaint or a question about a past action as a
    new command."""
    message = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's message into exactly one word: "
                    "CREATE, READ, UPDATE, DELETE, or NONE.\n\n"
                    "CREATE = asking to add a brand new event.\n"
                    "READ = asking what's scheduled.\n"
                    "UPDATE = asking to change or move an EXISTING event.\n"
                    "DELETE = asking to cancel or remove an EXISTING event.\n"
                    "NONE = anything else -- including questions, "
                    "complaints, or asking whether something ALREADY "
                    "happened. Asking about a past action is NOT a new "
                    "command.\n\n"
                    "Examples:\n"
                    "'add a break at 3pm' -> CREATE\n"
                    "'what's on my calendar' -> READ\n"
                    "'move my meeting to 4' -> UPDATE\n"
                    "'cancel my dentist appointment' -> DELETE\n"
                    "'why is it at 250 and I wanted 230' -> NONE\n"
                    "'did you delete the break and add one at 4?' -> NONE\n"
                    "'that's not what I meant' -> NONE\n"
                    "'tell me a joke' -> NONE\n\n"
                    "Reply with ONLY the single word, nothing else."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        temperature=0.0,
    )
    label = message["content"].strip().upper()
    for intent in VALID_INTENTS:
        if intent in label:
            return intent
    return "NONE"


def _match_events(user_text, events):
    """Find which events the message could be referring to.

    Time is treated as the stronger, more specific signal: if a time is
    mentioned, we narrow to events at that hour FIRST, then use any
    remaining meaningful words to narrow further. This matters because
    generic titles like "Break" match almost every event on overlap
    alone -- without narrowing by time first, mentioning "7:15pm"
    was being ignored entirely, since title overlap alone was enough
    to include an event regardless of whether the time matched.
    """
    lowered = user_text.lower()
    text_words = set(re.findall(r"[a-z]+", lowered))
    time_match = re.search(r"\b(\d{1,2})(:\d{2})?\s*(am|pm)?\b", lowered)

    candidates = list(events)

    if time_match:
        hour = time_match.group(1)
        by_time = [e for e in candidates if hour == e["start"].lstrip("0").split(":")[0]]
        if by_time:
            candidates = by_time

    generic_words = {
        "delete", "cancel", "remove", "add", "create", "schedule", "move",
        "change", "update", "reschedule", "the", "my", "at", "on", "from",
        "to", "a", "an", "please", "can", "you", "could", "would", "am", "pm",
    }
    meaningful_words = text_words - generic_words

    by_title = [
        e for e in candidates
        if meaningful_words & set(re.findall(r"[a-z]+", e["summary"].lower()))
    ]
    if by_title:
        candidates = by_title

    return candidates


def _resolve_target(user_text, events, context):
    """Returns ("NONE", None) / ("ONE", event) / ("MANY", [events])."""
    candidates = _match_events(user_text, events)

    if not candidates:
        lowered = user_text.lower()
        if ("that" in lowered or " it " in f" {lowered} ") and context.get("last_event"):
            return "ONE", context["last_event"]
        return "NONE", None

    if len(candidates) == 1:
        return "ONE", candidates[0]

    # If every remaining candidate is functionally identical (same
    # title and time -- true duplicates), asking "which one?" gives the
    # user no real choice to make. Just act on the first one.
    signatures = {(e["summary"], e["start"], e["end"]) for e in candidates}
    if len(signatures) == 1:
        return "ONE", candidates[0]

    return "MANY", candidates


def _local_iso(dt):
    return dt.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo).isoformat()


def _extract_new_time(text_for_time, base_date):
    """Find a new time inside a sentence that has other words around it
    (e.g. "move it to 4pm"). Uses search_dates, not parse -- parse
    expects the WHOLE string to be a date and fails/gets confused on
    mixed text or when two times appear in one sentence. If more than
    one time is mentioned ("from 7:15 to 4pm"), we take the LAST one,
    matching how people actually phrase "move X from OLD to NEW"."""
    found = search_dates(
        text_for_time,
        settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base_date},
    )
    if not found:
        return None
    return found[-1][1]  # list of (matched_text, datetime); take the last


def _do_delete(target, context):
    delete_event(target["id"])
    context["last_event"] = None
    context["pending"] = None
    return f"Deleted {target['summary']} at {target['start']}.", context


def _do_update(target, text_for_time, context):
    base_date = datetime.datetime.fromisoformat(target["start_iso"])
    new_start = _extract_new_time(text_for_time, base_date)

    if not new_start:
        return (
            f"I found {target['summary']}, but couldn't tell what new time you want -- "
            "try saying something like 'move it to 4pm'."
        ), context

    duration = datetime.datetime.fromisoformat(target["end_iso"]) - base_date
    new_end = new_start + duration

    update_event(target["id"], new_start_iso=_local_iso(new_start), new_end_iso=_local_iso(new_end))
    context["last_event"] = target
    context["pending"] = None
    return f"Moved {target['summary']} to {new_start.strftime('%I:%M %p')}.", context


def handle_calendar_request(user_text, context, skip_read=False):
    """
    Main entry point. Returns (reply_text, updated_context).

    context keys:
    - "last_event": the most recently discussed event, for "move that"/"delete it"
    - "pending": set when we just asked a clarifying question. The NEXT
      call checks THIS first, so a short answer resolves against the
      candidates we already offered instead of being classified fresh
      as if it were an unrelated new request.
    """
    pending = context.get("pending")
    if pending:
        matches = _match_events(user_text, pending["candidates"])
        if len(matches) == 1:
            target = matches[0]
            context["pending"] = None
            if pending["intent"] == "DELETE":
                return _do_delete(target, context)
            if pending["intent"] == "UPDATE":
                # the new time might've been in the original request or
                # in this clarifying reply -- check both
                combined_text = f"{pending['original_text']} {user_text}"
                return _do_update(target, combined_text, context)
        # Still can't tell -- drop the pending state and let it
        # classify fresh rather than getting stuck in a loop forever.
        context["pending"] = None

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
        phrase = extract_event_phrase(user_text, events_context=events)
        print(f"  [quick-add phrase: {phrase}]")

        looks_like_question = "?" in phrase or phrase.lower().startswith(
            ("why", "what", "how", "who", "when is")
        )
        has_a_time = bool(search_dates(phrase))

        if looks_like_question or not has_a_time:
            return (
                "I couldn't pin down a clear event and time from that -- "
                "mind rephrasing it, like 'add a break at 3pm'?"
            ), context

        confirmation = create_event(phrase)
        return confirmation, context

    # UPDATE and DELETE both need a specific real event, resolved by ID.
    status, result = _resolve_target(user_text, events, context)

    if status == "NONE":
        return (
            "I couldn't find an event matching that -- "
            "can you tell me the title or time?"
        ), context

    if status == "MANY":
        options = ", ".join(f"{e['summary']} at {e['start']}" for e in result)
        context["pending"] = {
            "intent": intent,
            "candidates": result,
            "original_text": user_text,
        }
        return f"I found a few things that could match: {options}. Which one did you mean?", context

    target = result  # status == "ONE"

    if intent == "DELETE":
        return _do_delete(target, context)

    if intent == "UPDATE":
        return _do_update(target, user_text, context)

    return "Something went wrong handling that calendar request.", context
