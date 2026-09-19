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
import difflib
import re

from dateparser.search import search_dates

from calendar_service import (
    get_todays_events, create_event, delete_event, update_event, set_event_color,
)
from conversation import extract_event_phrase
from llm_client import chat_completion

VALID_INTENTS = {"CREATE", "READ", "UPDATE", "DELETE", "COLOR", "NONE"}

# Spoken color -> Google Calendar colorId (see calendar_service.EVENT_COLORS
# for the palette). Google's palette has odd names (Tomato, Peacock...), so
# everyday words are mapped to the nearest one. Order matters: multi-word
# phrases ("light blue") must be tried before their single-word tails.
_COLOR_WORDS = [
    (r"light blue|sky blue|teal|turquoise|cyan|peacock", "7"),
    (r"light green|sage|mint", "2"),
    (r"light purple|lavender|lilac", "1"),
    (r"dark blue|navy|indigo|blueberry", "9"),
    (r"dark green|basil", "10"),
    (r"purple|violet|grape", "3"),
    (r"pink|flamingo", "4"),
    (r"yellow|banana", "5"),
    (r"orange|tangerine", "6"),
    (r"red|tomato", "11"),
    (r"blue", "9"),
    (r"green", "10"),
    (r"gray|grey|graphite|silver", "8"),
]
_COLOR_RESET = re.compile(
    r"\b(default|original|normal|reset)\b|\bno colou?r\b|\bremove (?:the |its |my )?colou?r\b",
    re.IGNORECASE,
)
# Words that describe the color request, not the event -- kept out of
# title matching so "make my break red" can't match an event called "Red team".
_COLOR_TOKENS = {
    word for pattern, _ in _COLOR_WORDS for word in re.findall(r"[a-z]+", pattern)
} | {"color", "colour", "colored", "coloured", "make", "set", "turn", "paint", "mark"}
_AFFIRMATIVE_WORDS = {"yes", "yeah", "yep", "yup", "confirm", "correct", "sure"}
_AFFIRMATIVE_PHRASES = ("do it", "go ahead")
_NEGATIVE_WORDS = {
    "no", "nope", "nah", "not", "never", "don't", "dont", "cancel", "stop",
    "wait", "negative",
}


def _words(text):
    # Whisper sometimes emits curly apostrophes ("don’t").
    return re.findall(r"[a-z']+", text.lower().replace("’", "'"))


def _is_negative(text):
    return any(word in _NEGATIVE_WORDS for word in _words(text))


def _is_affirmative(text):
    """Whole-word match, and any negation wins. Substring matching used
    to count "no, don't do it" as a yes (it contains "do it") -- on a
    delete confirmation, a false yes is the expensive mistake, so when
    in doubt this says no."""
    if _is_negative(text):
        return False
    words = _words(text)
    if any(word in _AFFIRMATIVE_WORDS for word in words):
        return True
    joined = " ".join(words)
    return any(phrase in joined for phrase in _AFFIRMATIVE_PHRASES)


def _normalize_bare_times(text):
    """Turn a bare 3-4 digit cluster into a colon-separated time."""
    def _replace(match):
        digits = match.group(1)
        if len(digits) == 3:
            return f"{digits[0]}:{digits[1:]}"
        if len(digits) == 4 and not (1900 <= int(digits) <= 2100):
            return f"{digits[:2]}:{digits[2:]}"
        return digits

    return re.sub(r"\b(\d{3,4})\b", _replace, text)


def _normalize_meridian(text):
    """Normalize period-separated am/pm before time detection."""
    return re.sub(r"\b([ap])\.?\s*m\.?\b", r"\1m", text, flags=re.IGNORECASE)


def awaiting_answer(context):
    """True if the last reply asked the user a question (confirm or
    "which one?") that the next utterance is meant to answer."""
    return bool(context.get("confirm_pending") or context.get("pending"))


def clear_pending(context):
    """Abandon any unanswered question, e.g. after the user stayed silent."""
    context.pop("confirm_pending", None)
    context["pending"] = None


def _fuzzy_word_overlap(words_a, words_b, threshold=0.75):
    for word_a in words_a:
        if difflib.get_close_matches(word_a, words_b, n=1, cutoff=threshold):
            return True
    return False


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
                    "CREATE, READ, UPDATE, DELETE, COLOR, or NONE.\n\n"
                    "CREATE = asking to add a brand new event.\n"
                    "READ = asking what's scheduled.\n"
                    "UPDATE = asking to change the TIME of an EXISTING event.\n"
                    "DELETE = asking to cancel or remove an EXISTING event.\n"
                    "COLOR = asking to change the COLOR of an EXISTING event.\n"
                    "NONE = anything else -- including questions, "
                    "complaints, or asking whether something ALREADY "
                    "happened. Asking about a past action is NOT a new "
                    "command.\n\n"
                    "Examples:\n"
                    "'add a break at 3pm' -> CREATE\n"
                    "'what's on my calendar' -> READ\n"
                    "'move my meeting to 4' -> UPDATE\n"
                    "'cancel my dentist appointment' -> DELETE\n"
                    "'make my break red' -> COLOR\n"
                    "'change the color of chemistry to blue' -> COLOR\n"
                    "'turn it green' -> COLOR\n"
                    "'what color is my break' -> NONE\n"
                    "'my favorite color is red' -> NONE\n"
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


# "move my break to 4.30pm": everything from "to <time>" onward is the
# DESTINATION, not a description of which event is meant. The lookahead
# requires a digit so titles containing "to" ("go to gym") aren't cut.
_DESTINATION_SPLIT = re.compile(
    r"\b(?:to|until|till)\s+(?=(?:around\s+|about\s+|like\s+)?\d)", re.IGNORECASE
)


def _split_target_and_destination(text):
    """Returns (part identifying the event, part holding the new time).
    With no "to <time>" marker, both are the whole text."""
    match = _DESTINATION_SPLIT.search(text)
    if not match:
        return text, text
    return text[: match.start()], text[match.start():]


def _starts_at_hour(event, hour, meridian):
    """Does `event` start at the spoken hour? With am/pm, only that exact
    hour of the day matches. Without it, 7 matches both 7 AM and 7 PM
    (the user didn't say which), and 13-23 / 0 are read as 24-hour."""
    if event["start"] == "All day":
        return False
    start_hour = datetime.datetime.fromisoformat(event["start_iso"]).hour
    if meridian:
        return start_hour == hour % 12 + (12 if meridian == "pm" else 0)
    if hour > 12 or hour == 0:
        return start_hour == hour
    return start_hour % 12 == hour % 12


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
    target_text, _ = _split_target_and_destination(user_text)
    lowered = _normalize_meridian(target_text.lower())
    lowered = _normalize_bare_times(lowered)
    text_words = set(re.findall(r"[a-z]+", lowered))
    time_match = re.search(r"\b(\d{1,2})(:\d{2})?\s*(am|pm)?\b", lowered)

    candidates = list(events)

    if time_match:
        hour = int(time_match.group(1))
        meridian = time_match.group(3)
        candidates = [e for e in candidates if _starts_at_hour(e, hour, meridian)]
    print(f"  [match: {len(candidates)} after time filter: {[e['summary'] for e in candidates]}]")

    generic_words = {
        "delete", "cancel", "remove", "add", "create", "schedule", "move",
        "change", "update", "reschedule", "the", "my", "at", "on", "from",
        "to", "a", "an", "please", "can", "you", "could", "would", "am", "pm",
    }
    meaningful_words = text_words - generic_words - _COLOR_TOKENS

    by_title = [
        e for e in candidates
        if _fuzzy_word_overlap(
            meaningful_words, set(re.findall(r"[a-z]+", e["summary"].lower()))
        )
    ]
    if by_title:
        candidates = by_title
    print(f"  [match: {len(candidates)} after title filter: {[e['summary'] for e in candidates]}]")

    return candidates


def _resolve_target(user_text, events, context):
    """Returns ("NONE", None) / ("ONE", event) / ("MANY", [events])."""
    candidates = _match_events(user_text, events)

    # "move it" / "make that red": nothing in the message singled an event
    # out (no candidates, or the matcher couldn't narrow the full list),
    # so a pronoun means the event we were just talking about.
    refers_back = re.search(r"\b(it|that|this)\b", user_text, re.IGNORECASE)
    if refers_back and context.get("last_event"):
        if not candidates or len(candidates) == len(events):
            return "ONE", context["last_event"]

    if not candidates:
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


_DAY_WORDS = re.compile(
    r"\b(tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"\d{1,2}(st|nd|rd|th))\b",
    re.IGNORECASE,
)


def _extract_new_time(text_for_time, base_date):
    """Find a new time inside a sentence that has other words around it
    (e.g. "move it to 4pm" or "move it to 4.30"). Uses search_dates, not
    parse -- parse expects the WHOLE string to be a date. Prefers a match
    that looks like an actual clock time (colon OR period separator,
    am/pm, etc.) over a bare date. Unless the message explicitly names a
    different day, the result is forced to stay on base_date's day --
    dateparser was silently jumping the result to a different day
    entirely when given an ambiguous bare time with no am/pm."""
    text_for_time = _normalize_meridian(text_for_time)
    text_for_time = _normalize_bare_times(text_for_time)
    found = search_dates(
        text_for_time,
        settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": base_date},
    )
    print(f"  [search_dates found: {found}]")
    if not found:
        return None

    time_like = [
        (t, dt) for t, dt in found
        if re.search(r"(:\d{2}|\.\d{2}|am|pm|o'clock|noon|midnight)", t.lower())
    ]
    if not time_like:
        return None
    matched_text, new_dt = time_like[-1]

    if not _DAY_WORDS.search(text_for_time):
        new_dt = new_dt.replace(
            year=base_date.year, month=base_date.month, day=base_date.day
        )

    return new_dt


def _do_delete(target, context):
    delete_event(target["id"])
    context["last_event"] = None
    context["pending"] = None
    return f"Deleted {target['summary']} at {target['start']}.", context


def _extract_color(text):
    """Find the color the user asked for. Returns (colorId, spoken label),
    (None, "default") for a reset, or None if no color was mentioned.
    Decided in Python, never by the LLM. If several colors are named
    ("make my red break blue") the LAST one is the destination."""
    if _COLOR_RESET.search(text):
        return None, "default"

    remaining = text.lower()
    found = []
    for pattern, color_id in _COLOR_WORDS:
        for match in re.finditer(rf"\b(?:{pattern})\b", remaining):
            found.append((match.start(), color_id, match.group(0)))
            # Blank the span so later, shorter patterns ("blue") can't
            # re-match inside an earlier one ("light blue").
            remaining = (
                remaining[: match.start()]
                + " " * (match.end() - match.start())
                + remaining[match.end():]
            )
    if not found:
        return None
    _, color_id, label = max(found)
    return color_id, label


def _do_color(target, text, context):
    color = _extract_color(text)
    if color is None:
        return (
            f"I found {target['summary']}, but which color? "
            "Try saying something like 'make it red'."
        ), context

    color_id, label = color
    set_event_color(target["id"], color_id)
    context["last_event"] = target
    context["pending"] = None
    if color_id is None:
        return f"Reset {target['summary']} to its default color.", context
    return f"Made {target['summary']} {label}.", context


def _apply_to_target(intent, target, text, context):
    """Carry out an intent on one resolved event. Destructive or
    time-changing actions ask for confirmation first; recoloring is
    trivially reversible, so it just happens."""
    if intent == "COLOR":
        return _do_color(target, text, context)
    return _prepare_confirmation(intent, target, text, context)


def _prepare_confirmation(intent, target, text_for_time, context):
    if intent == "DELETE":
        context["confirm_pending"] = {
            "intent": intent,
            "target": target,
        }
        return f"Delete {target['summary']} at {target['start']}?", context

    base_date = datetime.datetime.fromisoformat(target["start_iso"])
    _, destination_text = _split_target_and_destination(text_for_time)
    new_start = _extract_new_time(destination_text, base_date)
    if not new_start:
        return (
            f"I found {target['summary']}, but couldn't tell what new time you want -- "
            "try saying something like 'move it to 4pm'."
        ), context

    duration = datetime.datetime.fromisoformat(target["end_iso"]) - base_date
    new_end = new_start + duration
    context["confirm_pending"] = {
        "intent": intent,
        "target": target,
        "new_start_iso": _local_iso(new_start),
        "new_end_iso": _local_iso(new_end),
        "new_start": new_start,
    }
    return f"Move {target['summary']} to {new_start.strftime('%I:%M %p')}?", context


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
    confirm_pending = context.pop("confirm_pending", None)
    if confirm_pending:
        if _is_affirmative(user_text):
            target = confirm_pending["target"]
            if confirm_pending["intent"] == "DELETE":
                return _do_delete(target, context)

            update_event(
                target["id"],
                new_start_iso=confirm_pending["new_start_iso"],
                new_end_iso=confirm_pending["new_end_iso"],
            )
            context["last_event"] = target
            context["pending"] = None
            return (
                f"Moved {target['summary']} to "
                f"{confirm_pending['new_start'].strftime('%I:%M %p')}.",
                context,
            )
        if _is_negative(user_text):
            return "Okay, I won't make that change.", context
        # Neither yes nor no: the user moved on to something else. Drop
        # the change and handle this message as a fresh request instead
        # of swallowing it.

    pending = context.get("pending")
    if pending:
        matches = _match_events(user_text, pending["candidates"])
        if len(matches) == 1:
            target = matches[0]
            context["pending"] = None
            # The answer ("the 7:15 one") only picks the event; the new
            # time comes from the original request, so a time in the
            # answer can't be mistaken for the destination.
            return _apply_to_target(
                pending["intent"], target, pending["original_text"], context
            )
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
        looks_incomplete = phrase.strip().endswith(("...", " to", " from", "-"))
        has_a_time = bool(search_dates(phrase))

        if looks_like_question or looks_incomplete or not has_a_time:
            return (
                "I couldn't pin down a clear event and time from that -- "
                "mind rephrasing it, like 'add a break at 3pm'?"
            ), context

        new_event = create_event(phrase)
        context["last_event"] = new_event
        return f"Added {new_event['summary']} from {new_event['start']} to {new_event['end']}.", context

    # UPDATE, DELETE and COLOR all need a specific real event, resolved by ID.
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

    return _apply_to_target(intent, result, user_text, context)  # status == "ONE"
