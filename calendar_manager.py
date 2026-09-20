"""
calendar_manager.py

Sits between the voice/conversation layer and calendar_service.py. Owns
the decisions that are too risky (or too easy to get wrong) for a small
local model to handle alone:

1. What kind of request is this? (classify_intent -- clock questions and
   "am I free" are caught by regex before the model is even asked)
2. Which real event does the user mean? (_resolve_target -- today plus the
   next TARGET_DAYS days; date arithmetic lives in time_utils.py)
3. If we just asked a clarifying question, does THIS message answer it?
   (the `pending` mechanism below) -- without this, a short answer like
   "the 7:15 one" gets classified fresh as if it were an unrelated new
   command, which is exactly what was going wrong before.

The LLM is used for narrow jobs only: classifying intent, and naming a
new event (extract_event_title). It never picks an event ID out of thin air,
never works out a date or time (a small model turned "next Friday" into a
Sunday), and never decides a destructive action happened without Python
resolving a real target first.
"""

import datetime
import difflib
import re
import traceback
from collections import namedtuple

from dateparser.search import search_dates

import time_utils
from calendar_service import (
    get_events, event_day, has_not_ended, insert_event, insert_events,
    delete_event, delete_events, update_event, set_event_color,
    set_event_colors,
)
from conversation import extract_event_title
from llm_client import chat_completion

VALID_INTENTS = {"CREATE", "READ", "UPDATE", "DELETE", "COLOR", "TIME", "NONE"}

# How far ahead update/delete/color look for the event you mean (in days,
# counting today), and how far a search like "do I have a chemistry test?"
# looks when you didn't say when.
TARGET_DAYS = 7
SEARCH_DAYS = 14
# Most events read out in one reply -- past this it's a wall of text.
MAX_SPOKEN_EVENTS = 8

# Asking for the clock or the date. Matched in Python before the LLM is
# consulted: it's unambiguous, faster, and the small model shouldn't be
# trusted with it. "what time is my meeting" deliberately doesn't match.
_TIME_QUESTION = re.compile(
    r"\bwhat(?:'s|s| is)\s+the\s+(?:current\s+)?time\b(?!\s+(?:of|for|on|does|do)\b)"
    r"|\bwhat\s+time\s+is\s+it\b|\bcurrent\s+time\b|\b(?:tell|give)\s+me\s+the\s+time\b"
    r"|\bwhat(?:'s|s| is)\s+(?:the|today's)\s+date\b(?!\s+(?:of|for)\b)"
    r"|\bwhat\s+(?:day|date)\s+(?:of\s+the\s+week\s+)?is\s+(?:it|today)\b"
    r"|\btoday's\s+date\b",
    re.IGNORECASE,
)
_WANTS_DATE = re.compile(r"\b(date|day)\b", re.IGNORECASE)
_WANTS_TIME = re.compile(r"\btime\b", re.IGNORECASE)
# "what time is my dentist appointment" / "what time does chemistry start":
# a question about an event, not about the clock.
_EVENT_TIME_QUESTION = re.compile(r"\bwhat\s+time\s+(?:is|does|do|will|are)\s+(?!it\b)", re.IGNORECASE)
_FREE_OR_BUSY = re.compile(r"\b(?:am\s+i|are\s+we)\s+(?:free|busy)\b", re.IGNORECASE)

# Ways of asking for the schedule that the small model kept sending to chat
# ("give me a rundown for tomorrow", "summary of my week"). Decided here in
# Python: asking for a summary/rundown/overview, naming the schedule or
# calendar, or asking about "my day/week" -- unless the message is an edit
# ("add this to my calendar", "move my schedule...").
_EDIT_VERBS = re.compile(
    r"\b(add|create|book|put|set|move|reschedule|delete|cancel|remove|change|update)\b",
    re.IGNORECASE,
)
_SUMMARY_WORDS = re.compile(
    r"\b(rundown|run\s+down|summary|summari[sz]e|overview|recap|breakdown|agenda|"
    r"briefing|brief\s+me|catch\s+me\s+up)\b",
    re.IGNORECASE,
)
_SCHEDULE_NOUNS = re.compile(
    r"\b(?:my|the|today's|tomorrow's|this\s+week's)\s+(?:schedule|calendar|agenda|plans)\b"
    r"|\bon\s+(?:my\s+)?calendar\b|\b(?:any|what)\s+plans\b",
    re.IGNORECASE,
)
_DAY_OR_WEEK_QUESTION = re.compile(
    r"\b(?:what|what's|whats|how|how's|hows|show|give|tell|check|read|look|see)\b.*"
    r"\b(?:my|the)\s+(?:day|week)\b",
    re.IGNORECASE,
)
# "and tomorrow?" / "what about friday?" right after a schedule answer.
_FOLLOW_UP_DAY = re.compile(r"^\W*(?:and\s+|what\s+about\s+|how\s+about\s+)", re.IGNORECASE)


def _is_schedule_request(text):
    if _EDIT_VERBS.search(text):
        return False
    if _SUMMARY_WORDS.search(text) or _SCHEDULE_NOUNS.search(text) or _DAY_OR_WEEK_QUESTION.search(text):
        return True
    return bool(
        _FOLLOW_UP_DAY.search(text)
        and time_utils.parse_date_range(text, time_utils.now())
    )


# Only a question about a specific thing ("do I have a chemistry TEST", "when
# is my dentist appointment") filters events by the words in it. A plain
# request for the schedule ("check my calendar for tomorrow", "yes please")
# lists everything, whatever other words come along with it.
_SPECIFIC_SEARCH = re.compile(
    r"\b(?:do|does)\s+i\s+have\b|\b(?:is|are)\s+there\b|\bany\b|"
    r"\bwhen(?:'s|\s+is|\s+are)\b|\bwhat\s+time\s+(?:is|does|do|are)\b|\bhow\s+many\b",
    re.IGNORECASE,
)
# A yes/no question about the calendar gets an answer that starts with "Yes."
_YES_NO_QUESTION = re.compile(
    r"^\W*(?:hey\s+jarvis\W+)?(?:do|does|did|am|are|is|will|have|has|can)\b", re.IGNORECASE
)

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
} | {
    "color", "colour", "colored", "coloured", "make", "set", "turn", "paint",
    "mark", "reset", "default", "original", "normal", "back",
}
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

    # The lookahead also lets "430pm" (no space) through, where there's no
    # word boundary between the digits and the "pm".
    return re.sub(r"\b(\d{3,4})(?=\b|[ap]m\b)", _replace, text)


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
    if _TIME_QUESTION.search(user_text):
        return "TIME"
    if _FREE_OR_BUSY.search(user_text) or _is_schedule_request(user_text):
        return "READ"

    message = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's message into exactly one word: "
                    "CREATE, READ, UPDATE, DELETE, COLOR, TIME, or NONE.\n\n"
                    "CREATE = asking to add a brand new event.\n"
                    "READ = asking what's scheduled, on any day, or whether "
                    "something (a test, a meeting) is coming up.\n"
                    "TIME = asking for the current time or today's date.\n"
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
                    "'what do I have tomorrow' -> READ\n"
                    "'do I have a chemistry test this week' -> READ\n"
                    "'how does my week look' -> READ\n"
                    "'what time is my dentist appointment' -> READ\n"
                    "'what time does my chemistry test start' -> READ\n"
                    "'am I free on saturday' -> READ\n"
                    "'what time is it' -> TIME\n"
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
            # The clock questions were all caught above. "what time is my
            # dentist appointment" is about the calendar, whatever the small
            # model made of the word "time".
            if intent == "TIME" and _EVENT_TIME_QUESTION.search(user_text):
                return "READ"
            return intent
    return "NONE"


# "move my break to 4.30pm": everything from "to <time>" onward is the
# DESTINATION, not a description of which event is meant. The lookahead
# requires a time or day word so titles containing "to" ("go to gym") aren't
# cut. That also keeps "move it to tomorrow at 3pm" from treating "tomorrow"
# as a description of WHICH event.
_DESTINATION_SPLIT = re.compile(
    r"\b(?:to|until|till)\s+(?=(?:around\s+|about\s+|like\s+)?"
    r"(?:\d|noon\b|midnight\b|tomorrow\b|tonight\b|today\b|next\b|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b))",
    re.IGNORECASE,
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
    # A spoken date ("september 25") would otherwise read as the hour 25.
    lowered = _normalize_meridian(time_utils.strip_date_expressions(target_text))
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
        "each", "every", "all", "both", "those", "these", "them", "of", "for",
        "and", "or", "in", "one", "ones",
    }
    meaningful_words = (
        text_words - generic_words - _COLOR_TOKENS - time_utils.RANGE_WORDS
    )

    by_title = [
        e for e in candidates
        if _fuzzy_word_overlap(
            meaningful_words, set(re.findall(r"[a-z]+", e["summary"].lower()))
        )
    ]
    if by_title:
        candidates = by_title
        # "chemistry test" should prefer "Chemistry Test" over "Chemistry
        # Study Block", which shares only one of the two words.
        by_all_words = [
            e for e in by_title
            if all(
                _fuzzy_word_overlap({word}, set(re.findall(r"[a-z]+", e["summary"].lower())))
                for word in meaningful_words
            )
        ]
        if by_all_words:
            candidates = by_all_words
    print(f"  [match: {len(candidates)} after title filter: {[e['summary'] for e in candidates]}]")

    return candidates


def _named_days(user_text):
    """The day or span of days the message says the EVENT is on ("the friday
    chemistry test", "tomorrow's break"), or None. Only the part before
    "to <time/day>" counts -- after that it's where the event is being moved."""
    target_text, _ = _split_target_and_destination(user_text)
    return time_utils.parse_date_range(target_text, time_utils.now())


def _scope_to_days(events, date_range):
    first, last = date_range.start.date(), date_range.end.date()
    return [e for e in events if first <= event_day(e) <= last]


def _resolve_target(user_text, events, context):
    """Returns ("NONE", None) / ("ONE", event) / ("MANY", [events])."""
    now = time_utils.now()
    today = now.date()
    named = _named_days(user_text)
    pool = _scope_to_days(events, named) if named else events
    candidates = _match_events(user_text, pool)
    unnarrowed = len(candidates) == len(pool)

    # "make them green" / "delete those": the events just added together.
    batch = context.get("last_batch")
    if batch and _BATCH_REF.search(user_text) and (not candidates or unnarrowed):
        return ("MANY", batch) if len(batch) > 1 else ("ONE", batch[0])

    # No day named: an event today is the likelier meaning than a same-named
    # one later in the week, and asking "which one?" every time would be
    # annoying. Later days are only considered when nothing today matches.
    if not named:
        # Something already over is a worse guess than something still ahead.
        ahead = [e for e in candidates if has_not_ended(e, now)]
        if ahead:
            candidates = ahead
        todays = [e for e in candidates if event_day(e) == today]
        if todays:
            candidates = todays

    # "move it" / "make that red": nothing in the message singled an event
    # out (no candidates, or the matcher couldn't narrow the full list),
    # so a pronoun means the event we were just talking about.
    refers_back = re.search(r"\b(it|that|this)\b", user_text, re.IGNORECASE)
    if refers_back and context.get("last_event"):
        if not candidates or unnarrowed:
            return "ONE", context["last_event"]

    if not candidates:
        return "NONE", None

    if len(candidates) == 1:
        return "ONE", candidates[0]

    # If every remaining candidate is functionally identical (same
    # title and time -- true duplicates), asking "which one?" gives the
    # user no real choice to make. Just act on the first one.
    # The full start/end, not the clock time: with several days in play,
    # "School 08:20 AM" on Monday and on Tuesday are different events.
    signatures = {(e["summary"], e["start_iso"], e["end_iso"]) for e in candidates}
    if len(signatures) == 1:
        return "ONE", candidates[0]

    return "MANY", candidates


def _local_iso(dt):
    return dt.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo).isoformat()


# Month names only count when a day number follows ("may 5"), so the verb
# "may" in "may you move it" isn't mistaken for a date.
_DAY_WORDS = re.compile(
    r"\b(tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|\d{1,2}(st|nd|rd|th)|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2})\b",
    re.IGNORECASE,
)


# A spoken clock time: "11am", "4:30 pm", "4.30pm", "16:00", "noon", "midnight".
# Bare hours with no am/pm and no minutes ("to 5") deliberately don't match --
# there's no telling which 5 was meant.
_CLOCK_TIME = re.compile(
    r"\b(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?\s*(?P<meridian>am|pm)\b"
    r"|\b(?P<h24>\d{1,2})[:.](?P<m24>\d{2})\b"
    r"|\b(?P<word>noon|midnight)\b",
    re.IGNORECASE,
)


def _parse_clock_time(text):
    """Return (hour, minute, span) for the LAST clock time in `text` (the
    last one, because in "move it from 3pm to 4pm" the destination is what
    matters), or None. Done with a regex rather than dateparser: dateparser
    reads "to 11am" / "to 10am" as the MONTH (November / October) and throws
    the time away, which turned every such move into 12:00 AM."""
    for match in reversed(list(_CLOCK_TIME.finditer(text))):
        if match["word"]:
            return (12 if match["word"].lower() == "noon" else 0), 0, match.span()
        if match["meridian"]:
            hour, minute = int(match["hour"]), int(match["minute"] or 0)
            if not (1 <= hour <= 12 and minute < 60):
                continue
            hour = hour % 12 + (12 if match["meridian"].lower() == "pm" else 0)
            return hour, minute, match.span()
        hour, minute = int(match["h24"]), int(match["m24"])
        if hour < 24 and minute < 60:
            return hour, minute, match.span()
    return None


def _extract_new_time(text_for_time, base_date):
    """Find a new time inside a sentence that has other words around it
    (e.g. "move it to 4pm" or "move it to 4.30").

    The clock time comes from _parse_clock_time. The DAY is base_date's day
    unless the message names another one ("tomorrow", "friday", "the 25th"),
    in which case dateparser resolves just that day word -- with the clock
    time cut out first, so it can't misread "11am" as a month."""
    text_for_time = _normalize_meridian(text_for_time)
    text_for_time = _normalize_bare_times(text_for_time)
    clock = _parse_clock_time(text_for_time)
    print(f"  [clock time found: {clock}]")
    if not clock:
        return None
    hour, minute, (start, end) = clock

    day = base_date.date()
    if _DAY_WORDS.search(text_for_time):
        without_clock = text_for_time[:start] + " " + text_for_time[end:]
        found = search_dates(
            without_clock,
            # "tomorrow" means tomorrow from NOW, not from the event's day.
            settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": time_utils.now()},
        )
        print(f"  [search_dates found: {found}]")
        days = [dt for matched, dt in (found or []) if _DAY_WORDS.search(matched)]
        if days:
            day = days[-1].date()

    return datetime.datetime.combine(day, datetime.time(hour, minute))


# --- creating events: the day and times are worked out here, not by the model

# How long a new event lasts when neither an end time nor a length is given.
DEFAULT_EVENT_MINUTES = 30

_AMOUNT = (
    r"\d+(?:\.\d+)?|forty[\s-]?five|forty|fifteen|twenty|thirty|sixty|"
    r"an?|one|two|three|four|five|six|seven|eight|nine|ten"
)
_WORD_AMOUNTS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "sixty": 60,
}
_UNIT = r"(?:minutes?|mins?|hours?|hrs?)"
# "in 30 minutes" starts later; "for 30 minutes" / "a 5 minute break" is how long.
_STARTS_IN = re.compile(rf"\bin\s+(?P<n>{_AMOUNT})\s+(?P<unit>{_UNIT})\b", re.IGNORECASE)
_LASTS = re.compile(
    rf"\b(?P<n>{_AMOUNT})[\s-]+(?P<unit>{_UNIT})\b|\bhalf\s+an?\s+hour\b", re.IGNORECASE
)
_STARTS_NOW = re.compile(r"\b(?:right\s+)?now\b|\bimmediately\b|\basap\b", re.IGNORECASE)

# A clock time, with the word before it if there is one. A bare number
# ("at 3", "to 4") only counts as a time with such a word, or when it is the
# first half of a range ("from 3 to 4pm"), so "a 5 minute break" or "2
# hours" are never mistaken for 5 o'clock or 2 o'clock.
_TIME_TOKEN = re.compile(
    r"(?:(?P<prep>\b(?:at|from|to|until|till|around|by)\b|@|-|–)\s*)?"
    r"(?:\b(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<mer>am|pm)?\b|\b(?P<word>noon|midnight)\b)",
    re.IGNORECASE,
)
_RANGE_PREPS = {"to", "until", "till", "-", "–"}
_TIME_PREPS = {"at", "from", "around", "by", "@"} | _RANGE_PREPS


def _normalize_spoken_digits(text):
    """"at 715" / "from 715 to 8" / "715pm" -> "7:15". Unlike
    _normalize_bare_times this leaves other numbers alone, so the "101" in
    "physics 101" isn't turned into 1:01."""
    def clock(digits):
        if len(digits) == 3:
            return f"{digits[0]}:{digits[1:]}"
        return digits if 1900 <= int(digits) <= 2100 else f"{digits[:2]}:{digits[2:]}"

    text = re.sub(
        r"\b(at|from|to|until|till|around|by)\s+(\d{3,4})\b",
        lambda m: f"{m[1]} {clock(m[2])}", text,
    )
    return re.sub(r"\b(\d{3,4})(?=\s*[ap]m\b)", lambda m: clock(m[1]), text)


def _amount_minutes(amount, unit):
    key = re.sub(r"[\s-]+", " ", amount.lower())
    number = _WORD_AMOUNTS[key] if key in _WORD_AMOUNTS else float(key)
    return number * (60 if unit.lower().startswith("h") else 1)


def _spoken_times(text):
    """The clock times in `text`, in order, as (hour, minute, meridian,
    preposition). meridian is "am", "pm", "24" (13:00, 0:30 -- no doubt about
    it) or None (a bare "3": am or pm not said)."""
    matches = list(_TIME_TOKEN.finditer(text))
    found = []
    for index, match in enumerate(matches):
        prep = (match["prep"] or "").lower()
        if match["word"]:
            noon = match["word"].lower() == "noon"
            found.append((12, 0, "pm" if noon else "am", prep))
            continue
        hour, minute = int(match["h"]), int(match["m"] or 0)
        meridian = (match["mer"] or "").lower() or None
        if minute > 59 or hour > 23 or (meridian and not 1 <= hour <= 12):
            continue
        certain = bool(meridian or match["m"] or prep in _TIME_PREPS)
        if not certain and index + 1 < len(matches):
            # "3 to 4pm": the 3 is the start of a range.
            following = matches[index + 1]
            gap = text[match.end():following.start()].strip()
            certain = (following["prep"] or "").lower() in _RANGE_PREPS and not gap
        if not certain:
            continue
        if meridian is None and (hour > 12 or hour == 0):
            meridian = "24"
        found.append((hour, minute, meridian, prep))
    return found


def _minutes_of_day(hour, minute, meridian):
    return (hour % 12 + (12 if meridian == "pm" else 0)) * 60 + minute if meridian in ("am", "pm") \
        else hour * 60 + minute


def _clock_candidates(hour, minute, meridian):
    """Minutes-since-midnight this spoken time could mean."""
    if meridian is not None:
        return [_minutes_of_day(hour, minute, meridian)]
    return [(hour % 12) * 60 + minute, (hour % 12 + 12) * 60 + minute]


def _parse_new_event(text, ref):
    """Work out when a new event is, entirely in Python. Returns (start,
    end) as timezone-aware datetimes, or None if the message gives no time
    that can be pinned down (the caller then falls back to the model's
    reading of it).

    Understood: "next friday at 2.30 p.m.", "tomorrow from 3 to 4pm", "at
    715 to 8", "starting now", "in 20 minutes", and a length as "for an
    hour" / "a 5 minute break"; with no end and no length, an event lasts
    DEFAULT_EVENT_MINUTES. A bare "at 3" is the next 3 o'clock from now if
    it's today, otherwise 9-11 mean morning and the rest afternoon/evening.
    """
    lowered = text.lower()

    starts_in = None
    if match := _STARTS_IN.search(lowered):
        starts_in = datetime.timedelta(minutes=_amount_minutes(match["n"], match["unit"]))
        lowered = lowered[: match.start()] + " " + lowered[match.end():]

    length = None
    if match := _LASTS.search(lowered):
        minutes = 30 if not match["n"] else _amount_minutes(match["n"], match["unit"])
        length = datetime.timedelta(minutes=minutes)
        lowered = lowered[: match.start()] + " " + lowered[match.end():]

    # Durations are gone, so a number like "120" can't be misread as 1:20.
    cleaned = _normalize_spoken_digits(
        _normalize_meridian(time_utils.strip_date_expressions(lowered))
    )
    times = _spoken_times(cleaned)

    named = time_utils.parse_date_range(text, ref)
    day_named = bool(named and named.start.date() == named.end.date())
    day = named.start.date() if day_named else ref.date()

    def at(date, minutes):
        return datetime.datetime.combine(
            date, datetime.time(minutes // 60 % 24, minutes % 60)
        ).astimezone() + datetime.timedelta(days=minutes // 1440)

    if times:
        start_h, start_m, start_mer, _ = times[0]
        end_spec = times[1] if len(times) > 1 and times[1][3] in _RANGE_PREPS else None

        candidates = _clock_candidates(start_h, start_m, start_mer)
        if len(candidates) == 1:
            start_min = candidates[0]
        elif end_spec and end_spec[2] in ("am", "pm"):
            # "from 3 to 4pm": the start is on the same side of noon as the end.
            end_min = _minutes_of_day(*end_spec[:3])
            same_side = candidates[1] if end_spec[2] == "pm" else candidates[0]
            start_min = same_side if same_side < end_min else min(candidates)
        elif day == ref.date():
            now_min = ref.hour * 60 + ref.minute
            ahead = [c for c in candidates if c >= now_min]
            if ahead:
                start_min = ahead[0]
            else:
                # Both 3 AM and 3 PM have passed: it can only mean tomorrow.
                start_min, day = candidates[0], day + datetime.timedelta(days=1)
        else:
            start_min = candidates[0] if 9 <= start_h <= 11 else candidates[1]

        start = at(day, start_min)
        if end_spec:
            options = _clock_candidates(*end_spec[:3])
            later = [c for c in options if c > start_min]
            end = at(day, min(later)) if later else at(day, min(options) + 1440)
        else:
            end = start + (length or datetime.timedelta(minutes=DEFAULT_EVENT_MINUTES))
    elif starts_in is not None:
        start = (ref + starts_in).replace(second=0, microsecond=0)
        end = start + (length or datetime.timedelta(minutes=DEFAULT_EVENT_MINUTES))
    elif _STARTS_NOW.search(lowered):
        start = ref.replace(second=0, microsecond=0)
        end = start + (length or datetime.timedelta(minutes=DEFAULT_EVENT_MINUTES))
    else:
        return None

    # "at 9am" said at 11am, with no day named, means tomorrow -- unless
    # the event would still be going on.
    if not day_named and not starts_in and end <= ref:
        start += datetime.timedelta(days=1)
        end += datetime.timedelta(days=1)
    return start, end


# "add a break after chemistry" / "put lunch before my 2pm meeting": the time
# is relative to another event on the schedule.
_RELATIVE = re.compile(
    r"\b(?P<rel>after|before)\s+(?:my\s+|the\s+)?(?P<name>[a-z][a-z' -]*?)"
    r"(?=\s+(?:for|on|at|tomorrow|today|tonight|next|this|every|each|daily|"
    r"until|till|through|starting|from|and)\b|[.,?!]|$)",
    re.IGNORECASE,
)


def _anchor_terms(match):
    return set(re.findall(r"[a-z]+", match["name"].lower())) - _SEARCH_STOPWORDS - time_utils.RANGE_WORDS


def _events_matching(terms, first, last):
    """Timed events between two datetimes whose titles match the terms. If
    some contain every term as a whole word, only those count: "bus" means
    the Bus, not a "Business Quiz" that merely resembles it."""
    matches = [
        e for e in get_events(first, last)
        if e["start"] != "All day" and _matches_terms(e, terms)
    ]
    exact = [
        e for e in matches
        if terms <= set(re.findall(r"[a-z]+", e["summary"].lower()))
    ]
    return exact or matches


def _relative_times(match, anchor, text):
    """(start, end) of a new event placed right after / before `anchor`."""
    length = datetime.timedelta(minutes=DEFAULT_EVENT_MINUTES)
    if lasts := _LASTS.search(text):
        length = datetime.timedelta(
            minutes=30 if not lasts["n"] else _amount_minutes(lasts["n"], lasts["unit"])
        )
    if match["rel"].lower() == "after":
        start = datetime.datetime.fromisoformat(anchor["end_iso"])
        return start, start + length
    end = datetime.datetime.fromisoformat(anchor["start_iso"])
    return end - length, end


def _parse_relative_event(text, ref):
    """Time a new event relative to an existing one. Returns (start, end,
    span of the words used), a string to say if the named event can't be
    found, or None if the message has no "after/before <event>" in it.

    With no day named, the next matching event counts, however far off in
    the next two weeks: on a Sunday, "after my bus" means Monday's bus."""
    match = _RELATIVE.search(text)
    if not match:
        return None
    terms = _anchor_terms(match)
    if not terms:
        return None

    named = time_utils.parse_date_range(text, ref)
    if named and named.start.date() == named.end.date():
        day = named.start.date()
        first = ref if day == ref.date() else time_utils.start_of_day(day)
        last = time_utils.end_of_day(day)
        where = time_utils.day_phrase(day, ref.date())
    else:
        first = ref
        last = time_utils.end_of_day(ref.date() + datetime.timedelta(days=SEARCH_DAYS - 1))
        where = "in the next two weeks"

    anchors = _events_matching(terms, first, last)
    if not anchors:
        return f"I couldn't find {match['name'].strip()} on the calendar {where}."
    start, end = _relative_times(match, anchors[0], text)
    return start, end, match.span()


_TITLE_DROP = {
    "add", "schedule", "create", "put", "book", "set", "up", "make", "a", "an",
    "the", "please", "can", "you", "could", "would", "for", "at", "on", "from",
    "to", "until", "till", "starting", "start", "now", "next", "this", "in",
    "tomorrow", "today", "tonight", "my", "calendar", "minute", "minutes",
    "hour", "hours", "am", "pm", "hey", "jarvis", "of", "and", "half", "one",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "fifteen", "twenty", "thirty", "forty", "sixty", "min", "mins", "hr", "hrs",
} | set(time_utils.WEEKDAYS)


def _fallback_title(text):
    """A title for when the model didn't give a usable one: what's left of
    the message after the command, date and time words are removed."""
    text = _normalize_meridian(time_utils.strip_date_expressions(text))
    words = re.findall(r"[a-z][a-z'-]*", text)
    kept = [w for w in words if w not in _TITLE_DROP]
    return " ".join(kept) if kept else "Event"


def _added_reply(event, today):
    day = event_day(event)
    where = "" if day == today else " " + time_utils.day_phrase(day, today)
    return f"Added {event['summary']}{where} from {event['start']} to {event['end']}."


# --- repeating events: "every day at 6.30 pm", "after school every weekday"

# With no end given ("until friday", "for 3 weeks") a repeat covers this many
# days: an open-ended series would clutter the calendar, and the reply says
# where it stops so it's easy to ask for more. MAX_REPEAT_DAYS is a hard cap.
DEFAULT_REPEAT_DAYS = 14
MAX_REPEAT_DAYS = 90

_DAY_NAMES = "|".join(time_utils.WEEKDAYS)
_EVERY_DAY = re.compile(
    r"\b(?:every\s+(?:single\s+)?(?:day|night|morning|afternoon|evening)|everyday|"
    r"each\s+(?:day|night|morning|evening)|daily)\b",
    re.IGNORECASE,
)
_EVERY_WEEKDAY = re.compile(
    r"\b(?:every\s+weekday|(?:on\s+)?weekdays|(?:monday|mon)\s+(?:to|through|thru)\s+(?:friday|fri))\b",
    re.IGNORECASE,
)
_EVERY_WEEKEND = re.compile(r"\b(?:every\s+weekend|(?:on\s+)?weekends)\b", re.IGNORECASE)
# "every tuesday and thursday", "every monday, wednesday and friday"
_EVERY_NAMED = re.compile(
    rf"\b(?:every|each)\s+(?P<days>(?:(?:{_DAY_NAMES})s?(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s+)?)+)",
    re.IGNORECASE,
)
# "tuesdays and thursdays"
_PLURAL_NAMED = re.compile(
    rf"\b(?P<days>(?:(?:{_DAY_NAMES})s(?:\s*,\s*(?:and\s+)?|\s+and\s+)?)+)", re.IGNORECASE
)
_WEEKLY = re.compile(r"\b(?:weekly|every\s+week|once\s+a\s+week)\b", re.IGNORECASE)
# How long it goes on: "for 3 weeks", "until friday", "through the 30th".
_FOR_SPAN = re.compile(
    r"\bfor\s+(?:the\s+)?(?:next\s+)?(?P<n>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s+(?P<unit>days?|weeks?|months?)\b",
    re.IGNORECASE,
)
_UNTIL = re.compile(
    r"\b(?:until|till|through|thru)\s+(?P<when>(?:my\s+|the\s+|next\s+|this\s+)?[a-z0-9/' ]+?)"
    r"(?=\s+(?:at|from|for|every|after|before)\b|[,.?!]|$)",
    re.IGNORECASE,
)

Repeat = namedtuple("Repeat", "weekdays first last text")


def _cut(text, match):
    return text[: match.start()] + " " + text[match.end():]


def _parse_repeat(text, ref):
    """Does the message ask for something on a repeating basis? Returns a
    Repeat -- the weekdays it falls on (0=Monday), the first and last date
    it covers, and the message with the repeat words cut out -- or None.

    The dates are all worked out here: "every day for 2 weeks", "every
    tuesday and thursday until friday", "on weekdays next week"."""
    weekly = False
    if match := _EVERY_DAY.search(text):
        weekdays = set(range(7))
    elif match := _EVERY_WEEKDAY.search(text):
        weekdays = {0, 1, 2, 3, 4}
    elif match := _EVERY_WEEKEND.search(text):
        weekdays = {5, 6}
    elif (match := _EVERY_NAMED.search(text)) or (match := _PLURAL_NAMED.search(text)):
        weekdays = {
            time_utils.WEEKDAYS.index(name)
            for name in re.findall(_DAY_NAMES, match["days"].lower())
        }
    elif match := _WEEKLY.search(text):
        weekdays, weekly = set(), True
    else:
        return None
    text = _cut(text, match)

    span_days = None
    if match := _FOR_SPAN.search(text):
        n = int(match["n"]) if match["n"].isdigit() else _WORD_AMOUNTS[match["n"].lower()]
        span_days = int(n * {"d": 1, "w": 7, "m": 30}[match["unit"][0].lower()])
        text = _cut(text, match)

    until = None
    if match := _UNTIL.search(text):
        found = time_utils.parse_date_range(match["when"], ref)
        if found and found.start.date() == found.end.date():
            until = found.end.date()
            text = _cut(text, match)

    today = ref.date()
    first, last = today, None
    if named := time_utils.parse_date_range(text, ref):
        first = max(named.start.date(), today)
        if named.start.date() != named.end.date():
            last = named.end.date()  # "every day next week"
    if until and until >= first:
        last = until
    elif span_days:
        last = first + datetime.timedelta(days=span_days - 1)
    if last is None or last < first:
        last = first + datetime.timedelta(days=DEFAULT_REPEAT_DAYS - 1)
    last = min(last, first + datetime.timedelta(days=MAX_REPEAT_DAYS - 1))
    if weekly:
        weekdays = {first.weekday()}
    return Repeat(weekdays, first, last, text)


def _weekday_label(weekdays):
    if len(weekdays) == 7:
        return "every day"
    if weekdays == {0, 1, 2, 3, 4}:
        return "on weekdays"
    if weekdays == {5, 6}:
        return "on weekends"
    return "every " + _join_words([time_utils.WEEKDAYS[i].capitalize() for i in sorted(weekdays)])


def _create_repeating(repeat, ref, context):
    """Work out every occurrence, then ask before creating them: a series is
    a lot of events to add on the strength of one transcript, and a
    misheard word would otherwise mean cleaning them all up by hand.

    Either every day at the same time ("every day at 6.30 pm"), or tied to
    another event whose time differs from day to day ("after school every
    day"), in which case only days that have that event get one."""
    text = repeat.text
    dates = [
        repeat.first + datetime.timedelta(days=i)
        for i in range((repeat.last - repeat.first).days + 1)
        if (repeat.first + datetime.timedelta(days=i)).weekday() in repeat.weekdays
    ]

    anchor_name = None
    occurrences = []
    relative = _RELATIVE.search(text)
    terms = _anchor_terms(relative) if relative else None
    if terms:
        first = ref if repeat.first == ref.date() else time_utils.start_of_day(repeat.first)
        by_day = {}
        for event in _events_matching(terms, first, time_utils.end_of_day(repeat.last)):
            by_day.setdefault(event_day(event), event)
        occurrences = [_relative_times(relative, by_day[d], text) for d in dates if d in by_day]
        if not occurrences:
            return (
                f"I couldn't find {relative['name'].strip()} on your calendar "
                f"between {repeat.first:%B} {repeat.first.day} and "
                f"{repeat.last:%B} {repeat.last.day}."
            ), context
        anchor_name = next(iter(by_day.values()))["summary"]
        text = _cut(text, relative)
    else:
        parsed = _parse_new_event(text, ref)
        if not parsed:
            return (
                "What time should they be? Try something like 'every day at "
                "6.30 pm', or 'every day after school'."
            ), context
        start, end = parsed
        occurrences = [
            (datetime.datetime.combine(d, start.time()).astimezone(),
             datetime.datetime.combine(d, start.time()).astimezone() + (end - start))
            for d in dates
        ]

    # Today's occurrence only counts if it hasn't already finished.
    occurrences = sorted(o for o in occurrences if o[1] > ref)
    if not occurrences:
        return "Every one of those would already be over, so I didn't add anything.", context

    title = extract_event_title(text) or _fallback_title(text)
    title = title[:1].upper() + title[1:]
    if anchor_name:
        label = _weekday_label({start.weekday() for start, _ in occurrences})
        what = f"right after {anchor_name}" if relative["rel"].lower() == "after" else f"right before {anchor_name}"
    else:
        label = _weekday_label(repeat.weekdays)
        what = f"at {occurrences[0][0]:%I:%M %p}"
    last_day = occurrences[-1][0].date()
    spoken = f"{title} {what} {label}, through {last_day:%B} {last_day.day}"
    print(f"  [repeat: {title!r}, {len(occurrences)} events, {occurrences[0][0]:%a %b %d} to {last_day:%a %b %d}]")

    context["confirm_pending"] = {
        "intent": "CREATE_MANY",
        "title": title,
        "spoken": spoken,
        "occurrences": [(s.isoformat(), e.isoformat()) for s, e in occurrences],
    }
    return f"Add {spoken}? That's {len(occurrences)} events.", context


def _do_create_many(pending, context):
    items = [(pending["title"], s, e) for s, e in pending["occurrences"]]
    created, error = insert_events(items)
    if created:
        context["last_event"] = created[-1]
        context["last_batch"] = created  # so "make them green" means these
    if error:
        traceback.print_exception(error)
        return (
            f"I added {len(created)} of {len(items)} {pending['title']} events, "
            "then something went wrong with the calendar."
        ), context
    return f"Added {pending['spoken']}: {len(items)} events.", context


def _when(event, today):
    """How to say when an event is: "at 03:00 PM" for today, otherwise with
    the day first ("tomorrow at 03:00 PM", "on Friday at 01:00 PM")."""
    day = event_day(event)
    prefix = "" if day == today else time_utils.day_phrase(day, today) + " "
    return prefix + ("all day" if event["start"] == "All day" else f"at {event['start']}")


def _do_delete(target, context):
    today = time_utils.now().date()
    delete_event(target["id"])
    context["last_event"] = None
    context["pending"] = None
    return f"Deleted {target['summary']} {_when(target, today)}.", context


def _do_time(user_text):
    """The current time and/or date, straight from the system clock."""
    ref = time_utils.now()
    wants_date = bool(_WANTS_DATE.search(user_text))
    wants_time = bool(_WANTS_TIME.search(user_text))
    if wants_date and wants_time:
        return f"{time_utils.time_reply(ref)} {time_utils.date_reply(ref)}"
    if wants_date:
        return time_utils.date_reply(ref)
    return time_utils.time_reply(ref)


# Words in a READ request that say "what's on" rather than WHAT to look for.
# Whatever is left ("chemistry test") is a search: only matching events are
# reported, and with no day named the search covers the next two weeks.
_SEARCH_STOPWORDS = {
    "what", "whats", "s", "do", "does", "did", "i", "have", "has", "had", "on",
    "my", "the", "a", "an", "any", "anything", "something", "calendar",
    "schedule", "schedules", "agenda", "event", "events", "planned",
    "scheduled", "is", "are", "was", "there", "up", "tell", "me", "show",
    "list", "read", "out", "for", "in", "at", "to", "of", "and", "or", "look",
    "looks", "like", "going", "get", "got", "how", "when", "am", "be", "busy",
    "free", "plans", "happening", "can", "could", "you", "please", "give",
    "check", "see", "hey", "jarvis", "later", "left", "rest", "else", "other",
    "all", "everything", "much", "full", "packed", "now", "still", "just",
    "so", "okay", "ok", "um", "uh", "it", "that", "this", "we", "will", "would",
    "about", "appointment", "appointments", "time", "start", "starts",
    "begin", "begins", "end", "ends", "long", "yes", "yeah", "yep", "sure",
    "thanks", "thank", "right", "well", "actually", "then", "next", "if",
    "whether", "want", "need", "know", "let", "also", "too", "again",
    "summary", "rundown", "overview", "recap", "briefing",
}
# Different words for the same kind of event.
_SYNONYMS = [
    {"test", "tests", "exam", "exams", "quiz", "quizzes", "midterm", "midterms",
     "final", "finals"},
    {"homework", "assignment", "assignments", "hw"},
]


def _search_terms(text):
    words = set(re.findall(r"[a-z]+", time_utils.strip_date_expressions(text)))
    return words - _SEARCH_STOPWORDS - time_utils.RANGE_WORDS


def _term_in_title(term, title_words):
    for word in title_words:
        if word == term:
            return True
        # "chem" ~ "chemistry", "tests" ~ "test". Both at least 4 letters, so
        # "bus" is not "business".
        if min(len(term), len(word)) >= 4 and (word.startswith(term) or term.startswith(word)):
            return True
    group = next((g for g in _SYNONYMS if term in g), None)
    if group and title_words & group:
        return True
    return bool(difflib.get_close_matches(term, title_words, n=1, cutoff=0.85))


def _matches_terms(event, terms):
    """Every search term has to be in the title, so "chemistry test" finds
    the Chemistry Test but not a Chemistry Study Block."""
    title_words = set(re.findall(r"[a-z]+", event["summary"].lower()))
    return all(_term_in_title(term, title_words) for term in terms)


def _event_line(event):
    if event["start"] == "All day":
        return f"{event['summary']} all day"
    return f"{event['summary']} at {event['start']}"


# An event title that shows up on this many different days is a routine
# ("School", "Bus", "Code Ninjas"), and is described as a pattern instead of
# being listed day by day.
MIN_REPEATS = 3


def _join_words(items):
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _days_phrase(dates):
    """Sorted distinct dates -> "weekdays", "the weekend", "Tuesday and
    Thursday", "Monday through Friday", "every day"."""
    weekdays = {d.weekday() for d in dates}
    if len(dates) == 7:
        return "every day"
    if len(dates) == 5 and weekdays == {0, 1, 2, 3, 4}:
        return "weekdays"
    if len(dates) == 2 and weekdays == {5, 6}:
        return "the weekend"
    runs = []
    for d in dates:
        if runs and (d - runs[-1][-1]).days == 1:
            runs[-1].append(d)
        else:
            runs.append([d])
    return _join_words([
        f"{run[0]:%A} through {run[-1]:%A}" if len(run) >= 3
        else _join_words([f"{d:%A}" for d in run])
        for run in runs
    ])


def _routine_sentence(title, occurrences):
    """One repeating event as a pattern: "School on weekdays from 08:20 AM
    to 02:42 PM", or with different times on different days, "Code Ninjas on
    Tuesday and Thursday from 04:40 PM to 08:00 PM, and on Saturday from
    10:40 AM to 03:30 PM"."""
    by_hours = {}
    for event in occurrences:
        hours = "all day" if event["start"] == "All day" else f"from {event['start']} to {event['end']}"
        by_hours.setdefault(hours, []).append(event_day(event))
    parts = [
        f"on {_days_phrase(sorted(set(dates)))} {hours}"
        for hours, dates in by_hours.items()
    ]
    return f"{title} " + ", and ".join(parts)


def _split_routines(events):
    """Separate events into (routines, everything else). A routine is a
    title seen on MIN_REPEATS+ different days. Only tried for a span of a
    week or less, where a weekday name means one specific day."""
    days = [event_day(e) for e in events]
    if not days or (max(days) - min(days)).days >= 7:
        return [], events

    by_title = {}
    for event in events:
        by_title.setdefault(event["summary"].strip().lower(), []).append(event)
    routine_titles = {
        title for title, group in by_title.items()
        if len({event_day(e) for e in group}) >= MIN_REPEATS
    }
    routines = [
        (group[0]["summary"], group)
        for title, group in by_title.items() if title in routine_titles
    ]
    others = [e for e in events if e["summary"].strip().lower() not in routine_titles]
    return routines, others


def _speak_events(events, today):
    """One sentence per day -- "Today: A at 3 PM; B at 5 PM. Friday: C at 1
    PM." -- capped so a busy week doesn't become a monologue. Events that
    repeat on 3+ days are pulled out first and described as routines ("School
    on weekdays from 08:20 AM to 02:42 PM"), which is what makes a week
    readable."""
    routines, events = _split_routines(events)
    if routines:
        sentence = "You have " + "; ".join(_routine_sentence(t, g) for t, g in routines) + "."
        if not events:
            return sentence
        return f"{sentence} Other than that: {_speak_events(events, today)}"

    shown = events[:MAX_SPOKEN_EVENTS]
    by_day = {}
    for event in shown:
        by_day.setdefault(event_day(event), []).append(event)
    text = " ".join(
        f"{time_utils.day_heading(day, today)}: "
        + "; ".join(_event_line(e) for e in day_events) + "."
        for day, day_events in by_day.items()
    )
    rest = events[len(shown):]
    if rest:
        # Say which days the cut-off events are on, so a busy week doesn't
        # just silently lose its later days.
        names = [
            time_utils.day_heading(d, today)
            for d in dict.fromkeys(event_day(e) for e in rest)
        ]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        text += f" And {len(rest)} more, on {joined}."
    return text


def _do_read(user_text, context):
    """What's coming up. Only events that haven't finished yet are
    reported. The day(s) come from the message ("tomorrow", "friday", "next
    week"); with none, it's the rest of today -- or, if the message is a
    search ("do I have a chemistry test?"), the next two weeks."""
    ref = time_utils.now()
    today = ref.date()
    date_range = time_utils.parse_date_range(user_text, ref)
    terms = _search_terms(user_text) if _SPECIFIC_SEARCH.search(user_text) else set()

    if date_range is None:
        if terms:
            last_day = today + datetime.timedelta(days=SEARCH_DAYS - 1)
            date_range = time_utils.DateRange(
                ref, time_utils.end_of_day(last_day), "in the next two weeks"
            )
        else:
            date_range = time_utils.DateRange(ref, time_utils.end_of_day(today), "today")
    print(f"  [read: {date_range.label}, search terms: {sorted(terms)}]")

    events = get_events(date_range.start, date_range.end)
    if terms:
        events = [e for e in events if _matches_terms(e, terms)]

    if not events:
        if terms:
            return f"I don't see anything like that {date_range.label}.", context
        if date_range.label == "today":
            return "You've got nothing else scheduled today.", context
        return f"You've got nothing scheduled {date_range.label}.", context

    context["last_event"] = events[0]
    yes = "Yes. " if terms and _YES_NO_QUESTION.search(user_text) else ""
    return yes + _speak_events(events, today), context


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


# "make them green", "delete those" -- the events just added together.
_BATCH_REF = re.compile(r"\b(?:those|these|them)\b", re.IGNORECASE)
# "all of them", "each of those", "every chemistry study block": every match,
# not one of them.
_WANTS_ALL = re.compile(r"\b(?:all|every|each|both|everything)\b|\b(?:those|these|them)\b", re.IGNORECASE)


def _titles_phrase(targets):
    """"all 5 Chemistry study block events", or with mixed titles "all 6
    events (5 Chemistry study block, 1 Chemistry test)"."""
    counts = {}
    for event in targets:
        counts[event["summary"]] = counts.get(event["summary"], 0) + 1
    if len(counts) == 1:
        return f"all {len(targets)} {next(iter(counts))} events"
    breakdown = ", ".join(f"{n} {title}" for title, n in counts.items())
    return f"all {len(targets)} events ({breakdown})"


def _apply_to_all(intent, targets, text, context, note=""):
    """COLOR or DELETE across several events at once ("each of those").
    Recoloring just happens (it's easy to undo); deleting this many events
    asks first."""
    what = _titles_phrase(targets)
    ids = [event["id"] for event in targets]
    context["pending"] = None

    if intent == "DELETE":
        context["confirm_pending"] = {"intent": "DELETE_MANY", "targets": targets, "spoken": what}
        return f"Delete {what}{note}?", context

    color = _extract_color(text)
    if color is None:
        return f"I found {what}, but which color? Try saying something like 'make them red'.", context
    color_id, label = color
    done, error = set_event_colors(ids, color_id)
    if error:
        traceback.print_exception(error)
        return (
            f"I changed {done} of {len(ids)} events before something went "
            "wrong with the calendar."
        ), context
    if color_id is None:
        return f"Reset {what} to the default color.", context
    return f"Made {what} {label}.", context


def _do_delete_many(pending, context):
    targets = pending["targets"]
    done, error = delete_events([event["id"] for event in targets])
    context["last_event"] = None
    context["last_batch"] = None
    if error:
        traceback.print_exception(error)
        return (
            f"I deleted {done} of {len(targets)} events before something went "
            "wrong with the calendar."
        ), context
    return f"Deleted {pending['spoken']}.", context


def _apply_to_target(intent, target, text, context):
    """Carry out an intent on one resolved event. Destructive or
    time-changing actions ask for confirmation first; recoloring is
    trivially reversible, so it just happens."""
    if intent == "COLOR":
        return _do_color(target, text, context)
    return _prepare_confirmation(intent, target, text, context)


def _prepare_confirmation(intent, target, text_for_time, context):
    today = time_utils.now().date()
    if intent == "DELETE":
        context["confirm_pending"] = {
            "intent": intent,
            "target": target,
        }
        return f"Delete {target['summary']} {_when(target, today)}?", context

    if target["start"] == "All day":
        return (
            f"{target['summary']} is an all-day event, so there's no time to "
            "move it to."
        ), context

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

    # "Move Chemistry on Friday to 04:30 PM tomorrow?" -- say the day when
    # it isn't today, and again if the move changes it.
    target_day = event_day(target)
    where = "" if target_day == today else " " + time_utils.day_phrase(target_day, today)
    new_day = new_start.date()
    new_where = "" if new_day == target_day else " " + time_utils.day_phrase(new_day, today)
    spoken = f"{target['summary']}{where} to {new_start.strftime('%I:%M %p')}{new_where}"

    context["confirm_pending"] = {
        "intent": intent,
        "target": target,
        "new_start_iso": _local_iso(new_start),
        "new_end_iso": _local_iso(new_end),
        "spoken": spoken,
    }
    return f"Move {spoken}?", context


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
            if confirm_pending["intent"] == "CREATE_MANY":
                return _do_create_many(confirm_pending, context)
            if confirm_pending["intent"] == "DELETE_MANY":
                return _do_delete_many(confirm_pending, context)
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
            return f"Moved {confirm_pending['spoken']}.", context
        if _is_negative(user_text):
            return "Okay, I won't make that change.", context
        # Neither yes nor no: the user moved on to something else. Drop
        # the change and handle this message as a fresh request instead
        # of swallowing it.

    pending = context.get("pending")
    if pending:
        candidates = pending["candidates"]
        named = _named_days(user_text)  # "the friday one", "tomorrow's"
        if named:
            candidates = _scope_to_days(candidates, named)
        # "all of them" / "each": the answer to "which one?" is every one.
        if pending["intent"] in ("COLOR", "DELETE") and _WANTS_ALL.search(user_text):
            return _apply_to_all(pending["intent"], candidates, pending["original_text"], context)
        matches = _match_events(user_text, candidates)
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

    if intent == "TIME":
        return _do_time(user_text), context

    if intent == "READ":
        if skip_read:
            return None, context
        return _do_read(user_text, context)

    if intent == "CREATE":
        ref = time_utils.now()
        if repeat := _parse_repeat(user_text, ref):
            return _create_repeating(repeat, ref, context)

        title_text = user_text
        parsed = _parse_new_event(user_text, ref)
        if not parsed:
            relative = _parse_relative_event(user_text, ref)
            if isinstance(relative, str):
                return relative, context
            if relative:
                start, end, (cut_from, cut_to) = relative
                parsed = start, end
                # The title is "Break", not "Break after chemistry".
                title_text = user_text[:cut_from] + user_text[cut_to:]
        if parsed:
            start, end = parsed
            title = extract_event_title(title_text) or _fallback_title(title_text)
            title = title[:1].upper() + title[1:]
            print(f"  [create: {title!r}, {start:%a %b %d %I:%M %p} to {end:%I:%M %p}]")
            new_event = insert_event(title, start.isoformat(), end.isoformat())
            context["last_event"] = new_event
            context["last_batch"] = None  # "them" no longer means the earlier batch
            return _added_reply(new_event, ref.date()), context

        return (
            "I couldn't tell when that should be -- try something like "
            "'add a break at 3pm', or 'add a break after chemistry'."
        ), context

    # UPDATE, DELETE and COLOR all need a specific real event, resolved by ID.
    # Today plus the next few days are candidates ("delete my study block
    # tomorrow").
    today = time_utils.now().date()
    events = get_events(
        time_utils.start_of_day(today),
        time_utils.end_of_day(today + datetime.timedelta(days=TARGET_DAYS - 1)),
    )
    status, result = _resolve_target(user_text, events, context)

    if status == "NONE":
        return (
            "I couldn't find an event matching that -- "
            "can you tell me the title or time?"
        ), context

    if status == "MANY" and intent in ("COLOR", "DELETE") and _WANTS_ALL.search(user_text):
        # Only the next week was searched -- say so before deleting, unless
        # the events came from a named day or the batch just created.
        searched_week = not _named_days(user_text) and result is not context.get("last_batch")
        return _apply_to_all(
            intent, result, user_text, context,
            note=" over the next 7 days" if searched_week else "",
        )

    if status == "MANY":
        options = ", ".join(f"{e['summary']} {_when(e, today)}" for e in result)
        context["pending"] = {
            "intent": intent,
            "candidates": result,
            "original_text": user_text,
        }
        return f"I found a few things that could match: {options}. Which one did you mean?", context

    return _apply_to_target(intent, result, user_text, context)  # status == "ONE"
