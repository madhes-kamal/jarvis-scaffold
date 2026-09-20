"""
brief_generator.py

Builds the morning briefing entirely in Python from templates, rather than
asking an LLM to write it. It is given only the events that haven't
finished yet (calendar_service.get_upcoming_events), so a brief asked for at
noon doesn't recite the morning.

Why: at low temperature (needed for accuracy), a small local model barely
varies its own phrasing no matter what style hint you give it -- and
raising the temperature to get variety reopens the door to hallucinating
times/events. Templating sidesteps the trade-off entirely: the facts are
inserted directly from your calendar data (impossible to get wrong,
since there's no LLM in the path to misremember them), and variety comes
from randomly combining openers/transitions/closers, which gives far
more real variation than an LLM at temperature 0.1 ever produced anyway.
"""

import random

OPENERS = [
    "Good morning, sir.",
    "Morning, sir. Hope you slept well.",
    "Rise and shine, sir.",
    "Sir, good morning.",
    "Morning, sir.",
]

TRANSITIONS = [
    "Then",
    "After that",
    "Following that",
    "Next up",
    "Once that wraps up",
    "From there",
    "Moving on",
]

CLOSERS = [
    "That's the full picture for today, sir.",
    "That covers everything on the books.",
    "That's your day, sir.",
    "That's the rundown.",
    "That's everything on the schedule.",
]

# The brief only covers what hasn't finished yet, so these have to be true
# whether the day was empty or everything is already behind you.
NO_EVENTS_LINES = [
    "There's nothing left on the calendar today, sir -- a clean slate.",
    "Good news, sir: your calendar is clear for the rest of today.",
    "Nothing else scheduled today, sir.",
]


def _format_event_list(events):
    """Builds the factual portion of the brief. Every word describing an
    event/time comes directly from `events` -- nothing here is generated
    or guessed."""
    parts = []
    for i, e in enumerate(events):
        if e["start"] == "All day":
            time_str = "(all day)"
        else:
            time_str = f"from {e['start']} to {e['end']}"

        if i == 0:
            sentence = f"First up is {e['summary']} {time_str}"
        else:
            connector = random.choice(TRANSITIONS)
            sentence = f"{connector}, there's {e['summary']} {time_str}"

        parts.append(sentence.strip())

    return ". ".join(parts) + "."


HEADS_UP_LEADS = [
    "Heads up, sir:",
    "One thing to keep in mind, sir:",
    "Looking ahead, sir:",
]


def _prep_clause(item):
    """How to say whether prep time is set aside. `prep_count` is None when
    the event has no recognisable subject (nothing to look for), otherwise
    how many study/prep events already sit before it."""
    count, soon = item["prep_count"], item["soon"]
    if count:
        plural = "block" if count == 1 else "blocks"
        return f", with {count} {item['subject']} study {plural} lined up before it"
    if count == 0:
        if soon:
            return ", and nothing is scheduled to prep for it"
        return ", and nothing is scheduled to prep for it yet, so make sure to prep ahead of that"
    return ", so make sure you're ready" if soon else ", so make sure to prep ahead of that"


def _format_heads_up(items, more=0):
    """Builds the look-ahead sentences. As with the event list, the facts
    (titles, days, counts) are inserted straight from calendar data."""
    if not items:
        return ""
    sentences = []
    for i, item in enumerate(items):
        lead = random.choice(HEADS_UP_LEADS) + " you have" if i == 0 else "You also have"
        sentences.append(f"{lead} {item['title']} {item['when']}{_prep_clause(item)}.")
    if more:
        sentences.append(f"There {'is' if more == 1 else 'are'} {more} more after that.")
    return " ".join(sentences)


def generate_brief(events, heads_up=None, offer=None, more=0):
    """events: what's left of today. heads_up: dicts describing tests,
    quizzes and deadlines coming up (title, when, soon, subject,
    prep_count). offer: a question to end on, like "Want me to add study
    blocks...?", which replaces the closing line."""
    opener = random.choice(OPENERS)
    heads_up_text = _format_heads_up(heads_up, more)

    if not events:
        parts = [opener, random.choice(NO_EVENTS_LINES)]
    else:
        parts = [opener, _format_event_list(events)]
    if heads_up_text:
        parts.append(heads_up_text)
    if offer:
        parts.append(offer)
    elif events:
        parts.append(random.choice(CLOSERS))
    return " ".join(parts)


if __name__ == "__main__":
    # Run this a few times -- the facts stay identical, the phrasing shifts.
    sample_events = [
        {"summary": "SAMPLE TASK", "start": "05:00 PM", "end": "05:45 PM"},
        {"summary": "SAMPLE TASK 2", "start": "05:45 PM", "end": "06:30 PM"},
    ]

    for i in range(5):
        print(f"--- Run {i + 1} ---")
        print(generate_brief(sample_events))
        print()
