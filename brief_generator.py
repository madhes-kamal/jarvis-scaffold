"""
brief_generator.py

Builds the morning briefing entirely in Python from templates, rather than
asking an LLM to write it.

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

NO_EVENTS_LINES = [
    "There's nothing on the calendar today, sir -- a rare clean slate.",
    "Good news, sir: your calendar is completely clear today.",
    "Nothing scheduled today, sir.",
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
            sentence = f"Today starts with {e['summary']} {time_str}"
        else:
            connector = random.choice(TRANSITIONS)
            sentence = f"{connector}, there's {e['summary']} {time_str}"

        parts.append(sentence.strip())

    return ". ".join(parts) + "."


def generate_brief(events):
    if not events:
        return f"{random.choice(OPENERS)} {random.choice(NO_EVENTS_LINES)}"

    opener = random.choice(OPENERS)
    closer = random.choice(CLOSERS)
    body = _format_event_list(events)

    return f"{opener} {body} {closer}"


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
