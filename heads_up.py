"""
heads_up.py

The look-ahead half of "good morning": tests, quizzes and deadlines coming up
in the next two weeks, whether there's prep time set aside for them, and --
when there isn't -- a concrete offer of study blocks.

Everything here is worked out in Python from calendar data. The model is not
involved, so it can't invent a test, miscount the study blocks, or put one on
top of something you're already doing.

- find_heads_up()   what's coming up, and how much prep is already scheduled
- plan_study_blocks()   one block per day before the test, at a chosen time;
                    a block that would land on another event moves to just
                    after it, or is skipped if that leaves no room
- describe_plan()   the spoken question for a plan
- build_morning_brief()   the whole "good morning" text plus the offer
"""

import datetime
import re
from collections import namedtuple

import time_utils
from brief_generator import generate_brief
from calendar_service import event_day, get_upcoming_events

# How far ahead "good morning" looks, and how many items it will read out.
HEADS_UP_DAYS = 14
MAX_HEADS_UP = 3

# Ask about this time first when offering study blocks (minutes after
# midnight). It changes per subject, so it is always confirmed, never assumed.
DEFAULT_STUDY_MINUTES = 18 * 60 + 30
STUDY_BLOCK_MINUTES = 45
# For a test far off, only the last few days before it get blocks; more than
# this is a lot to add on the strength of one answer.
MAX_STUDY_BLOCKS = 7
# A block that would have to end after this is skipped instead of moved.
LATEST_STUDY_END_MINUTES = 22 * 60

# Titles that make an event worth a heads-up. Whole words only ("Final
# Fantasy night" is the price of "final"; "attest" doesn't count).
TRIGGER_WORDS = {
    "test", "tests", "exam", "exams", "quiz", "quizzes", "midterm", "midterms",
    "final", "finals", "due", "deadline", "deadlines", "presentation",
    "presentations",
}
# An event counts as prep for a test if its title has one of these.
STUDY_WORDS = {
    "study", "studying", "prep", "review", "revision", "revise", "practice",
    "cram", "homework",
}
# Words in a test's title that aren't its subject ("Chemistry Unit 3 Test").
_NOT_SUBJECT = TRIGGER_WORDS | {
    "the", "a", "an", "of", "for", "and", "unit", "chapter", "section", "part",
    "module", "lesson", "week", "my", "on", "in",
}
# How many different days a title has to show up on to be a routine (a weekly
# quiz, say) rather than a one-off worth warning about.
_ROUTINE_DAYS = 3

HeadsUp = namedtuple("HeadsUp", "event days_away subject prep_count")
StudyPlan = namedtuple("StudyPlan", "occurrences shifted skipped")


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def subject_of(title):
    """"Chemistry Unit 3 Test" -> "chemistry"; "Test" -> "" (no subject)."""
    return " ".join([w for w in _words(title) if w not in _NOT_SUBJECT][:2])


def _start(event):
    """When an event starts, as an aware datetime (midnight for all-day)."""
    if event["start"] == "All day":
        return time_utils.start_of_day(event_day(event))
    return datetime.datetime.fromisoformat(event["start_iso"])


def _end(event):
    if event["start"] == "All day":
        return time_utils.start_of_day(event_day(event) + datetime.timedelta(days=1))
    return datetime.datetime.fromisoformat(event["end_iso"])


def _is_prep_for(event, subject_words):
    words = set(_words(event["summary"]))
    if not words & STUDY_WORDS:
        return False
    return all(
        any(w == s or (min(len(w), len(s)) >= 4 and (w.startswith(s) or s.startswith(w))) for w in words)
        for s in subject_words
    )


def find_heads_up(events, ref):
    """Tests, quizzes and deadlines from tomorrow on, nearest first.

    `events` are upcoming calendar events (calendar_service format). One-offs
    only: a title on 3+ different days is a routine and is skipped, and so is
    anything today (the day's own list already reads it out). prep_count is
    the number of study/prep events for the same subject that finish before
    it starts, or None when the title has no subject to look for."""
    today = ref.date()
    days_seen = {}
    for event in events:
        days_seen.setdefault(event["summary"].strip().lower(), set()).add(event_day(event))

    items = []
    for event in events:
        away = (event_day(event) - today).days
        if away < 1 or not set(_words(event["summary"])) & TRIGGER_WORDS:
            continue
        if len(days_seen[event["summary"].strip().lower()]) >= _ROUTINE_DAYS:
            continue

        subject = subject_of(event["summary"])
        prep = None
        if subject:
            subject_words = subject.split()
            prep = sum(
                1 for other in events
                if other is not event and _is_prep_for(other, subject_words)
                and _end(other) <= _start(event)
            )
        items.append(HeadsUp(event, away, subject, prep))
    return sorted(items, key=lambda item: (_start(item.event), item.event["summary"]))


def days_before(event, ref):
    """Every day from today up to the day before `event`."""
    return [
        ref.date() + datetime.timedelta(days=n)
        for n in range((event_day(event) - ref.date()).days)
    ]


def plan_study_blocks(test_event, events, ref, start_minutes, length_minutes=STUDY_BLOCK_MINUTES):
    """Study blocks on each day from today to the day before a test (see
    plan_study_days)."""
    days = days_before(test_event, ref)
    others = [e for e in events if e is not test_event]
    return plan_study_days(days, others, ref, start_minutes, length_minutes)


def plan_study_days(days, events, ref, start_minutes, length_minutes=STUDY_BLOCK_MINUTES,
                    limit=MAX_STUDY_BLOCKS):
    """One study block on each of `days`, at `start_minutes` after midnight
    (a day whose time has already passed is left out).

    A day where that time is already taken doesn't get a block on top of
    it: the block moves to right after whatever's in the way (repeating if
    that runs into something else), or is skipped when that would end after
    LATEST_STUDY_END_MINUTES. Only the last `limit` days are used (None = all
    of them -- for days the user picked themselves). Returns a StudyPlan of
    (start, end) pairs, plus (day, new start, what was in the way, what it
    now follows) for the moved ones -- the last two differ when moving past
    one event runs into another -- and (day, what it would have followed or
    None) for the skipped ones."""
    length = datetime.timedelta(minutes=length_minutes)
    busy = [
        (_start(e), _end(e), e["summary"]) for e in events
        if e["start"] != "All day"
    ]

    occurrences, shifted, skipped = [], [], []
    for day in sorted(days):
        start = time_utils.at_minutes(day, start_minutes)
        if start < ref:
            continue
        in_the_way = follows = None
        while True:
            clash = [b for b in busy if b[0] < start + length and b[1] > start]
            if not clash:
                break
            in_the_way = in_the_way or clash[0][2]
            latest = max(clash, key=lambda b: b[1])
            follows, start = latest[2], latest[1]
        end = start + length
        if end.date() != day or end.hour * 60 + end.minute > LATEST_STUDY_END_MINUTES:
            skipped.append((day, follows))
        else:
            occurrences.append((start, end))
            if in_the_way:
                shifted.append((day, start, in_the_way, follows))

    if limit:
        occurrences = occurrences[-limit:]
    first = occurrences[0][0].date() if occurrences else None
    shifted = [m for m in shifted if m[0] >= first] if first else []
    skipped = [k for k in skipped if first and k[0] >= first]
    return StudyPlan(occurrences, shifted, skipped)


def clock_text(minutes):
    return f"{minutes // 60 % 12 or 12:02d}:{minutes % 60:02d} {'AM' if minutes // 60 % 24 < 12 else 'PM'}"


def describe_plan(subject, plan, minutes, today):
    """The spoken question for a StudyPlan: how many blocks, when, and how
    they work around what's already on the calendar."""
    n = len(plan.occurrences)
    wanted = clock_text(minutes)
    kind = f"{subject} study" if subject else "study"

    if n == 1:
        start = plan.occurrences[0][0]
        when = time_utils.day_phrase(start.date(), today)
        text = f"Want me to add a {kind} block {when} at {start:%I:%M %p}?"
        if plan.shifted:
            _, _, in_the_way, follows = plan.shifted[0]
            after = "it" if follows == in_the_way else follows
            text += f" {wanted} is during {in_the_way}, so it goes right after {after}."
        return text

    dates = [start.date() for start, _ in plan.occurrences]
    first = time_utils.day_label(dates[0], today)
    last = time_utils.day_label(dates[-1], today)
    contiguous = all((b - a).days == 1 for a, b in zip(dates, dates[1:]))
    if contiguous:
        span = f"from {first} through {last}"
    else:
        # Days were left out (Code Ninjas days, say): name the days, so
        # "from today through Friday" doesn't sound like every one of them.
        names = time_utils.join_words([f"{d:%A}" if d != today else "today" for d in dates])
        span = names if dates[0] == today else f"on {names}"
    if not plan.shifted:
        text = f"Want me to add {n} {kind} blocks at {wanted}, {span}?"
    else:
        by_blocker = {}
        for day, new_start, in_the_way, follows in plan.shifted:
            by_blocker.setdefault((in_the_way, follows), []).append((day, new_start))
        notes = []
        for (in_the_way, follows), moves in by_blocker.items():
            times = {f"{new_start:%I:%M %p}" for _, new_start in moves}
            at = f", at {next(iter(times))}" if len(times) == 1 else ""
            after = "it" if follows == in_the_way else follows
            if len(plan.shifted) == n:
                notes.append(f"{wanted} is during {in_the_way}, so I'd put them right after {after}{at}.")
            else:
                days = time_utils.days_phrase([day for day, _ in moves])
                notes.append(f"On {days} {wanted} clashes with {in_the_way}, so those go right after {after}{at}.")
        lead = f"Want me to add {n} {kind} blocks {span}?"
        if len(plan.shifted) < n:
            lead = f"Want me to add {n} {kind} blocks at {wanted}, {span}?"
        text = f"{lead} " + " ".join(notes)

    if plan.skipped:
        days = time_utils.days_phrase([day for day, _ in plan.skipped])
        reason = f"{plan.skipped[0][1]} runs too late" if plan.skipped[0][1] else "it would run too late"
        text += f" I'd skip {days}, when {reason}."
    return text


def _when(event, today):
    day = time_utils.day_phrase(event_day(event), today)
    return day if event["start"] == "All day" else f"{day} at {event['start']}"


def make_study_offer(items, events, ref):
    """For the nearest test with a subject and no prep scheduled: the spoken
    offer and the state to remember, or None if there's nothing to offer (or
    no room before it). The state is read back by
    calendar_manager._handle_study_plan when the user answers."""
    for item in items:
        if not item.subject or item.prep_count:
            continue
        offer = offer_for(item, events, ref)
        if offer:
            return offer
    return None


def offer_for(item, events, ref, minutes=DEFAULT_STUDY_MINUTES, days=None):
    """The spoken offer (and state to remember) of study blocks for one
    HeadsUp item at `minutes` after midnight, on `days` (default: every day
    from today to the day before it), or None if no block fits."""
    if days is None:
        days = days_before(item.event, ref)
    others = [e for e in events if e is not item.event]
    return offer_for_days(item.subject, days, others, ref, minutes, test=item.event)


def offer_for_days(subject, days, events, ref, minutes=DEFAULT_STUDY_MINUTES, test=None):
    """Study blocks on the given days, whether or not there's a test behind
    them. Returns (spoken offer, state to remember) or None if nothing fits."""
    capped = MAX_STUDY_BLOCKS if test is not None else None  # your own days aren't trimmed
    plan = plan_study_days(days, events, ref, minutes, limit=capped)
    if not plan.occurrences:
        return None
    pending = {
        "intent": "STUDY_PLAN",
        "stage": "offer",
        "subject": subject,
        "test": test,
        "days": [d.isoformat() for d in days],
        "minutes": minutes,
        "occurrences": [(s.isoformat(), e.isoformat()) for s, e in plan.occurrences],
    }
    return describe_plan(subject, plan, minutes, ref.date()), pending


def build_morning_brief(include_offer=True):
    """The "good morning" text and, if it ends on an offer, the pending state
    to store so the next thing you say answers it. Returns (text, pending or
    None)."""
    ref = time_utils.now()
    upcoming = get_upcoming_events(days=HEADS_UP_DAYS)
    today = ref.date()
    todays = [e for e in upcoming if event_day(e) == today]
    items = find_heads_up(upcoming, ref)

    shown = [
        {
            "title": item.event["summary"],
            "when": _when(item.event, today),
            "soon": item.days_away <= 1,
            "subject": item.subject,
            "prep_count": item.prep_count,
        }
        for item in items[:MAX_HEADS_UP]
    ]
    offer = make_study_offer(items, upcoming, ref) if include_offer else None
    text = generate_brief(
        todays, heads_up=shown, offer=offer[0] if offer else None,
        more=max(len(items) - MAX_HEADS_UP, 0),
    )
    return text, (offer[1] if offer else None)
