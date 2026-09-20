"""
time_utils.py

Everything about "now" and "when" that Jarvis decides in plain Python, so
the LLM never has to guess a date:

- now() -- the ONE place the current time comes from
- parse_date_range() -- turns "tomorrow", "friday", "next week", "the 25th"
  into a real start/end datetime range
- spoken helpers -- "today", "tomorrow", "on Friday" for replies

No imports from the rest of the project, so it's easy to test by itself.
"""

import datetime
import re
from collections import namedtuple

# start/end are timezone-aware datetimes; label is how a reply refers to the
# range ("tomorrow", "on Friday", "this week").
DateRange = namedtuple("DateRange", "start end label")

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Words that only describe WHEN, so they never count as part of an event's
# title when matching ("delete tomorrow's break" is about "break").
RANGE_WORDS = set(WEEKDAYS) | {
    "today", "tomorrow", "tonight", "yesterday", "week", "weekend", "next",
    "this", "coming", "days", "day", "upcoming", "morning", "afternoon", "evening",
}

_MONTH_NAMES = (
    "january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
    "august|aug|september|sept|sep|october|oct|november|nov|december|dec"
)
_MONTH_NUMBER = {
    name: number
    for number, names in enumerate(
        ["jan january", "feb february", "mar march", "apr april", "may", "jun june",
         "jul july", "aug august", "sep sept september", "oct october",
         "nov november", "dec december"],
        start=1,
    )
    for name in names.split()
}
_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
}
_COUNT = r"\d+|" + "|".join(_NUMBER_WORDS)

# Spoken dates. Kept as patterns so strip_date_expressions() can cut the very
# same phrases out of a sentence.
_MONTH_THEN_DAY = re.compile(rf"\b({_MONTH_NAMES})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b")
_DAY_THEN_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_NAMES})\b")
_NUMERIC_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
_ORDINAL_DAY = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b")
_IN_N_DAYS = re.compile(rf"\b(?:in|next|coming)\s+(?:{_COUNT})\s+days?\b")


def now():
    """Current local time, timezone-aware."""
    return datetime.datetime.now().astimezone()


def start_of_day(day):
    return datetime.datetime.combine(day, datetime.time.min).astimezone()


def end_of_day(day):
    return datetime.datetime.combine(day, datetime.time.max).astimezone()


def _count(text):
    return int(text) if text.isdigit() else _NUMBER_WORDS[text]


def _next_weekday(today, weekday, following_week=False):
    """Date of the coming `weekday` (0=Monday), today included. With
    following_week, the one in NEXT calendar week (Monday-Sunday) instead."""
    if following_week:
        next_monday = today + datetime.timedelta(days=7 - today.weekday())
        return next_monday + datetime.timedelta(days=weekday)
    return today + datetime.timedelta(days=(weekday - today.weekday()) % 7)


def strip_date_expressions(text):
    """Blank out spoken dates ("september 25", "the 25th", "9/25", "in 3
    days") so their numbers aren't mistaken for an hour of the day."""
    text = text.lower()
    for pattern in (_MONTH_THEN_DAY, _DAY_THEN_MONTH, _NUMERIC_DATE, _ORDINAL_DAY, _IN_N_DAYS):
        text = pattern.sub(" ", text)
    return text


def _explicit_date(text, today):
    """A calendar date spoken outright: "september 25", "25th of sept",
    "9/25", "the 25th". Dates already past roll to next year (or next month
    for a bare day number). Returns None if there isn't one or it's invalid."""
    month = day = None
    if match := _MONTH_THEN_DAY.search(text):
        month, day = _MONTH_NUMBER[match[1]], int(match[2])
    elif match := _DAY_THEN_MONTH.search(text):
        month, day = _MONTH_NUMBER[match[2]], int(match[1])
    elif match := _NUMERIC_DATE.search(text):
        month, day = int(match[1]), int(match[2])

    try:
        if month:
            date = datetime.date(today.year, month, day)
            return date if date >= today else datetime.date(today.year + 1, month, day)
        if match := _ORDINAL_DAY.search(text):
            day = int(match[1])
            date = today.replace(day=day) if day >= today.day else None
            if date is None:
                first_of_next = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
                date = first_of_next.replace(day=day)
            return date
    except ValueError:
        return None
    return None


def parse_date_range(text, ref):
    """Find the day or span of days a message talks about. Returns a
    DateRange, or None when the message names no time frame (the caller
    decides what the default is).

    `ref` is "now". A range that includes today starts at `ref`, not at
    midnight, so anything already finished is left out; ranges on later days
    cover the whole day. Only unambiguous phrasings are recognised, and the
    date arithmetic is all here in Python.
    """
    text = text.lower().replace("’", "'")
    today = ref.date()

    def one_day(date, label):
        start = ref if date == today else start_of_day(date)
        return DateRange(start, end_of_day(date), label)

    def span(first, last, label):
        return DateRange(ref if first <= today else start_of_day(first), end_of_day(last), label)

    if match := re.search(rf"\b(?:next|coming)\s+({_COUNT})\s+days?\b", text):
        n = _count(match[1])
        return DateRange(ref, ref + datetime.timedelta(days=n), f"in the next {n} days")

    if match := re.search(rf"\bin\s+({_COUNT})\s+days?\b", text):
        n = _count(match[1])
        return one_day(today + datetime.timedelta(days=n), f"in {n} days")

    if re.search(r"\bday after tomorrow\b", text):
        return one_day(today + datetime.timedelta(days=2), "the day after tomorrow")

    if re.search(r"\btomorrow\b", text):
        return one_day(today + datetime.timedelta(days=1), "tomorrow")

    next_week = re.search(r"\bnext week\b", text)
    for index, name in enumerate(WEEKDAYS):
        if re.search(rf"\b{name}s?\b", text):
            following = bool(next_week or re.search(rf"\bnext {name}\b", text))
            date = _next_weekday(today, index, following_week=following)
            return one_day(date, "today" if date == today else day_phrase(date, today))

    if date := _explicit_date(text, today):
        return one_day(date, day_phrase(date, today))

    if re.search(r"\b(today|tonight|this (?:morning|afternoon|evening))\b", text):
        return one_day(today, "today")

    if re.search(r"\bweekend\b", text):
        saturday = today + datetime.timedelta(days=max(5 - today.weekday(), 0))
        sunday = saturday + datetime.timedelta(days=1) if today.weekday() < 6 else today
        return span(saturday, sunday, "this weekend")

    if next_week:
        monday = today + datetime.timedelta(days=7 - today.weekday())
        return span(monday, monday + datetime.timedelta(days=6), "next week")

    if re.search(r"\bweek\b", text):
        # Through Sunday -- but on Saturday or Sunday that would be a day or
        # two, and "how does my week look" then means the coming week.
        if today.weekday() >= 5:
            return span(today, today + datetime.timedelta(days=6), "this week")
        sunday = today + datetime.timedelta(days=6 - today.weekday())
        return span(today, sunday, "this week")

    if re.search(r"\b(upcoming|coming up)\b", text):
        return DateRange(ref, ref + datetime.timedelta(days=7), "in the next week")

    return None


def day_phrase(day, today):
    """How a reply refers to a date: "today", "tomorrow", "on Friday" (within
    the next week), or "on September 25"."""
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if delta == -1:
        return "yesterday"
    if 1 < delta < 7:
        return f"on {day:%A}"
    return f"on {day:%B} {day.day}"


def day_heading(day, today):
    """Start-of-sentence version of day_phrase: "Today", "Friday"."""
    phrase = day_phrase(day, today)
    phrase = phrase[3:] if phrase.startswith("on ") else phrase
    return phrase[0].upper() + phrase[1:]


def time_reply(ref):
    return f"It's {ref.strftime('%I:%M %p').lstrip('0')}."


def date_reply(ref):
    return f"Today is {ref:%A}, {ref:%B} {ref.day}."


def join_words(items):
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def days_phrase(dates):
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
    return join_words([
        f"{run[0]:%A} through {run[-1]:%A}" if len(run) >= 3
        else join_words([f"{d:%A}" for d in run])
        for run in runs
    ])


def day_label(day, today):
    """day_phrase without the leading "on": "today", "tomorrow", "Friday",
    "October 2" -- for "through Friday", "until tomorrow"."""
    phrase = day_phrase(day, today)
    return phrase[3:] if phrase.startswith("on ") else phrase


def at_minutes(day, minutes):
    """The moment `minutes` after midnight on `day`, local and timezone-aware."""
    return datetime.datetime.combine(
        day, datetime.time(minutes // 60 % 24, minutes % 60)
    ).astimezone()
