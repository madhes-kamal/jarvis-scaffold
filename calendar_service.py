"""
calendar_service.py

Handles authenticating with Google Calendar, fetching today's events, and
creating new ones.

First run: this opens a browser window asking you to log into Google and
approve calendar access. After that, a token.json file is saved so you
won't have to log in again until the token expires.
"""

import datetime
import os.path
import uuid

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

import time_utils

# calendar.events scope: full read/write on events specifically, without
# the broader calendar-management permissions "calendar" would grant.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# Every event Jarvis creates carries a hidden tag saying which request it came
# from, so "the chemistry study blocks" can mean exactly that set of events
# (found by the tag, not guessed from titles). It lives in the event's
# private extended properties: invisible in Google Calendar, kept by Google.
GROUP_KEY = "jarvis_group"

# Google Calendar's fixed per-event color palette (colorId -> name as
# shown in the Calendar UI). Events with no colorId use the calendar's
# own default color.
EVENT_COLORS = {
    "1": "Lavender",
    "2": "Sage",
    "3": "Grape",
    "4": "Flamingo",
    "5": "Banana",
    "6": "Tangerine",
    "7": "Peacock",
    "8": "Graphite",
    "9": "Blueberry",
    "10": "Basil",
    "11": "Tomato",
}


def _get_credentials():
    """Load cached credentials, or run the OAuth flow if none exist yet."""
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "credentials.json not found. Download it from the Google "
                    "Cloud Console (see README.md) and put it in this folder."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    return creds


def _simplify_event(event):
    start_raw = event["start"].get("dateTime", event["start"].get("date"))
    end_raw = event["end"].get("dateTime", event["end"].get("date"))
    # All-day events carry a bare "date". Don't detect them by fromisoformat
    # failing: since Python 3.11 it accepts "2026-09-20", so they were being
    # read as timed events at 12:00 AM.
    if "dateTime" in event["start"]:
        start_dt = datetime.datetime.fromisoformat(start_raw)
        end_dt = datetime.datetime.fromisoformat(end_raw)
        start_clean = start_dt.strftime("%I:%M %p")
        end_clean = end_dt.strftime("%I:%M %p")
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()
    else:
        start_clean = "All day"
        end_clean = ""
        start_iso = start_raw  # "YYYY-MM-DD"; the end date is exclusive
        end_iso = end_raw
    return {
        "id": event["id"],
        "summary": event.get("summary", "(no title)"),
        "start": start_clean,
        "end": end_clean,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "color": EVENT_COLORS.get(event.get("colorId"), "default"),
        # which Jarvis request created it (None for events made elsewhere)
        "group": ((event.get("extendedProperties") or {}).get("private") or {}).get(GROUP_KEY),
        # for one occurrence of a repeating event: the id of the whole series
        "recurring_id": event.get("recurringEventId"),
        # True for the repeating event itself (its id is the whole series)
        "is_series": bool(event.get("recurrence")),
        "created": event.get("created"),
    }


def event_day(event):
    """The calendar date an event starts on."""
    return datetime.date.fromisoformat(event["start_iso"][:10])


def has_not_ended(event, now):
    """True while an event is upcoming or in progress. All-day events count
    for the whole day (their end date is exclusive)."""
    if event["start"] == "All day":
        return datetime.date.fromisoformat(event["end_iso"][:10]) > now.date()
    return datetime.datetime.fromisoformat(event["end_iso"]) > now


def get_events(start, end):
    """Return the events that overlap [start, end] as a list of dicts:
    {id, summary, start, end, start_iso, end_iso, color}.

    `start`/`end` are timezone-aware datetimes. Google's timeMin keeps only
    events whose END is after it, so passing start=now means "not finished
    yet": events still in progress are included, ones already over are not.

    `start`/`end` in the result are clean 12-hour display strings (e.g.
    "05:00 PM") for speaking out loud. `start_iso`/`end_iso` are the real
    datetimes, kept around so update/delete can target an exact event
    instead of guessing from a description.
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    return [_simplify_event(event) for event in events_result.get("items", [])]


def get_todays_events():
    """The whole of today, including events that are already over."""
    today = time_utils.now().date()
    return get_events(time_utils.start_of_day(today), time_utils.end_of_day(today))


def get_upcoming_events(days: int = 1) -> list:
    """Events that haven't finished yet, from right now through the end of
    the day `days` days from today (days=1: the rest of today, days=7: the
    next week).

    Args:
        days: How many calendar days to cover, counting today.
    """
    now = time_utils.now()
    last_day = now.date() + datetime.timedelta(days=max(days, 1) - 1)
    return get_events(now, time_utils.end_of_day(last_day))


def delete_event(event_id: str) -> str:
    """Delete an event by its ID. Never called with a description --
    the caller (calendar_manager.py) must resolve a description to a
    real ID first, so we never delete the wrong thing on a guess."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    service.events().delete(calendarId="primary", eventId=event_id).execute()
    return "Deleted."


def update_event(event_id: str, new_start_iso: str = None, new_end_iso: str = None) -> str:
    """Update an event's time by its ID. Only pass the fields you want
    changed; ISO strings should already include a timezone offset."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    body = {}
    if new_start_iso:
        body["start"] = {"dateTime": new_start_iso}
    if new_end_iso:
        body["end"] = {"dateTime": new_end_iso}

    service.events().patch(calendarId="primary", eventId=event_id, body=body).execute()
    return "Updated."


def set_event_color(event_id: str, color_id: str = None) -> str:
    """Set an event's color by its ID. `color_id` is a key of EVENT_COLORS;
    None resets the event to the calendar's default color."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    # A null colorId in a patch clears the field.
    service.events().patch(
        calendarId="primary", eventId=event_id, body={"colorId": color_id}
    ).execute()
    return "Updated."


def _for_each_event(event_ids, action):
    """Run action(service, event_id) for each id over one connection.
    Returns (how many succeeded, error or None) -- a failure part way still
    reports what was already done, so the caller can say "changed 3 of 5"."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    done = 0
    try:
        for event_id in event_ids:
            action(service, event_id)
            done += 1
    except Exception as error:
        return done, error
    return done, None


def set_event_colors(event_ids, color_id=None):
    """set_event_color for many events at once. Returns (done, error)."""
    return _for_each_event(
        event_ids,
        lambda service, event_id: service.events().patch(
            calendarId="primary", eventId=event_id, body={"colorId": color_id}
        ).execute(),
    )


def update_events(items):
    """Change the time of several events at once. `items` is a list of
    (event_id, new_start_iso, new_end_iso), ISO strings with a timezone
    offset. Returns (done, error), like the other bulk calls."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    done = 0
    try:
        for event_id, start_iso, end_iso in items:
            service.events().patch(
                calendarId="primary", eventId=event_id,
                body={"start": {"dateTime": start_iso}, "end": {"dateTime": end_iso}},
            ).execute()
            done += 1
    except Exception as error:
        return done, error
    return done, None


def delete_events(event_ids):
    """delete_event for many events at once. Returns (done, error)."""
    return _for_each_event(
        event_ids,
        lambda service, event_id: service.events().delete(
            calendarId="primary", eventId=event_id
        ).execute(),
    )


def new_group_id():
    return uuid.uuid4().hex[:12]


def _tags(group):
    return {"private": {GROUP_KEY: group}}


def insert_event(title: str, start_iso: str, end_iso: str) -> dict:
    """Create an event at an exact time. Unlike create_event, nothing is
    left for Google (or a language model) to interpret: the caller has
    already worked out the real start and end, as ISO strings that include a
    timezone offset.

    Returns the created event as a dict, same shape as get_events().
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    return _insert(service, title, start_iso, end_iso, new_group_id())


def _insert(service, title, start_iso, end_iso, group):
    created = service.events().insert(
        calendarId="primary",
        body={
            "summary": title,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
            "extendedProperties": _tags(group),
        },
    ).execute()
    return _simplify_event(created)


def insert_events(items):
    """Create several events over one connection, all tagged as one group.
    `items` is a list of (title, start_iso, end_iso).

    Returns (created, error): the events that were made, and the exception
    that stopped the run (None if everything went through). A failure part
    way doesn't hide what was already added -- the caller can say "added 6 of
    10" instead of leaving the user to find out from the calendar.
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    group = new_group_id()
    created = []
    try:
        for title, start_iso, end_iso in items:
            created.append(_insert(service, title, start_iso, end_iso, group))
    except Exception as error:
        return created, error
    return created, None


def get_calendar_timezone():
    """The calendar's time zone as Google names it ("America/Toronto"), which
    Google requires for a repeating event. It comes back on every events.list
    reply, so no extra permission is needed. None if it can't be read."""
    try:
        creds = _get_credentials()
        service = build("calendar", "v3", credentials=creds)
        return service.events().list(calendarId="primary", maxResults=1).execute().get("timeZone")
    except Exception:
        return None


def insert_recurring_event(title: str, start_iso: str, end_iso: str, rrule: str, timezone: str) -> dict:
    """Create ONE repeating event. `start_iso`/`end_iso` are its first
    occurrence, `rrule` is the recurrence line ("RRULE:FREQ=DAILY;COUNT=14"),
    and `timezone` the calendar's IANA zone name. Tagged as its own group.

    Returns the series' first event, same shape as get_events(); its id is
    the id of the whole series."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    created = service.events().insert(
        calendarId="primary",
        body={
            "summary": title,
            "start": {"dateTime": start_iso, "timeZone": timezone},
            "end": {"dateTime": end_iso, "timeZone": timezone},
            "recurrence": [rrule],
            "extendedProperties": _tags(new_group_id()),
        },
    ).execute()
    return _simplify_event(created)


def get_group_events(group, start, end):
    """The events Jarvis created together (same group tag) that overlap
    [start, end], repeating ones expanded into their occurrences, in order.
    Found by the tag, so it isn't limited to a few days or to matching titles."""
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)
    result = service.events().list(
        calendarId="primary",
        timeMin=start.isoformat(),
        timeMax=end.isoformat(),
        privateExtendedProperty=f"{GROUP_KEY}={group}",
        singleEvents=True,
        orderBy="startTime",
        maxResults=250,
    ).execute()
    return [_simplify_event(event) for event in result.get("items", [])]


def create_event(event_description: str) -> dict:
    """Create a calendar event from a natural-language description, the
    same way a person would type it into Google Calendar's quick-add box.

    Google parses the date, time, and title itself -- do NOT convert
    anything to ISO format or compute a timezone offset. Just pass
    through a natural description of what the user asked for.

    Args:
        event_description: A natural-language description including the
            title and time, e.g. "Dinner today from 7:15pm to 8pm" or
            "Break tomorrow 3pm to 3:10pm".

    Returns:
        The created event as a dict: {id, summary, start, end, start_iso,
        end_iso, color}.
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    created = service.events().quickAdd(
        calendarId="primary", text=event_description
    ).execute()

    return _simplify_event(created)


if __name__ == "__main__":
    print("Testing Google Calendar connection directly...")
    events = get_todays_events()
    if not events:
        print("No events found for today.")
    for e in events:
        print(f"{e['start']} to {e['end']} - {e['summary']}")
