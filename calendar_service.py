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

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# calendar.events scope: full read/write on events specifically, without
# the broader calendar-management permissions "calendar" would grant.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


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


def get_todays_events():
    """Return today's events as a list of dicts: {summary, start, end}.

    Times come back as clean 12-hour strings (e.g. "05:00 PM") rather than
    raw ISO datetimes, so the LLM never has to do timezone/format math to
    speak them back naturally.
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    now = datetime.datetime.now().astimezone()
    start_of_day = datetime.datetime.combine(now.date(), datetime.time.min).replace(
        tzinfo=now.tzinfo
    ).isoformat()
    end_of_day = datetime.datetime.combine(now.date(), datetime.time.max).replace(
        tzinfo=now.tzinfo
    ).isoformat()

    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=start_of_day,
            timeMax=end_of_day,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = events_result.get("items", [])
    simplified = []

    for event in events:
        start_raw = event["start"].get("dateTime", event["start"].get("date"))
        end_raw = event["end"].get("dateTime", event["end"].get("date"))

        try:
            start_dt = datetime.datetime.fromisoformat(start_raw)
            end_dt = datetime.datetime.fromisoformat(end_raw)
            start_clean = start_dt.strftime("%I:%M %p")
            end_clean = end_dt.strftime("%I:%M %p")
        except ValueError:
            start_clean = "All day"
            end_clean = ""

        simplified.append(
            {
                "summary": event.get("summary", "(no title)"),
                "start": start_clean,
                "end": end_clean,
            }
        )

    return simplified


def create_event(event_description: str) -> str:
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
        A short confirmation string.
    """
    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds)

    created = service.events().quickAdd(
        calendarId="primary", text=event_description
    ).execute()

    return f"Added: {created.get('summary', event_description)}"


if __name__ == "__main__":
    print("Testing Google Calendar connection directly...")
    events = get_todays_events()
    if not events:
        print("No events found for today.")
    for e in events:
        print(f"{e['start']} to {e['end']} - {e['summary']}")
