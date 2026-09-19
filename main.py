"""
main.py

Run this with: python main.py

Flow for every message:
1. If it contains "good morning", speak the tuned briefing.
2. Ask calendar_manager to classify + (if relevant) handle it -- covers
   read/create/update/delete, resolving to a real event ID rather than
   guessing. If it wasn't calendar-related, calendar_manager says so.
3. If it wasn't calendar-related (and wasn't just "good morning" on its
   own), fall through to general conversation.

Press Ctrl+C to exit.
"""

import re

from voice import listen, speak
from calendar_service import get_todays_events
from brief_generator import generate_brief
from conversation import run_turn
from calendar_manager import handle_calendar_request


def main():
    history = []
    calendar_context = {"last_event": None, "pending": None}
    print("Listening for 'Hey Jarvis'. Press Ctrl+C to exit.")

    while True:
        user_text = listen(show_status=False)

        wake_match = re.search(r"\bhey\s+jarvis\b", user_text, re.IGNORECASE)
        if not wake_match:
            continue

        user_text = re.sub(
            r"\bhey\s+jarvis\b[:,]?\s*", "", user_text, count=1, flags=re.IGNORECASE
        ).strip()
        if not user_text:
            user_text = listen(show_status=False)
            if not user_text.strip():
                continue

        said_good_morning = "good morning" in user_text.lower()
        if said_good_morning:
            speak(generate_brief(get_todays_events()))

        reply, calendar_context = handle_calendar_request(
            user_text, calendar_context, skip_read=said_good_morning
        )

        if reply is not None:
            speak(reply)
        elif not said_good_morning:
            reply, history = run_turn(user_text, history)
            speak(reply)


if __name__ == "__main__":
    main()
