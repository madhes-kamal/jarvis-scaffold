"""
main.py

Run this with: python main.py

Jarvis sleeps until a wake-word model hears "Hey Jarvis" (voice.py), then
records the request and runs the flow below.

Flow for every message:
1. If it contains "good morning", speak the tuned briefing: what's left of
   today, a heads-up about tests/quizzes/deadlines in the next two weeks, and
   -- if there's no prep time scheduled -- an offer of study blocks.
2. Ask calendar_manager to classify + (if relevant) handle it -- covers
   read/create/update/delete, resolving to a real event ID rather than
   guessing. If it wasn't calendar-related, calendar_manager says so.
3. If it wasn't calendar-related (and wasn't just "good morning" on its
   own), fall through to general conversation.

If Jarvis asks a question (confirm a delete/move, or "which one?"), the
next utterance is treated as the answer without needing "Hey Jarvis"
again. Silence for ANSWER_TIMEOUT seconds cancels the question.

Press Ctrl+C to exit.
"""

import re
import time
import traceback

from voice import listen, speak, wait_for_wake_word, set_vocabulary
from calendar_service import get_upcoming_events
from heads_up import build_morning_brief
from conversation import run_turn
from calendar_manager import handle_calendar_request, awaiting_answer, clear_pending

# The wake word itself is detected by a dedicated model (see voice.py), not
# by reading transcripts. This only cleans up the few words of it that can
# land at the start of a request ("Hey Jarvis, what's..." said in one go).
WAKE_PATTERN = r"^\W*(?:hey[\s,]+)?jarvis\b[\s,:.!?-]*"

# How long to wait for the user to START speaking -- an answer to a
# question Jarvis just asked ("Delete X?"), or the request after waking
# him -- before giving up. Also what ends a false wake.
ANSWER_TIMEOUT = 8
REQUEST_TIMEOUT = 8

# How often to refresh the words Whisper is told to expect (your event titles).
VOCABULARY_REFRESH_SECONDS = 1800


def _strip_wake_word(text):
    """Remove a leading wake phrase and any punctuation Whisper left
    behind, so a bare "Hey Jarvis!" comes out empty instead of as "!"."""
    text = re.sub(WAKE_PATTERN, "", text, count=1, flags=re.IGNORECASE)
    text = text.strip().lstrip(" ,.!?;:-")
    return text if re.search(r"\w", text) else ""


def _refresh_vocabulary():
    """Tell Whisper about the titles on your calendar so it recognises them.
    Only a hint -- if the calendar can't be reached, nothing depends on it."""
    try:
        titles = {event["summary"] for event in get_upcoming_events(days=14)}
        set_vocabulary(sorted(titles)[:20])
    except Exception:
        pass


def main():
    history = []
    calendar_context = {"last_event": None, "pending": None}
    _refresh_vocabulary()
    last_refresh = time.monotonic()
    print("Listening for 'Hey Jarvis'. Press Ctrl+C to exit.")

    while True:
        if time.monotonic() - last_refresh > VOCABULARY_REFRESH_SECONDS:
            last_refresh = time.monotonic()  # even on failure: retry in 30 minutes, not every loop
            _refresh_vocabulary()

        if awaiting_answer(calendar_context):
            # Jarvis just asked a question, so the very next utterance is
            # the answer -- no wake word needed. (A wake word is still
            # accepted and stripped if the user says one anyway.)
            user_text = _strip_wake_word(
                listen(prompt="Listening for your answer...", timeout=ANSWER_TIMEOUT)
            )
            if not user_text:
                print("  [no answer heard -- dropping the pending question]")
                clear_pending(calendar_context)
                continue
        else:
            wait_for_wake_word()
            user_text = _strip_wake_word(
                listen(prompt="Listening for your request...", timeout=REQUEST_TIMEOUT)
            )
            if not user_text:
                continue

        print(f"You said: {user_text}")

        try:
            history, calendar_context = _handle_request(
                user_text, history, calendar_context
            )
        except Exception:
            # A failed API call (Google, Ollama, network) shouldn't kill
            # the assistant -- report it, drop any half-finished question,
            # and go back to listening.
            traceback.print_exc()
            clear_pending(calendar_context)
            try:
                speak("Sorry, something went wrong with that.")
            except Exception:
                traceback.print_exc()


def _handle_request(user_text, history, calendar_context):
    said_good_morning = "good morning" in user_text.lower()

    reply, calendar_context = handle_calendar_request(
        user_text, calendar_context, skip_read=said_good_morning
    )

    if said_good_morning:
        # The brief is built after the request is handled: it may end on an
        # offer ("want me to add study blocks?"), and that pending question
        # must not be eaten as the "answer" to "good morning" itself. If the
        # same sentence also asked for something, skip the offer.
        brief, offer = build_morning_brief(include_offer=reply is None)
        speak(brief)
        if offer:
            calendar_context["confirm_pending"] = offer  # next words answer it, no wake word

    if reply is not None:
        speak(reply)
    elif not said_good_morning:
        reply, history = run_turn(user_text, history)
        speak(reply)

    return history, calendar_context


if __name__ == "__main__":
    main()
