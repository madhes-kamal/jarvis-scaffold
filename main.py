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
again. More generally, for FOLLOWUP_TIMEOUT seconds after ANY response
you can just keep talking -- no wake word needed; after that long with
nothing said, "Hey Jarvis" is required again. Saying "Hey Jarvis, stop"
(or "never mind" / "quiet") at any point -- even while Jarvis is still
mid-sentence -- cuts the response off and drops whatever was pending.

Press Ctrl+C to exit.
"""

import re
import time
import traceback

from voice import listen, speak, wait_for_wake_word, set_vocabulary, BargeIn
from calendar_service import get_upcoming_events
from heads_up import build_morning_brief
from conversation import run_turn
from calendar_manager import handle_calendar_request, awaiting_answer, clear_pending
from news_service import build_news_brief

# The wake word itself is detected by a dedicated model (see voice.py), not
# by reading transcripts. This only cleans up the few words of it that can
# land at the start of a request ("Hey Jarvis, what's..." said in one go).
WAKE_PATTERN = r"^\W*(?:hey[\s,]+)?jarvis\b[\s,:.!?-]*"

# "what's the news", "give me the headlines", "top stories" -- kept as its
# own check, same as WAKE_PATTERN and "good morning" below, so a small
# unrelated model never has to decide this isn't a calendar request.
NEWS_PATTERN = re.compile(r"\b(?:the\s+)?(?:news|headlines|top stories)\b", re.IGNORECASE)

# A bare cut-it-off command -- has to be the *whole* utterance ("stop", not
# "stop by the store"), so it never shadows a real request that happens to
# contain one of these words.
STOP_PATTERN = re.compile(
    r"^(?:stop(?:\s+talking)?|never\s*mind|quiet|shut\s+up)[.!]?$", re.IGNORECASE
)

# How long to wait for the user to START speaking before giving up. REQUEST
# is the window right after a fresh "Hey Jarvis"; FOLLOWUP is the (longer)
# window after any response during which no wake word is needed at all --
# covers both "answer the question Jarvis just asked" and "just keep
# talking". Once FOLLOWUP_TIMEOUT passes with nothing said, "Hey Jarvis" is
# required again.
REQUEST_TIMEOUT = 8
FOLLOWUP_TIMEOUT = 30

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


def _speak_ack(text):
    """speak() for a short acknowledgement ("Okay, stopping.", the "sorry,
    something went wrong" fallback) where a failure -- a second barge-in on
    top of it, or a TTS/network hiccup -- isn't worth chasing; just log it
    and move on rather than crashing the loop."""
    try:
        speak(text)
    except BargeIn:
        pass
    except Exception:
        traceback.print_exc()


def main():
    history = []
    calendar_context = {"last_event": None, "pending": None}
    _refresh_vocabulary()
    last_refresh = time.monotonic()
    print("Listening for 'Hey Jarvis'. Press Ctrl+C to exit.")

    # Once this many seconds from now pass with nothing said, "Hey Jarvis"
    # is required again; until then, any speech is treated as a follow-up.
    awake_until = 0.0
    # Text already captured (e.g. by a barge-in mid-response) waiting to be
    # handled -- skips listening again on the next loop.
    pending_text = None

    while True:
        if time.monotonic() - last_refresh > VOCABULARY_REFRESH_SECONDS:
            last_refresh = time.monotonic()  # even on failure: retry in 30 minutes, not every loop
            _refresh_vocabulary()

        if pending_text is not None:
            user_text = pending_text
            pending_text = None
        elif awaiting_answer(calendar_context) or time.monotonic() < awake_until:
            # Either Jarvis just asked a question, or we're still inside
            # the no-wake-word window after the last response -- either
            # way, just keep talking. (A wake word is still accepted and
            # stripped if the user says one anyway.)
            prompt = (
                "Listening for your answer..."
                if awaiting_answer(calendar_context)
                else "Listening (no need to say 'Hey Jarvis')..."
            )
            user_text = _strip_wake_word(listen(prompt=prompt, timeout=FOLLOWUP_TIMEOUT))
            if not user_text:
                if awaiting_answer(calendar_context):
                    print("  [no answer heard -- dropping the pending question]")
                    clear_pending(calendar_context)
                awake_until = 0.0
                continue
        else:
            wait_for_wake_word()
            user_text = _strip_wake_word(
                listen(prompt="Listening for your request...", timeout=REQUEST_TIMEOUT)
            )
            if not user_text:
                continue

        print(f"You said: {user_text}")

        if STOP_PATTERN.match(user_text):
            clear_pending(calendar_context)
            _speak_ack("Okay, stopping.")
            awake_until = 0.0
            continue

        try:
            history, calendar_context = _handle_request(
                user_text, history, calendar_context
            )
            awake_until = time.monotonic() + FOLLOWUP_TIMEOUT
        except BargeIn as e:
            # "Hey Jarvis" was heard while a reply was still playing, so it
            # was cut off mid-sentence -- calendar_context/history from this
            # turn are intentionally NOT saved (the reply may describe a
            # pending question the user never actually heard in full).
            # Whatever came right after the wake word -- "stop", nothing,
            # or a brand new ask -- is handled on the next loop, exactly
            # like a fresh wake.
            pending_text = _strip_wake_word(e.text) or None
            awake_until = 0.0
        except Exception:
            # A failed API call (Google, Ollama, network) shouldn't kill
            # the assistant -- report it, drop any half-finished question,
            # and go back to listening.
            traceback.print_exc()
            clear_pending(calendar_context)
            _speak_ack("Sorry, something went wrong with that.")
            awake_until = 0.0


def _handle_request(user_text, history, calendar_context):
    if NEWS_PATTERN.search(user_text):
        # Standalone, like a TIME question -- headlines aren't calendar
        # data and don't need the small model's judgment either.
        speak(build_news_brief())
        return history, calendar_context

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
