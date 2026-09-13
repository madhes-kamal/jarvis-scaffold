"""
main.py

Run this with: python main.py

Two modes, and they can both fire on the same message:
- If you say "good morning", you get the tuned, high-accuracy calendar
  briefing (brief_generator.py) -- always.
- If the message ALSO contains an action word ("add", "schedule", etc.),
  it's additionally passed to general conversation (conversation.py) so
  the calendar-modifying request doesn't get silently dropped just
  because you said good morning in the same breath.
- If there's no greeting at all, everything goes to general conversation.

Type 'quit' and press Enter (no talking needed) to exit.
"""

from voice import listen, speak
from calendar_service import get_todays_events, create_event
from brief_generator import generate_brief
from conversation import run_turn, extract_event_phrase

ACTION_KEYWORDS = (
    "add", "schedule", "book", "create", "move", "cancel",
    "remove", "delete", "change", "reschedule",
)


def main():
    history = []
    print("Press Enter to talk, or type 'quit' to exit.")

    while True:
        typed = input("\n[Enter to talk] ")
        if typed.strip().lower() == "quit":
            break

        user_text = listen()
        print(f"You said: {user_text}")

        lowered = user_text.lower()
        said_good_morning = "good morning" in lowered
        has_action = any(word in lowered for word in ACTION_KEYWORDS)

        if said_good_morning:
            events = get_todays_events()
            speak(generate_brief(events))

        if has_action:
            # Deterministic path: Python already decided this needs to
            # happen, so we don't ask the model to decide again -- just
            # clean the phrase and create the event directly.
            phrase = extract_event_phrase(user_text)
            print(f"  [quick-add phrase: {phrase}]")
            confirmation = create_event(phrase)
            speak(confirmation)
        elif not said_good_morning:
            reply, history = run_turn(user_text, history)
            speak(reply)


if __name__ == "__main__":
    main()
