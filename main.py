"""
main.py

Run this with: python main.py

Two modes in one loop:
- Say/type anything containing "good morning" -> the tuned, high-accuracy
  calendar briefing (brief_generator.py)
- Anything else -> general conversation, including changing your
  calendar, using tool calling (conversation.py)

Type 'quit' and press Enter (no talking needed) to exit.
"""

from voice import listen, speak
from calendar_service import get_todays_events
from brief_generator import generate_brief
from conversation import run_turn


def main():
    history = []
    print("Press Enter to talk, or type 'quit' to exit.")

    while True:
        typed = input("\n[Enter to talk] ")
        if typed.strip().lower() == "quit":
            break

        user_text = listen()
        print(f"You said: {user_text}")

        if "good morning" in user_text.lower():
            events = get_todays_events()
            reply = generate_brief(events)
        else:
            reply, history = run_turn(user_text, history)

        speak(reply)


if __name__ == "__main__":
    main()
