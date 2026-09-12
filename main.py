"""
main.py

Entry point. Run this with: python main.py

Pulls today's Google Calendar events, sends them to a local LLM (via
Ollama) for a casual summary, and prints it. This terminal version is the
precursor to the voice version -- once this is reliable, we swap print()
for text-to-speech.
"""

from calendar_service import get_todays_events
from brief_generator import generate_brief


def main():
    print("Fetching today's calendar...")
    events = get_todays_events()

    print("Generating your morning brief...\n")
    brief = generate_brief(events)

    print("=" * 50)
    print(brief)
    print("=" * 50)


if __name__ == "__main__":
    main()
