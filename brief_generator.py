"""
brief_generator.py

Takes a list of calendar events and asks a local LLM (via Ollama) to turn
them into a short, casual "morning brief".

This runs entirely on your machine through Ollama -- no API key, no cost,
no internet required once the model is downloaded.

Variation without sacrificing accuracy: a small pool of STYLES controls
phrasing/ordering only. The strict factual rules below apply identically
regardless of which style gets picked, so the facts never drift, only
how they're framed.
"""

import datetime
import random

import ollama

# Set this to "qwen2.5:1.5b" or "qwen3:0.6b" depending on what you have pulled
MODEL = "qwen2.5:1.5b"

STYLES = [
    "Start with a quick one-line sense of how busy or light the day looks, then walk through the events in order.",
    "Open with a brief, JARVIS-style greeting, then move straight into the first event and continue in order.",
    "Lead with whichever event is happening soonest, then cover the rest in chronological order.",
    "Start by stating how many things are on the schedule today, then list them in order.",
]


def generate_brief(events):
    today_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    style_instruction = random.choice(STYLES)

    if not events:
        events_text = "NO EVENTS TODAY"
    else:
        events_text = "\n".join(
            f"{i + 1}. {e['summary']} | {e['start']} - {e['end']}"
            if e["start"] != "All day"
            else f"{i + 1}. {e['summary']} | ALL DAY"
            for i, e in enumerate(events)
        )

    prompt = f"""You are JARVIS, an elegant and slightly witty AI personal assistant.

Create a short, natural, spoken-style morning briefing from the calendar data below.

IMPORTANT — THE CALENDAR DATA IS FACT:
- NEVER change a time.
- NEVER change an event name.
- NEVER invent an event.
- NEVER remove an event.
- NEVER combine events.
- NEVER split events.
- NEVER assume what an event means.
- NEVER add activities that are not listed.
- Keep every event in chronological order.
- Mention every event exactly once.
- You may only add natural connecting words around the calendar information.
- Address the user as "sir".
- Keep the response to 2-3 sentences.
- Do not use bullet points or a list.

STYLE FOR THIS BRIEFING (this changes phrasing and ordering only -- it
never overrides the facts above):
{style_instruction}

Today's date:
{today_date}

CALENDAR EVENTS:
{events_text}

Generate the briefing now. Use ONLY the information provided above."""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise calendar assistant. "
                    "Calendar events and times are immutable facts. "
                    "Never alter, infer, or invent calendar information."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.1,
            "think": False,
        },
    )

    raw_content = response.message.content

    # If using a reasoning model, isolate and drop any internal monologue.
    if "</think>" in raw_content:
        clean_content = raw_content.split("</think>")[-1].strip()
    else:
        clean_content = raw_content.strip()

    return clean_content


if __name__ == "__main__":
    # Test script in isolation with fake data. Runs it 3 times so you can
    # see the phrasing vary while the facts stay identical.
    print("Testing LLM generation with sample data (3 runs to show variation)...")

    sample_events = [
        {"summary": "SAMPLE TASK", "start": "05:00 PM", "end": "05:45 PM"},
        {"summary": "SAMPLE TASK 2", "start": "05:45 PM", "end": "06:30 PM"},
    ]

    for i in range(3):
        print(f"\n--- Run {i + 1} ---")
        print(generate_brief(sample_events))
