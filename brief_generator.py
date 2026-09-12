"""
brief_generator.py

Takes a list of calendar events and asks a local LLM (via Ollama) to turn
them into a short, casual "morning brief" -- the kind of thing you'd
actually want an assistant to say out loud, not a robotic list readout.

This runs entirely on your machine through Ollama -- no API key, no cost,
no internet required once the model is downloaded. It's also literally
the same setup you'll use for the offline fallback model on the Pi later,
so testing it now is testing that path too.

Setup: install Ollama (https://ollama.com), then run:
    ollama pull qwen2.5:1.5b
"""

import datetime 
import ollama

# Set this to "qwen2.5:1.5b" or "qwen3:0.6b" depending on what you have pulled
MODEL = "qwen2.5:1.5b"


def generate_brief(events):
    # Dynamically get today's actual day of the week and date
    today_date = datetime.datetime.now().strftime("%A, %B %d, %Y")

    if not events:
        events_text = "No events are scheduled for today."
    else:
        events_text = ""
        for e in events:
            if e['start'] == "All day":
                events_text += f"Event: {e['summary']} (All Day)\n"
            else:
                events_text += f"Event: {e['summary']} from {e['start']} to {e['end']}\n"

    # A highly restrictive system prompt with structural examples to coach a small LLM
    prompt = f"""You are JARVIS, an elegant, slightly witty, and highly intelligent AI personal assistant. 
Review the user's Google Calendar data for today and provide a highly conversational, spoken-style morning briefing.

[Rules]
1. Address the user directly as "sir" or "ma'am".
2. Never repeat the exact raw calendar layout or print bulleted lists. 
3. Blend the timings naturally into sentences (e.g., "You have X starting at Y, followed immediately by Z").
4. Keep the final response strictly under 3 sentences.

[Example of Great Output]
"Good morning, sir. Your schedule is perfectly clear until this afternoon, when you have a sync with the development team at 02:00 PM. Directly following that, you'll be wrapping up the weekly code review until 04:30 PM."

[Today's Date Context]
{today_date}

[Today's Raw Calendar Data]
{events_text}

[Task]
Generate the conversational briefing now based strictly on the raw data provided above."""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"think": True}, # Keeps reasoning enabled for thinking models
    )

    raw_content = response.message.content

    # If using a reasoning model, isolate and drop the internal monologue tags
    if "</think>" in raw_content:
        clean_content = raw_content.split("</think>")[-1].strip()
    else:
        clean_content = raw_content.strip()

    return clean_content


if __name__ == "__main__":
    # Test script in isolation with fake data to verify LLM performance
    print("Testing LLM generation with sample data...")
    sample_events = [
        {"summary": "SAMPLE TASK", "start": "05:00 PM", "end": "05:45 PM"},
        {"summary": "SAMPLE TASK 2", "start": "05:45 PM", "end": "06:30 PM"}
    ]
    print(generate_brief(sample_events))