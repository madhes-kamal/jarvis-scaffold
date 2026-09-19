# JARVIS Scaffold — Voice + Calendar Read/Write

Talk to it, and it can read and change your Google Calendar: add, move,
delete, and recolor events. A small local LLM handles the language work
(classifying what you asked, tidying up phrasing, general chat), while
Python makes every decision that touches your calendar — it never lets the
model pick an event or act on a guess.

## What's here

- `calendar_service.py` — Google Calendar OAuth, reading events (with real IDs and colors), and low-level create/update/delete/recolor
- `calendar_manager.py` — classifies each request (create/read/update/delete/color/none), resolves "that meeting" to a real event ID, asks for confirmation before moves and deletes, and handles follow-up answers ("the 7:15 one", "yes")
- `brief_generator.py` — builds the morning briefing from templates in Python (no LLM involved — guarantees accuracy, used only when you say "good morning")
- `conversation.py` — general chat fallback for anything not calendar-related, plus the phrase-cleanup helper `calendar_manager.py` reuses for create
- `llm_client.py` — the single place every LLM call goes through; swap models or backends here, not in the other files
- `voice.py` — speech-to-text (Whisper, local) and text-to-speech (edge-tts, free neural voices, needs internet)
- `main.py` — the voice loop; ties everything above together

## Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**PyAudio is the one dependency that sometimes fights back** (it wraps a
system audio library, not pure Python):
- **Mac**: `brew install portaudio` first, then re-run pip install
- **Windows**: if the normal install fails, `pip install pipwin` then `pipwin install pyaudio`
- **Linux**: `sudo apt-get install portaudio19-dev` first, then re-run pip install

### 2. Google Calendar

The app needs write access to calendar *events* (`calendar.events` scope).
If you already ran an earlier version with a different scope, **delete
`token.json`** in this folder before running — an old token won't carry
the new permissions. You'll be asked to log in again the first time you
run it.

(If you haven't set up `credentials.json` yet at all: Google Cloud
Console → enable Calendar API → OAuth client ID → Desktop app → download
as `credentials.json`.)

### 3. Ollama model

```bash
ollama pull qwen3:1.7b
```

The model name lives in exactly one place: `llm_client.py`'s `MODEL`
constant.

### 4. Run it

```bash
python main.py
```

Say "Hey Jarvis" followed by your request. Each utterance is transcribed
locally, and Jarvis only responds when the transcript contains the wake
phrase. Try:

- "What's on my calendar today?"
- "Add a 5 minute scrolling break starting now"
- "Move my break to 4.30pm"
- "Delete the 7pm one"
- "Make my break red" / "Change the color of chemistry to light blue" / "Make it green"
- "Good morning" (spoken briefing)

**Follow-up questions don't need the wake word.** When Jarvis asks
something ("Move Break to 04:30 PM?" or "Which one did you mean?"), just
answer. If you say nothing for 8 seconds (`ANSWER_TIMEOUT` in `main.py`)
the question is dropped. Anything that isn't a clear yes or no is treated
as a new request instead of a confirmation, and an unclear answer never
confirms a delete or move.

**Colors:** everyday words are mapped to the nearest Google Calendar event
color (red → Tomato, blue → Blueberry, light blue → Peacock, green →
Basil, purple → Grape, pink → Flamingo, orange → Tangerine, yellow →
Banana, gray → Graphite, plus Sage and Lavender; the palette is defined in
`calendar_service.EVENT_COLORS` and the word list in
`calendar_manager._COLOR_WORDS`). Say "default color" to reset one.

Press `Ctrl+C` to exit.

## Debugging tips

- To test the calendar logic without voice in the way:
  ```python
  from calendar_manager import handle_calendar_request
  reply, ctx = handle_calendar_request("cancel my dentist appointment", {"last_event": None, "pending": None})
  print(reply)
  ```
- The `[calendar intent: ...]` debug line prints on every message. If it's
  classifying things wrong (e.g. calling a delete request CREATE), tighten
  the few-shot examples in `calendar_manager.py`'s `classify_intent` — a
  wrong example there fixes more than more abstract instructions do.
- Event matching in `_match_events` is intentionally simple (hour + am/pm
  matching, then fuzzy title-word overlap). Everything after "to <time>" is
  treated as the *new* time, not as a description of the event. If it's
  failing to find an obvious event, that function is the one to improve,
  not the LLM prompt. The `[match: ...]` lines show each filtering step.
- If voice transcription is garbled, try a bigger Whisper model in
  `voice.py` (`WhisperModel("small", ...)` instead of `"base"`) — slower,
  more accurate. Whisper's VAD filter and no-speech check are on, so
  silence and room noise should transcribe to nothing.
- If nothing happens when you talk, check your OS's microphone
  permissions for your terminal/VS Code — this trips up a lot of people
  on Mac and Windows.
- Errors during a request (a Google API failure, Ollama not running) are
  printed with a traceback, Jarvis says "something went wrong", and the
  loop keeps running.
- Run `python brief_generator.py` on its own to see 5 sample briefings
  back to back — the facts stay identical while the phrasing varies.

## What to try next

- Swap edge-tts for a local TTS (Piper) so nothing needs internet — same
  idea as the LLM model swap, better for an offline device
- Focus mode: a small Python state machine for reminders, break requests
  ("can I have a 5 minute break?" → insert it and shift the rest of the
  day), and lenient-vs-strict rules about distractions
