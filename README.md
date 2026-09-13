# JARVIS Scaffold — Voice + Calendar Read/Write

Talk to it, and it can actually change your calendar (not just read it),
using tool calling: the LLM decides when to call `create_event` based on
what you say, instead of you writing if/else logic for every phrase.

## What's here

- `calendar_service.py` — Google Calendar OAuth, reading events (with real IDs), and low-level create/update/delete
- `calendar_manager.py` — classifies each request (create/read/update/delete/none) and resolves "that meeting" to a real event ID before acting; never lets the model guess at a destructive action
- `brief_generator.py` — builds the morning briefing from templates in Python (no LLM involved — guarantees accuracy, used only when you say "good morning")
- `conversation.py` — general chat fallback for anything not calendar-related, plus the phrase-cleanup helper `calendar_manager.py` reuses for create
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

### 2. Google Calendar — re-authorize with the new permissions

This version needs write access to calendar *events* specifically
(`calendar.events` scope). If you already ran an earlier version with a
different scope, **delete `token.json`** in this folder before running —
an old token won't carry the new permissions. You'll be asked to log in
again the first time you run it.

(If you haven't set up `credentials.json` yet at all, see the earlier
setup steps: Google Cloud Console → enable Calendar API → OAuth client ID
→ Desktop app → download as `credentials.json`.)

### 3. Ollama model

Tool calling (the LLM actually deciding to create an event) works far
more reliably on a bigger model than 1.5B — pull the 7B version:

```bash
ollama pull qwen2.5:7b
```

If your machine can't comfortably run 7B, you can drop `MODEL` back to
`qwen2.5:1.5b` in `conversation.py`, but watch for it claiming to add
events without actually calling the tool — check your real calendar to
confirm, don't just trust what it says out loud.

### 4. Run it

```bash
python main.py
```

Press Enter, then talk. Try:
- "What's on my calendar today?"
- "Add a 5 minute scrolling break starting now"
- "Schedule a 30 minute study session at 3pm called Focus block"

Type `quit` (no talking) to exit.

## Debugging tips

- If tool calls aren't happening at all, test `calendar_manager.py` directly, without voice in the way:
  ```python
  from calendar_manager import handle_calendar_request
  reply, ctx = handle_calendar_request("cancel my dentist appointment", {"last_event": None})
  print(reply)
  ```
- The `[calendar intent: ...]` debug line prints on every calendar-shaped
  message -- if it's classifying things wrong (e.g. calling a delete
  request CREATE), that's the place to tighten the prompt in
  `calendar_manager.py`'s `classify_intent`.
- Event matching in `_find_candidates` is intentionally simple (word
  overlap + hour matching) -- if it's failing to find an obvious event,
  that function is the one to improve, not the LLM prompt.
- If voice transcription is garbled, try a bigger Whisper model in
  `voice.py` (`model="small"` instead of `"base"`) — slower, more accurate.
- If nothing happens when you talk, check your OS's microphone
  permissions for your terminal/VS Code — this trips up a lot of people
  on Mac and Windows.
- Run `python brief_generator.py` on its own to see 5 sample briefings
  back to back — good way to confirm the phrasing varies while every
  event/time stays word-for-word identical across runs (it's templated,
  not LLM-generated, so this should always hold).

## What to try next

- Give it an `update_event`/`delete_event` tool the same way `create_event`
  works, so it can move or cancel things, not just add them
- Start sketching the focus state machine as its own module — this is
  where "be lenient about 5 minutes, strict after an hour" logic will live
- Swap `pyttsx3` for a nicer-sounding local TTS (Piper) once the core
  loop feels solid — same idea as the LLM model swap, better quality,
  more setup
