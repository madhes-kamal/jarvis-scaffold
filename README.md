# JARVIS Scaffold — Voice + Calendar Read/Write

Talk to it, and it can read and change your Google Calendar: add, move,
delete, and recolor events. A small local LLM handles the language work
(classifying what you asked, tidying up phrasing, general chat), while
Python makes every decision that touches your calendar — it never lets the
model pick an event or act on a guess.

## What's here

- `time_utils.py` — the current time and every date calculation ("tomorrow", "friday", "next week", "the 25th" → a real date range); pure Python, no LLM
- `calendar_service.py` — Google Calendar OAuth, reading events over any date range (with real IDs and colors), and low-level create/update/delete/recolor
- `calendar_manager.py` — classifies each request (create/read/update/delete/color/time/none), answers time and look-ahead questions, resolves "that meeting" to a real event ID, asks for confirmation before moves and deletes, and handles follow-up answers ("the 7:15 one", "yes")
- `brief_generator.py` — builds the morning briefing from templates in Python (no LLM involved — guarantees accuracy, used only when you say "good morning")
- `conversation.py` — general chat fallback for anything not calendar-related, plus `extract_event_title`, the one small job the model does when you add an event (naming it). `extract_event_phrase` is the old "model writes the whole quick-add phrase" helper and is no longer used
- `llm_client.py` — the single place every LLM call goes through; swap models or backends here, not in the other files
- `voice.py` — wake-word detection (openWakeWord), speech-to-text (Whisper, local) and text-to-speech (edge-tts, free neural voices, needs internet)
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

Say "Hey Jarvis", then your request (in the same breath or after a
pause). A small dedicated wake-word model (openWakeWord's pretrained
"hey jarvis") listens to the raw mic audio; Whisper only runs once it
fires, so nothing is transcribed while Jarvis is idle. Try:

- "What time is it?" / "What's the date?"
- "What's on my calendar today?" (only what hasn't finished yet)
- "What do I have tomorrow?" / "How does my week look?" / "What's on Friday?"
- "Do I have a chemistry test this week?" (looks two weeks ahead if you don't say when)
- "Schedule a chemistry test next Friday at 2.30 p.m."
- "Add dinner tomorrow from 7 to 8pm" / "Add a study session at 5 on Thursday for 2 hours"
- "Add a break after chemistry" / "Add a 15 minute break before my dentist appointment"
- "Add chemistry study blocks every day at 6.30 pm" / "Add chess every tuesday and thursday at 5 for 3 weeks"
- "Schedule study blocks after my bus every day" (then answer yes or no)
- "Add a 5 minute scrolling break starting now" / "Add a break in 20 minutes"
- "Move my break to 4.30pm"
- "Delete the 7pm one"
- "Make my break red" / "Change the color of chemistry to light blue" / "Make it green"
- "Good morning" (spoken briefing)

**Time and look-ahead.** Jarvis reads the clock from your computer, and
reading the calendar only reports events that haven't finished: one in
progress still counts, one that ended this morning doesn't. Days are
understood in plain Python (`time_utils.parse_date_range`): today, tomorrow,
a weekday ("friday"; "next friday" means the one in next week), "this
week", "next week", "this weekend", "the next 3 days", "in 3 days", and dates
like "the 25th" or "september 30". Anything that isn't a day is treated as a
search: "do I have a chemistry test?" only reports events with those words in
the title ("test", "exam" and "quiz" count as the same word). Replies for
other days say which day: "Tomorrow: School at 08:20 AM. Friday: ...".

**Routines.** In a week summary, an event that shows up on 3 or more
different days is described as a pattern instead of listed day by day:
"You have School on weekdays from 08:20 AM to 02:42 PM; Code Ninjas on
Tuesday and Thursday from 04:40 PM to 08:00 PM, and on Saturday from 10:40
AM to 03:30 PM. Other than that: Today: ..." (`MIN_REPEATS` in
`calendar_manager.py`). Only done for spans of a week or less, where a weekday
name means one specific day.

**Adding events.** The day and times are worked out in Python
(`calendar_manager._parse_new_event`), never by the model -- a small model
once turned "next Friday" into a Sunday. It understands "at 2.30 p.m.",
"from 3 to 4pm", "at 715", "starting now", "in 20 minutes", "for an hour" /
"a 5 minute break", and "after/before <another event>". With no end time and
no length, an event lasts 30 minutes (`DEFAULT_EVENT_MINUTES`). A bare "at 3"
is the next 3 o'clock if it's today; on another day 9-11 mean morning and
everything else afternoon or evening, so say "am" or "pm" when it matters.
A time that has already passed today, with no day named, means tomorrow.
The model only picks the event's title. If Jarvis can't tell when, it asks
instead of guessing.

**Repeating events.** "Add chemistry study blocks every day at 6.30 pm" and
"schedule study blocks after my bus every day" create one event per day.
Understood: every day / daily, weekdays, weekends, "every tuesday and
thursday", "weekly", and how long it goes on: "for 3 weeks", "until friday",
"until my chemistry test on friday", "next week". With no end given it covers
`DEFAULT_REPEAT_DAYS` (14) days, never more than `MAX_REPEAT_DAYS` (90), and
Jarvis says where it stops. Tied to another event ("after my bus"), each day's
event is placed right after *that day's* bus, and days without one are skipped
(no bus on weekends means weekdays only). Because that is a lot of events, Jarvis
asks first -- "Add Chemistry study block right after Bus on weekdays, through
October 2? That's 10 events." -- and only creates them on a yes. These are
ordinary separate events (not a Google "recurring series"), so changing or
deleting one leaves the rest alone. Editing or deleting a whole series in one
go isn't supported yet.

**Changing several at once.** "Change the color of each of those chemistry
study blocks to dark green", "make them red" (the events just added), "reset
all my chemistry study blocks to the default color". Words like all / each /
every / those / them mean every match instead of one. Recoloring just happens;
deleting several asks first, and says what it found: "Delete all 6 events (5
Chemistry study block, 1 Chemistry test) over the next 7 days?". If Jarvis asks
"which one?", "all of them" is a valid answer. Moving many events at once isn't
supported. Like everything that edits events, this only looks at today plus the
next 7 days.

Moving, deleting and recoloring look at today plus the next 7 days
(`TARGET_DAYS` in `calendar_manager.py`). If you don't name a day, an event
still ahead today wins over the same title later in the week; name one
("delete tomorrow's break", "the friday one") to pick another. Which-one
questions and confirmations include the day when it isn't today.

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

**Wake word tuning:** every wake prints `[wake word heard, score 0.94]`.
If Jarvis misses you, lower `WAKE_THRESHOLD` in `voice.py` (default 0.5);
if it wakes on similar-sounding phrases ("Hey Travis"), raise it. A false
wake is harmless — it just times out after `REQUEST_TIMEOUT` seconds of
silence. The models ship with the pip package; on a fresh install, if the
model file is missing, run
`python -c "import openwakeword; openwakeword.utils.download_models()"`.

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
- If Jarvis never wakes, look for the `[wake word heard ...]` line: if it never
  appears, lower `WAKE_THRESHOLD` or check the mic level (Windows Sound
  settings → Input). If it wakes but nothing is transcribed, check your OS's microphone
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
