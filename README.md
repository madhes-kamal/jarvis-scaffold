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
- `heads_up.py` — the look-ahead part of "good morning": finds tests/quizzes/deadlines in the next 14 days, checks whether prep time is scheduled, and plans study blocks around what's already on your calendar
- `conversation.py` — general chat fallback for anything not calendar-related, plus `extract_event_title`, the one small job the model does when you add an event (naming it). `extract_event_phrase` is the old "model writes the whole quick-add phrase" helper and is no longer used
- `news_service.py` — fetches top headlines from NewsAPI.org and builds the spoken news brief from templates in Python (no LLM involved, same reasoning as `brief_generator.py`), used when you ask for the news
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

### 4. News (optional)

Get a free key at [newsapi.org/register](https://newsapi.org/register), copy
`.env.example` to `.env`, and put it in:

```
NEWS_API_KEY=your-key-here
```

`.env` is gitignored, so the key never gets committed. Without a key,
Jarvis still runs fine — asking for the news just gets a short "I don't
have a news API key set up yet" instead of headlines.

### 5. Run it

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
- "What's the news?" / "Give me the headlines" (needs `NEWS_API_KEY`, see Setup step 4)
- "Hey Jarvis, stop" (cuts Jarvis off, even mid-response -- see "Hey Jarvis, stop" below)

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

**"Good morning".** Reads what's left of today, then a heads-up about tests,
quizzes, exams, midterms, finals, deadlines, presentations and things "due" in
the next 14 days (`HEADS_UP_DAYS` in `heads_up.py`), at most 3, nearest first.
Anything on a title that repeats on 3+ days is a routine, not a heads-up. For
each one it counts the study/prep events for the same subject that finish
*before* it starts ("Chemistry test" → events with "chemistry" and study, prep,
review, practice or homework in the title): "…Chemistry test on Friday at 10:00
AM, with 4 chemistry study blocks lined up before it" or "…and nothing is
scheduled to prep for it yet, so make sure to prep ahead of that."

**"What's the news?"** Checked before anything calendar-related — like the
wake word and "good morning", a plain regex (`NEWS_PATTERN` in `main.py`)
catches "news", "headlines" and "top stories" so a small unrelated model
never has to decide this isn't a calendar request. `news_service.py` fetches
top US headlines from NewsAPI.org (`get_headlines`) — the model never picks
which headlines exist, so it can't drop a real one or invent one that
wasn't reported. With 2 or more headlines, the small local model is then
given those exact headlines and sources and asked to narrate them
naturally, like a person reading the news out loud, instead of the old
fixed "...from Reuters. Meanwhile, ...from the BBC." template — it's told
to only reword the delivery, never add a fact, number or detail that isn't
literally in the headline. `_looks_complete` (`news_service.py`)
double-checks the model's narration still mentions every headline before
it's trusted; if the model call fails, or a headline quietly went missing,
it falls back to the old fixed template instead. A single headline always
uses the template directly — with nothing to narrate around, asking the
model added nothing but a chance to invent a second, fake story. Repeated
requests within 10 minutes (`CACHE_SECONDS`) reuse the last fetch instead
of spending another call — the free tier is 100 requests/day. No key set,
or the request fails: a short spoken line says so ("I don't have a news
API key set up yet, sir." / "I couldn't reach the news service, sir."),
never a crash or a made-up headline.

If the nearest test with a subject has no prep, the brief ends on an offer, and
your next words answer it (no wake word): "Want me to add 5 chemistry study
blocks at 06:30 PM, from today through Thursday?"
- **yes** adds them (45 minutes each, one a day up to the day before, at most
  the last 7 days).
- **a time** ("7", "3:30 pm", "how about 8") re-plans at that time.
- **no** asks what time works; **no thanks** / **not now** drops it.

**Asking for study blocks directly.** "Add study blocks for my chemistry test"
(or "study periods", and "sturdy blocks" / "study plots", which is what
speech-to-text tends to make of them) finds that test on your calendar and
makes the same offer the morning brief does, at 6:30 PM unless you say a time
("...at 3:30 pm"). If you don't name the subject, or Whisper mishears it
("camera"), and there are several tests, Jarvis lists them and asks which one.
Plural "study blocks" means a series; a singular "study block at 6pm" is one
event. With "every day" or "after my bus" the blocks stop the day before the
test. Anything that spans time without saying "every" -- "until Friday", "for 5
days" -- is a series too, and any series that lands on something already on
your calendar warns you how many overlap.

**Study blocks with no test, and days to leave out.** "Add chemistry study
blocks this week" (or "until Friday", "for 5 days") plans those days at 6:30
PM the same way, even when there's no test on the calendar. You can leave days
out: "...to the days I don't have Code Ninjas", "...except code ninjas days",
"...except Tuesday and Thursday", "...skip weekends". Left-out days are
dropped, not moved, and Jarvis names the days it's using ("today, Monday,
Wednesday and Friday"). If a test exists for that subject, the blocks always
stop the day before it, even if you said "this week". If you name a subject
that isn't a test on the calendar ("chemistry", but only a Business quiz
exists), Jarvis says so and offers the tests it does see.

The time is always checked against your calendar. If it's taken, that day's
block moves to right after whatever is in the way, and Jarvis says so: "On
Monday through Thursday 03:30 PM clashes with Bus, so those go right after it,
at 03:50 PM." A day where that would run past 10 PM (`LATEST_STUDY_END_MINUTES`)
is skipped. There is no separate file of your schedule: the calendar is the
source of truth for when you're busy. If "good morning" also carries a request
("good morning, add a break at 3"), the request is handled and no offer is made.

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

**Repeating events and groups.** Every event Jarvis creates carries a hidden
tag (in the event's private extended properties: invisible in Google Calendar)
naming the request it came from, so "the chemistry study blocks" means exactly
that set of events, however far ahead they run, instead of a guess from
titles. Two things follow:
- When the days form a regular pattern at one time ("every day at 6:30",
  "Monday, Wednesday and Friday at 4"), Jarvis makes **one repeating event**
  (Google's own recurrence), so it's a single thing to recolor, move or delete in
  Google Calendar too. It says so: "Added ...: 14 events, as one repeating
  event." Recoloring or deleting "them" acts on the series as a whole.
- When a plan adapts (a day moved after Code Ninjas, "after my bus", days left
  out) the times differ, which a repeating event can't express, so it stays
  separate events -- but they share one tag, so recolor/delete "all of them"
  still means the whole plan (`get_group_events`). If a repeating event isn't
  possible (Google refuses, or the calendar's time zone can't be read) it falls
  back to separate events.
- After a restart, "make them red" means the most recent thing Jarvis made.
  With nothing Jarvis-made to refer to, it asks instead of guessing.
- Events made before this (or by hand) have no tag and are matched by title as
  before. Moving a whole repeating event isn't supported yet; moving one
  occurrence is.

**Changing several at once.** "Change the color of each of those chemistry
study blocks to dark green", "make them red" (the events just added), "reset
all my chemistry study blocks to the default color". Words like all / each /
every / those / them mean every match instead of one. Recoloring just happens;
deleting several asks first, and says what it found: "Delete all 6 events (5
Chemistry study block, 1 Chemistry test) over the next 7 days?". If Jarvis asks
"which one?", "all of them" is a valid answer.

**Moving several at once.** "Move all the chemistry study blocks to start at 4
p.m." (also "...at 350 p.m. to start at 4 p.m.", "move them to 4pm", "so they
start at 4") changes the start time of every one: each keeps its own day and its
length, Jarvis asks first, and says how many would land on something already on
your calendar. It can't change the *day* of several at once. A repeating event
is moved through its individual occurrences. The time after "to" / "to start at" is
where they're going, not which events you mean, so "4 p.m." no longer picks
whatever happens to start at 4.

**Changing how long events are.** One event or many ("them", "all the chemistry
study blocks", a repeating event's occurrences). The start never moves:
- set a length: "make my break 45 minutes long", "make it an hour and a half
  long", "the break should be 25 minutes", "change the length of the study
  blocks to an hour"
- add or take away time: "extend the break by 15 minutes", "add 15 minutes to
  my break", "make them 30 minutes longer", "shorten it by 10 minutes",
  "make my break 20 minutes shorter"
- set the end time: "make my break end at 5 pm", "change the end time to 4:30"
Jarvis asks first ("Make Break 45 minutes long, until 03:45 PM?") and warns if
the change would land on something already on your calendar. With no amount ("make
it longer") it asks how much. It won't shrink an event below 5 minutes or make
it longer than a day, or set an end time before the start. These requests are
recognised in Python, so they can't be mistaken for creating a new event.

**Renaming events.** "Rename my break to lunch", "change the title of my
meeting to team sync", "retitle it as ...", and once you're talking about
something, "call it lunch" / "call them chem review". It works on one event or
many ("rename the chemistry study blocks to Chem review"; "all of them" answers
"which one?"), and a repeating event is renamed once, at the series. Only the
part naming the events is used to find them, so the new name can't be mistaken for
the event ("rename my break to friday plan" isn't about Friday). It just happens
(rename it back to undo), and the reply names both titles: "Renamed Break to
Lunch." Asking for something that isn't on the calendar at all ("delete my
nonexistent thing") now says it couldn't find it, instead of listing today's events.

**Things Jarvis can't do, and things it didn't understand.** Inviting people
and setting reminders/alarms/timers aren't supported yet, and
Jarvis says so plainly ("I can't invite people yet."). A calendar-sounding
request it didn't understand ("change my class thing") gets a "say it plainly"
hint with examples instead of a chat answer. Both are decided in Python; the
small chat model used to make things up ("I'll invite Sam to your meeting", "I
don't have access to your calendar"). The chat model's only calendar tool is
now read-only (`get_upcoming_events`), so it can no longer create events from
text it wrote itself. Emoji in replies are stripped before printing and
speaking (a Windows console can't print them).

**Not understanding an answer.** If Jarvis asks "which one?" and what it hears
is neither an answer nor a new request (mostly a misheard or garbled
transcript), it says "Sorry, I didn't catch that" and repeats the options once,
then gives up. A question ("what's on tomorrow") is always a new request. Like everything that edits events, this only looks at today plus the
next 7 days.

Moving, deleting and recoloring look at today plus the next 7 days
(`TARGET_DAYS` in `calendar_manager.py`). If you don't name a day, an event
still ahead today wins over the same title later in the week; name one
("delete tomorrow's break", "the friday one") to pick another. Which-one
questions and confirmations include the day when it isn't today.

**Follow-ups don't need the wake word.** For 30 seconds
(`FOLLOWUP_TIMEOUT` in `main.py`) after any response, you can just keep
talking -- no need to say "Hey Jarvis" again. That covers answering a
question Jarvis just asked ("Move Break to 04:30 PM?" or "Which one did
you mean?") as well as any new, unrelated request. Once 30 seconds pass
with nothing said, the wake word is required again -- and if it was a
pending question specifically, it's dropped. Anything that isn't a clear
yes or no is treated as a new request instead of a confirmation, and an
unclear answer never confirms a delete or move. A yes or no has to be
short (a "no" up to 6 words, a "yes" up to 8): a long sentence that merely
contains "don't" or "sure" is a new request, not an answer.

**"Hey Jarvis, stop".** Say the wake word plus "stop" (or "never mind" /
"quiet") at any time, even while Jarvis is still mid-sentence, and it cuts
the response off right there, drops any pending question, and answers
"Okay, stopping." On Windows this is a real barge-in: while speaking,
Jarvis keeps listening for the wake word through the same small model used
to wake it up, and cuts the audio the instant it's heard (see
`_play_interruptible` in `voice.py`); other platforms fall back to
uninterruptible playback, so "stop" there only takes effect once the
current line finishes. Interrupting with something other than "stop" is
handled the same way as a fresh request -- Jarvis drops what it was saying
and answers the new one instead.

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
- If Jarvis stops listening in the middle of a sentence: the `speech_recognition`
  library's default keeps re-raising its "this is silence" level to 1.5x
  however loudly you're talking, so a softer stretch of a sentence counts as
  silence and the recording ends. `voice.py` measures your room once at
  startup and holds that level fixed (`MIN_ENERGY_THRESHOLD` is its floor),
  and waits `PAUSE_SECONDS` (1.5) of silence before ending a request. Lower
  `PAUSE_SECONDS` if Jarvis feels slow to respond; raise it if you still get
  cut off while thinking mid-sentence.
- Whisper is given a vocabulary hint (`BASE_VOCABULARY` in `voice.py` plus the
  titles on your calendar, refreshed every 30 minutes) so it prefers "study
  blocks" and "chemistry" over soundalikes. Add words there if it keeps
  mishearing something.
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
