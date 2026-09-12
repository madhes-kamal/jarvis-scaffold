# JARVIS Scaffold — Terminal Morning Brief

The first building block: a script that reads today's Google Calendar
events and asks a local LLM (via Ollama) to turn them into a casual,
spoken-style summary. No voice, no hardware, no paid API — this just
proves the core loop works, for free.

## What's here

- `calendar_service.py` — Google Calendar OAuth + fetching today's events
- `brief_generator.py` — sends events to a local Ollama model, gets back a casual summary
- `main.py` — ties the two together; this is the one you run
- `requirements.txt` — Python dependencies

## Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Google Calendar API access

1. Go to https://console.cloud.google.com/ and create a new project (or use an existing one).
2. Go to "APIs & Services" → "Library", search for "Google Calendar API", and enable it.
3. Go to "APIs & Services" → "Credentials" → "Create Credentials" → "OAuth client ID".
4. If prompted, configure the OAuth consent screen first: choose "External", fill in the required basics (app name, your email), and add your own Google account as a "test user". You don't need to publish the app.
5. For the OAuth client ID, choose **Application type: Desktop app**.
6. Download the resulting JSON file, rename it `credentials.json`, and put it in this folder.

### 3. Ollama (free, local, no API key)

1. Download and install from https://ollama.com
2. Pull the model: `ollama pull qwen2.5:1.5b` (try `qwen2.5:7b` instead if you want better quality — your laptop can handle more than the Pi will)
3. That's it — Ollama runs a local server in the background automatically once installed

### 4. Run it

```bash
python main.py
```

First run opens a browser window asking you to log into Google and approve
calendar access. Since the OAuth consent screen isn't "verified" (fine for
a personal project only you use), you'll likely see an "unverified app"
warning — click "Advanced" → "Go to [app name] (unsafe)". After approving,
a `token.json` file is saved so future runs skip the login step.

## Debugging tips (since you're newer to backend)

- Run `python calendar_service.py` on its own first — it just prints your
  events, no LLM involved. If this fails, the problem is Google auth, not
  Claude.
- Run `python brief_generator.py` on its own next — it uses fake calendar
  data, no Google auth involved. If this fails, check that Ollama is
  actually running (`ollama list` should show `qwen2.5:1.5b`) before
  suspecting your code.
- Only run `python main.py` once both of the above work individually —
  it's just those two pieces combined.

## What to try next

- Swap the prompt in `brief_generator.py` to change the tone/personality
- Add a `--speak` flag that pipes the output through a text-to-speech
  library instead of printing it
- Start sketching the focus state machine as its own module once this
  feels solid
