"""
voice.py

Speech-to-text and text-to-speech.

- STT: OpenAI's Whisper model running on your machine (free, local)
- TTS: edge-tts, using Microsoft's neural voices (free, no API key,
  MUCH more natural than pyttsx3 -- the one catch is it needs internet,
  since it's a cloud voice service Microsoft happens to offer for free)

pyttsx3 was dropped: it has a known bug, especially on Windows, where
the engine doesn't reliably reset after speaking once, so a second
speak() call silently does nothing. Switching engines was simpler than
working around it.
"""

import asyncio
import os
import tempfile

import edge_tts
import playsound
import speech_recognition as sr

_recognizer = sr.Recognizer()

# Try "en-US-GuyNeural" for American, or run `edge-tts --list-voices` to
# browse the full catalog and pick whatever sounds most JARVIS to you.
VOICE = "en-GB-RyanNeural"


def listen(prompt="Listening... (speak now)"):
    """Record from the mic until you stop talking, then transcribe it."""
    with sr.Microphone() as source:
        print(prompt)
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = _recognizer.listen(source)

    print("Transcribing...")
    text = _recognizer.recognize_whisper(audio, model="base")
    return text


async def _speak_async(text):
    communicate = edge_tts.Communicate(text, voice=VOICE)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    await communicate.save(temp_path)
    playsound.playsound(temp_path)
    os.remove(temp_path)


def speak(text):
    """Speak text out loud. Requires internet (edge-tts is a cloud voice)."""
    print(f"Jarvis: {text}")
    asyncio.run(_speak_async(text))
