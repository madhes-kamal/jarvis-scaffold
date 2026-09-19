"""
voice.py

Speech-to-text and text-to-speech.

- STT: faster-whisper running on your machine (free, local)
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
from faster_whisper import WhisperModel

_recognizer = sr.Recognizer()
_whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Try "en-US-GuyNeural" for American, or run `edge-tts --list-voices` to
# browse the full catalog and pick whatever sounds most JARVIS to you.
VOICE = "en-GB-RyanNeural"

# "+0%" is normal speed. Bump this up to talk faster -- "+25%" is a
# noticeable but still natural-sounding speedup, "+50%" starts to feel rushed.
RATE = "+25%"


def listen(prompt="Listening... (speak now)", show_status=True):
    """Record from the mic until you stop talking, then transcribe it."""
    with sr.Microphone() as source:
        if show_status:
            print(prompt)
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = _recognizer.listen(source)

    if show_status:
        print("Transcribing...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio.get_wav_data())
        temp_path = f.name

    segments, _ = _whisper_model.transcribe(temp_path)
    return " ".join(segment.text for segment in segments).strip()


async def _speak_async(text):
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    await communicate.save(temp_path)
    playsound.playsound(temp_path)
    os.remove(temp_path)


def speak(text):
    """Speak text out loud. Requires internet (edge-tts is a cloud voice)."""
    print(f"Jarvis: {text}")
    asyncio.run(_speak_async(text))
