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
import io
import os
import tempfile

import edge_tts
import playsound
import speech_recognition as sr
from faster_whisper import WhisperModel

_recognizer = sr.Recognizer()
_whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Skips per-utterance language detection (faster, and stops short clips
# being misdetected as another language). Set to None to auto-detect.
WHISPER_LANGUAGE = "en"

# Try "en-US-GuyNeural" for American, or run `edge-tts --list-voices` to
# browse the full catalog and pick whatever sounds most JARVIS to you.
VOICE = "en-GB-RyanNeural"

# "+0%" is normal speed. Bump this up to talk faster -- "+25%" is a
# noticeable but still natural-sounding speedup, "+50%" starts to feel rushed.
RATE = "+25%"


_calibrated = False


def _calibrate(source):
    """Measure background noise ONCE per run. Doing this on every listen()
    cost half a second each time, during which the start of a short reply
    like "yes" could be swallowed as "background noise". After this, the
    recognizer's dynamic_energy_threshold keeps adapting on its own."""
    global _calibrated
    if not _calibrated:
        _recognizer.adjust_for_ambient_noise(source, duration=1)
        _calibrated = True


def _transcribe(audio):
    """Whisper on the raw wav bytes -- no temp file. The VAD filter and the
    no-speech check are what stop Whisper from inventing text ("Thank you.")
    out of silence or room noise."""
    segments, _ = _whisper_model.transcribe(
        io.BytesIO(audio.get_wav_data()),
        language=WHISPER_LANGUAGE,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(
        segment.text.strip()
        for segment in segments
        if not (segment.no_speech_prob > 0.6 and segment.avg_logprob < -1.0)
    ).strip()


def listen(prompt="Listening... (speak now)", show_status=True, timeout=None):
    """Record from the mic until you stop talking, then transcribe it.

    timeout: seconds to wait for speech to START before giving up and
    returning "" (None = wait forever).
    """
    with sr.Microphone() as source:
        if show_status:
            print(prompt)
        _calibrate(source)
        try:
            audio = _recognizer.listen(source, timeout=timeout)
        except sr.WaitTimeoutError:
            return ""

    if show_status:
        print("Transcribing...")
    return _transcribe(audio)


async def _speak_async(text):
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        await communicate.save(temp_path)
        playsound.playsound(temp_path)
    finally:
        os.remove(temp_path)


def speak(text):
    """Speak text out loud. Requires internet (edge-tts is a cloud voice)."""
    print(f"Jarvis: {text}")
    asyncio.run(_speak_async(text))
