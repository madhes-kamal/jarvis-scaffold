"""
voice.py

Speech-to-text and text-to-speech.

- Wake word: openWakeWord's pretrained "hey jarvis" model, running on the
  raw mic audio (tiny, local, no transcription involved -- Whisper only
  runs once you've actually woken Jarvis up)
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
import atexit
import io
import os
import tempfile

import edge_tts
import numpy as np
import playsound
import speech_recognition as sr
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeModel

_recognizer = sr.Recognizer()
_whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# openWakeWord wants 16 kHz mono int16 audio in 80 ms (1280-sample) frames.
SAMPLE_RATE = 16000
CHUNK = 1280

# Pretrained model name (ships with openWakeWord) and how confident it must
# be, 0-1. Lower it if Jarvis misses you; raise it if it wakes on its own.
# A wake prints "[wake word heard, score ...]" so you can see where you sit.
WAKE_MODEL = "hey_jarvis"
WAKE_THRESHOLD = 0.5

# The ONNX backend needs only onnxruntime (the default tflite one doesn't
# install cleanly on every platform).
_wake_model = WakeModel(wakeword_models=[WAKE_MODEL], inference_framework="onnx")

# Skips per-utterance language detection (faster, and stops short clips
# being misdetected as another language). Set to None to auto-detect.
WHISPER_LANGUAGE = "en"

# Try "en-US-GuyNeural" for American, or run `edge-tts --list-voices` to
# browse the full catalog and pick whatever sounds most JARVIS to you.
VOICE = "en-GB-RyanNeural"

# "+0%" is normal speed. Bump this up to talk faster -- "+25%" is a
# noticeable but still natural-sounding speedup, "+50%" starts to feel rushed.
RATE = "+25%"


_mic = None
_source = None


def _close_mic():
    if _mic is not None:
        try:
            _mic.__exit__(None, None, None)
        except Exception:
            pass


def _get_source():
    """The ONE microphone stream the whole app shares, opened on first use.

    Wake-word detection and command recording read from the same stream, so
    nothing is lost in a gap between "Hey Jarvis" and the start of the
    request (closing and reopening the mic in between clipped the first
    words of "Hey Jarvis, what's on my calendar"). Background noise is
    measured once, here; after that the recognizer's
    dynamic_energy_threshold keeps adapting on its own."""
    global _mic, _source
    if _source is None:
        _mic = sr.Microphone(sample_rate=SAMPLE_RATE, chunk_size=CHUNK)
        _source = _mic.__enter__()
        atexit.register(_close_mic)
        _recognizer.adjust_for_ambient_noise(_source, duration=1)
    return _source


def _flush_input():
    """Discard audio that piled up while nobody was reading the stream --
    mainly Jarvis's own voice from the speakers, which would otherwise be
    heard as the user's answer."""
    if _source is None:
        return
    stream = _source.stream.pyaudio_stream
    try:
        while (available := stream.get_read_available()) > 0:
            stream.read(available, exception_on_overflow=False)
    except OSError:
        pass


def wait_for_wake_word():
    """Block until "Hey Jarvis" is heard. Runs the wake model on every
    80 ms of mic audio; no speech-to-text is involved."""
    source = _get_source()
    _flush_input()
    _wake_model.reset()
    while True:
        frame = np.frombuffer(source.stream.read(CHUNK), dtype=np.int16)
        score = max(_wake_model.predict(frame).values())
        if score >= WAKE_THRESHOLD:
            print(f"  [wake word heard, score {score:.2f}]")
            return


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
    source = _get_source()
    if show_status:
        print(prompt)
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
    _flush_input()
