"""Local speech-to-text, Jarvis wake-word detection, and Piper TTS."""

import os
import re
import tempfile
import wave
import winsound

from piper import PiperVoice
import speech_recognition as sr

_recognizer = sr.Recognizer()
WAKE_WORD = "jarvis"
PIPER_MODEL = os.environ.get(
    "PIPER_MODEL", os.path.join("models", "en_GB-alan-medium.onnx")
)
_piper_voice = None

def _transcribe(prompt):
    """Capture one utterance and transcribe it with the local Whisper model."""
    with sr.Microphone() as source:
        print(prompt)
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = _recognizer.listen(source)

    print("Transcribing...")
    text = _recognizer.recognize_whisper(audio, model="base")
    return text


def listen(prompt="Listening... (speak now)"):
    """Record from the mic until you stop talking, then transcribe it."""
    return _transcribe(prompt)


def listen_for_wake_word():
    """Wait for an utterance containing Jarvis and return its command.

    The command after the wake word is returned immediately, so both
    "Jarvis, what is on my calendar?" and a standalone "Jarvis" work.
    """
    text = _transcribe("Listening for 'Jarvis'...").strip()
    match = re.search(rf"\b{re.escape(WAKE_WORD)}\b", text, re.IGNORECASE)
    if not match:
        return None
    return text[match.end():].lstrip(" ,.!?")


def _get_piper_voice():
    global _piper_voice
    if _piper_voice is None:
        if not os.path.exists(PIPER_MODEL):
            raise FileNotFoundError(
                f"Piper model not found at {PIPER_MODEL!r}. "
                "Run the model download command from README.md."
            )
        _piper_voice = PiperVoice.load(PIPER_MODEL)
    return _piper_voice


def speak(text):
    """Speak text with the local Piper voice."""
    print(f"Jarvis: {text}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        with wave.open(temp_path, "wb") as wav_file:
            _get_piper_voice().synthesize_wav(text, wav_file)
        winsound.PlaySound(temp_path, winsound.SND_FILENAME)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
