"""
voice.py

Speech-to-text and text-to-speech, both free and fully local:

- STT: OpenAI's Whisper model running on your machine (no API, no
  internet needed once the model downloads the first time)
- TTS: pyttsx3, which uses your operating system's built-in voices
  (also fully offline)
"""

import speech_recognition as sr
import pyttsx3

_recognizer = sr.Recognizer()
_tts_engine = pyttsx3.init()


def listen(prompt="Listening... (speak now)"):
    """Record from the mic until you stop talking, then transcribe it."""
    with sr.Microphone() as source:
        print(prompt)
        _recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = _recognizer.listen(source)

    print("Transcribing...")
    # "base" is a good speed/accuracy tradeoff to start with. Whisper
    # model sizes: tiny, base, small, medium, large -- bigger is more
    # accurate but slower.
    text = _recognizer.recognize_whisper(audio, model="base")
    return text


def speak(text):
    """Speak text out loud."""
    print(f"Jarvis: {text}")
    _tts_engine.say(text)
    _tts_engine.runAndWait()
