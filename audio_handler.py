import speech_recognition as sr
import os
import sys
import ctypes

# Suppress ALSA/PyAudio "Unknown PCM" warnings that spam the terminal
# (completely harmless — PyAudio is just probing unavailable virtual devices)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so")
    _asound.snd_lib_error_set_handler(ctypes.c_void_p(None))
except Exception:
    pass


class AudioHandler:
    def __init__(self, wake_word="hey john"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()

        # Tuning:
        #  - pause_threshold: seconds of silence before the phrase is considered done
        #  - energy_threshold: mic sensitivity (auto-calibrated below)
        #  - dynamic_energy_threshold: auto-adjusts for background noise
        self.recognizer.pause_threshold = 1.2
        self.recognizer.dynamic_energy_threshold = True

        self.valid_wake_words = [
            self.wake_word,
            "hey john", "hey jon",   "a john",   "he john",
            "hey joan", "hey joe",   "hey jaw",   "hey jaw.",
            "hey john.", "hey john?", "hey john!", "hey, john",
        ]

        # Calibrate for ambient noise using the default system mic
        print("[Audio] Calibrating microphone for ambient noise...")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print(f"[Audio] Ready. Energy threshold: {int(self.recognizer.energy_threshold)}")

    # ─────────────────────────────────────────────────────────────────────
    def _listen_once(self, timeout=None, phrase_limit=10) -> str:
        """
        Opens the default system microphone, listens for one phrase,
        sends it to Google STT, and returns the transcript (lowercase).
        Returns "" on failure or silence.
        """
        try:
            with sr.Microphone() as source:
                print("\n(Listening...)", end="", flush=True)
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,           # how long to wait for speech to start
                    phrase_time_limit=phrase_limit,  # max seconds per phrase
                )
        except sr.WaitTimeoutError:
            return ""

        print("\r[Recognizing...]   ", end="", flush=True)
        try:
            text = self.recognizer.recognize_google(audio).lower()
            print(f"\r[Heard]: \"{text}\"" + " " * 30)
            return text
        except sr.UnknownValueError:
            print("\r[Heard]: (could not understand)" + " " * 20)
            return ""
        except sr.RequestError as e:
            print(f"\r[Google STT error]: {e}")
            return ""

    # ─────────────────────────────────────────────────────────────────────
    def listen_for_wake_word(self) -> bool:
        """Blocks until a wake word phrase is detected. Returns True."""
        print(f"\n[Waiting for wake word: '{self.wake_word}']")
        while True:
            text = self._listen_once(timeout=None, phrase_limit=6)
            if not text:
                continue
            for ww in self.valid_wake_words:
                if ww in text:
                    print(f"[Wake word matched: '{ww}']")
                    return True
            print(f"[No wake word in: \"{text}\"]")

    # ─────────────────────────────────────────────────────────────────────
    def listen_for_query(self) -> str:
        """Listens for a user query after activation. Returns transcript."""
        print("Listening for your query...")
        text = self._listen_once(timeout=5, phrase_limit=15)
        if not text:
            print("(Nothing heard)")
        return text
