import speech_recognition as sr
import os
import ctypes
import threading

# Suppress ALSA/PyAudio "Unknown PCM" warnings that spam the terminal
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so")
    _asound.snd_lib_error_set_handler(ctypes.c_void_p(None))
except Exception:
    pass


class AudioHandler:
    def __init__(self, wake_word="hey john"):
        self.wake_word = wake_word.lower()
        self.recognizer = sr.Recognizer()

        self.recognizer.pause_threshold = 1.2
        self.recognizer.dynamic_energy_threshold = True

        # Broad set of phonetically similar alternatives
        self.valid_wake_words = [
            "hey john", "hey jon",  "hey jan",  "hey joan",
            "a john",   "he john",  "hey joe",  "hey jaw",
            "hey june", "hey gin",  "hey jean", "hey yon",
            "hey john.", "hey john?", "hey john!", "hey, john",
        ]

        # Set from outside to abort listen_for_wake_word immediately
        self.cancel_event = threading.Event()

        # Mutex: only one _listen_once() may hold the mic at a time.
        # Acquire before listen_for_query() to guarantee the wake-word
        # sub-thread has fully released the mic first.
        self.mic_lock = threading.Lock()

        print("[Audio] Calibrating microphone for ambient noise...")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print(f"[Audio] Ready. Energy threshold: {int(self.recognizer.energy_threshold)}")

    # ─────────────────────────────────────────────────────────────────────
    def _listen_once(self, timeout=None, phrase_limit=10) -> str:
        """
        Listen for one phrase, transcribe with Google STT, return lowercase text.
        Holds mic_lock for its entire duration so no two listeners can overlap.
        Returns "" on silence/failure.
        """
        try:
            with self.mic_lock:
                with sr.Microphone() as source:
                    print("\n(Listening...)", end="", flush=True)
                    audio = self.recognizer.listen(
                        source,
                        timeout=timeout,
                        phrase_time_limit=phrase_limit,
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
        """
        Listen in short chunks until a wake word is heard OR cancel_event is set.
        Returns True on match, False if cancelled.
        Each chunk runs in a sub-thread so cancel_event is checked between chunks.
        """
        self.cancel_event.clear()
        print(f"[Waiting for wake word: '{self.wake_word}']")

        while not self.cancel_event.is_set():
            result = [""]
            done   = threading.Event()

            def _do_listen():
                result[0] = self._listen_once(timeout=3, phrase_limit=6)
                done.set()

            t = threading.Thread(target=_do_listen, daemon=True)
            t.start()

            # Wait for the listen chunk to finish, checking cancel every 100 ms
            while not done.is_set():
                if self.cancel_event.is_set():
                    print("[Audio] Wake word listener cancelled.")
                    return False
                done.wait(timeout=0.1)

            text = result[0]
            if not text:
                continue

            for ww in self.valid_wake_words:
                if ww in text:
                    print(f"[Wake word matched: '{ww}' in \"{text}\"]")
                    return True

            print(f"[No wake word in: \"{text}\"]")

        return False

    # ─────────────────────────────────────────────────────────────────────
    def listen_for_query(self) -> str:
        """Listen for a user query after activation. Returns transcript."""
        print("Listening for your query...")
        text = self._listen_once(timeout=5, phrase_limit=15)
        if not text:
            print("(Nothing heard)")
        return text
