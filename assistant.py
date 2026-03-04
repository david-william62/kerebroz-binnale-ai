import os
import threading
from dotenv import load_dotenv

load_dotenv()

from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler
from hotkey_handler import HotkeyHandler

# ─── Farewell phrases ─────────────────────────────────────────────────────────
FAREWELL_PHRASES = [
    "goodbye john", "goodbye, john", "bye john", "bye, john",
    "see you john", "see you later john", "that's all john",
    "thank you john", "thanks john", "stop john",
]
FAREWELL_RESPONSE = "Goodbye! It was a pleasure talking with you. Call me anytime."


def _is_farewell(text: str) -> bool:
    text = text.lower().strip()
    return any(phrase in text for phrase in FAREWELL_PHRASES)


def main():
    print("Initializing John AI Assistant...")

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found. Add it to your .env file.")
        return

    try:
        audio  = AudioHandler(wake_word="hey john")
        llm    = LLMHandler()
        tts    = TTSHandler(voice="af_heart")
        hotkey = HotkeyHandler()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    hotkey.start()

    print("\n┌─────────────────────────────────────────────────┐")
    print("│            John AI Assistant Ready             │")
    print("├─────────────────────────────────────────────────┤")
    print("│  Say  'Hey John'     OR press SPACE  to wake    │")
    print("│  Say  'Goodbye John' OR press ENTER  to sleep   │")
    print("│  Press Ctrl+C to quit                           │")
    print("└─────────────────────────────────────────────────┘\n")

    # ── wake_event: fired by wake word OR wake key ────────────────────────────
    wake_event = threading.Event()

    # ── Wake word thread: runs in background, fires wake_event on match ───────
    def wake_word_thread():
        while True:
            matched = audio.listen_for_wake_word()   # False when cancelled
            if matched:
                wake_event.set()

    threading.Thread(target=wake_word_thread, daemon=True, name="WakeWord").start()

    # ── Hotkey watcher: space pressed → cancel mic instantly → set wake_event ─
    def hotkey_watcher():
        while True:
            hotkey.wake_pressed.wait()       # zero-CPU block
            hotkey.clear_wake()
            print("[Hotkey] Wake key — interrupting mic listener...")
            audio.cancel_event.set()         # abort listen_for_wake_word NOW
            wake_event.set()                 # unblock standby immediately

    threading.Thread(target=hotkey_watcher, daemon=True, name="HotkeyWatcher").start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop — two states: STANDBY and SESSION
    # ─────────────────────────────────────────────────────────────────────────
    try:
        while True:

            # ══════════════════════════════════════════════════════════════════
            #  STANDBY  — mic active, both wake word and SPACE watched
            # ══════════════════════════════════════════════════════════════════
            hotkey.clear_wake()
            hotkey.clear_sleep()
            audio.cancel_event.clear()       # let the wake-word thread run
            wake_event.clear()

            print("\n[Standby — say 'Hey John' or press SPACE to begin]")
            wake_event.wait()                # block until woken by either path

            # Stop mic immediately; wait for any in-progress listen to release it
            audio.cancel_event.set()
            with audio.mic_lock:
                pass                         # ensures mic is fully free before query mode

            # ══════════════════════════════════════════════════════════════════
            #  SESSION  — query mode, ENTER key ends the session
            # ══════════════════════════════════════════════════════════════════
            hotkey.clear_sleep()
            tts.process_llm_stream(iter(["Yes? How can I help you?"]))
            print("\n[Session — say 'Goodbye John' or press ENTER to end]\n")

            while True:
                # Check sleep key first (immediate exit, no waiting for mic)
                if hotkey.sleep_pressed.is_set():
                    hotkey.clear_sleep()
                    print("[Sleep key] Ending session.")
                    tts.process_llm_stream(iter([FAREWELL_RESPONSE]))
                    print("\n[Session ended — returning to standby]\n")
                    break

                query = audio.listen_for_query()

                if not query:
                    if hotkey.sleep_pressed.is_set():
                        hotkey.clear_sleep()
                        tts.process_llm_stream(iter([FAREWELL_RESPONSE]))
                        print("\n[Session ended — returning to standby]\n")
                        break
                    print("(Didn't catch that — still listening...)")
                    continue

                if _is_farewell(query):
                    print(f"[Farewell detected: '{query}']")
                    tts.process_llm_stream(iter([FAREWELL_RESPONSE]))
                    print("\n[Session ended — returning to standby]\n")
                    break

                response_stream = llm.generate_response_stream(query)
                tts.process_llm_stream(response_stream)

    except KeyboardInterrupt:
        print("\nExiting John AI Assistant. Goodbye!")
    finally:
        hotkey.stop()


if __name__ == "__main__":
    main()
