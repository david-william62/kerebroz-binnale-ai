import os
import threading
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler
from hotkey_handler import HotkeyHandler

# ─── Phrases that end the conversation session ────────────────────────────────
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
        print("Error: GEMINI_API_KEY environment variable not found.")
        print("Please create a .env file and add your GEMINI_API_KEY.")
        return

    try:
        audio  = AudioHandler(wake_word="hey john")
        llm    = LLMHandler()
        tts    = TTSHandler(voice="af_heart")
        hotkey = HotkeyHandler()   # reads WAKE_KEY / SLEEP_KEY from .env
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return

    # Start hotkey listener in background
    hotkey.start()

    print("\n┌─────────────────────────────────────────────┐")
    print("│            John AI Assistant Ready           │")
    print("├─────────────────────────────────────────────┤")
    print("│  Say  'Hey John'     OR press F9   to wake  │")
    print("│  Say  'Goodbye John' OR press F10  to sleep │")
    print("│  Press Ctrl+C to quit                       │")
    print("└─────────────────────────────────────────────┘\n")

    # ── Shared: wake trigger (set by wake word OR hotkey) ──────────────────
    wake_event = threading.Event()

    def wake_word_thread():
        """Continuously listen for wake word and fire wake_event."""
        while True:
            audio.listen_for_wake_word()
            wake_event.set()

    ww_thread = threading.Thread(target=wake_word_thread, daemon=True, name="WakeWord")
    ww_thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────
    try:
        while True:
            # ── Standby: wait for wake word  OR  F9 hotkey ────────────────
            print("\n[Standby — say 'Hey John' or press F9 to begin]")

            while True:
                # Check hotkey first (non-blocking)
                if hotkey.wake_pressed.is_set():
                    hotkey.clear_wake()
                    print("[Hotkey] Waking up via key press.")
                    wake_event.set()

                if wake_event.is_set():
                    wake_event.clear()
                    break

                # Small sleep to avoid busy-wait
                threading.Event().wait(0.1)

            # ── Greet ─────────────────────────────────────────────────────
            tts.process_llm_stream(iter(["Yes? How can I help you?"]))

            # ── Conversation loop ──────────────────────────────────────────
            print("\n[Session started — say 'Goodbye John' or press F10 to end]\n")

            while True:
                # Check sleep hotkey before listening
                if hotkey.sleep_pressed.is_set():
                    hotkey.clear_sleep()
                    print("[Hotkey] Sleep key pressed — ending session.")
                    tts.process_llm_stream(iter([FAREWELL_RESPONSE]))
                    print("\n[Session ended — returning to standby]\n")
                    break

                query = audio.listen_for_query()

                if not query:
                    # Check sleep hotkey again after timeout
                    if hotkey.sleep_pressed.is_set():
                        hotkey.clear_sleep()
                        tts.process_llm_stream(iter([FAREWELL_RESPONSE]))
                        print("\n[Session ended — returning to standby]\n")
                        break
                    print("(Didn't catch that — still listening...)")
                    continue

                # Farewell via voice
                if _is_farewell(query):
                    print(f"[Farewell detected: '{query}']")
                    tts.process_llm_stream(iter([FAREWELL_RESPONSE]))
                    print("\n[Session ended — returning to standby]\n")
                    break

                # Normal query → LLM → TTS
                response_stream = llm.generate_response_stream(query)
                tts.process_llm_stream(response_stream)

    except KeyboardInterrupt:
        print("\nExiting John AI Assistant. Goodbye!")
    finally:
        hotkey.stop()


if __name__ == "__main__":
    main()
