"""
hotkey_handler.py
─────────────────
Listens for configurable hotkeys in a background thread.
Pressing the wake key triggers John even without saying "Hey John".
Pressing the sleep key ends the current conversation session immediately.

Default keys:
  WAKE_KEY  = F9   →  activate John
  SLEEP_KEY = F10  →  end current session / put John to sleep

To change keys, edit WAKE_KEY and SLEEP_KEY below,
or set them in your .env file:
  WAKE_KEY=f8
  SLEEP_KEY=f12

Valid key names:  f1-f12, space, enter, esc, tab,
                  ctrl, alt, shift, caps_lock,
                  or any single character like 'a', 'z', '/'
"""

import os
import threading
from pynput import keyboard as pynput_keyboard


def _parse_key(key_str: str):
    """Convert a string like 'f9' or 'space' to a pynput Key or KeyCode."""
    key_str = key_str.strip().lower()
    # Try as a special key (F1-F12, space, enter, etc.)
    try:
        return getattr(pynput_keyboard.Key, key_str)
    except AttributeError:
        pass
    # Fall back to a character key
    if len(key_str) == 1:
        return pynput_keyboard.KeyCode.from_char(key_str)
    raise ValueError(f"Unknown key: '{key_str}'. Use names like f9, f10, space, enter, or a single character.")


class HotkeyHandler:
    def __init__(self, wake_key: str = None, sleep_key: str = None):
        # Read from env, fall back to defaults
        wake_str  = wake_key  or os.getenv("WAKE_KEY",  "f9")
        sleep_str = sleep_key or os.getenv("SLEEP_KEY", "f10")

        self._wake_key  = _parse_key(wake_str)
        self._sleep_key = _parse_key(sleep_str)

        # Thread-safe events that assistant.py watches
        self.wake_pressed  = threading.Event()
        self.sleep_pressed = threading.Event()

        self._listener = None
        self._thread   = None

        print(f"[Hotkeys] Wake = {wake_str.upper()}  |  Sleep = {sleep_str.upper()}")

    # ─────────────────────────────────────────────────────────────────────
    def _on_press(self, key):
        if key == self._wake_key:
            print(f"\n[Hotkey] Wake key pressed!")
            self.wake_pressed.set()
        elif key == self._sleep_key:
            print(f"\n[Hotkey] Sleep key pressed!")
            self.sleep_pressed.set()

    # ─────────────────────────────────────────────────────────────────────
    def start(self):
        """Start listening for hotkeys in a background daemon thread."""
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()
        print("[Hotkeys] Listener started in background.")

    def stop(self):
        try:
            if self._listener:
                self._listener.stop()
        except AttributeError:
            pass  # Listener stopped before fully initializing — safe to ignore

    # ─────────────────────────────────────────────────────────────────────
    def clear_wake(self):
        self.wake_pressed.clear()

    def clear_sleep(self):
        self.sleep_pressed.clear()
