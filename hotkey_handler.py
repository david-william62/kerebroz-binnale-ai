"""
hotkey_handler.py
─────────────────
Listens for configurable keys in a background thread using raw terminal
input — no display server (X11/Wayland) required.

Default keys  (override in .env):
  WAKE_KEY  = space  →  activate John instantly
  SLEEP_KEY = enter  →  end session immediately

Supported key names:
  space, enter, esc, tab, backspace,
  f1-f12, or any single character like 'a', 'z', '/'
"""

import os
import sys
import select
import signal
import termios
import threading


# ── Map friendly names → raw bytes the terminal sends ────────────────────────
_SPECIAL_KEYS = {
    "space":     " ",
    "enter":     "\r",
    "newline":   "\n",
    "esc":       "\x1b",
    "escape":    "\x1b",
    "tab":       "\t",
    "backspace": "\x7f",
    # Standard VT/ANSI escape codes for F-keys
    "f1":  "\x1bOP",  "f2":  "\x1bOQ",  "f3":  "\x1bOR",  "f4":  "\x1bOS",
    "f5":  "\x1b[15~","f6":  "\x1b[17~","f7":  "\x1b[18~","f8":  "\x1b[19~",
    "f9":  "\x1b[20~","f10": "\x1b[21~","f11": "\x1b[23~","f12": "\x1b[24~",
}


def _parse_key(key_str: str) -> str:
    key_str = key_str.strip().lower()
    if key_str in _SPECIAL_KEYS:
        return _SPECIAL_KEYS[key_str]
    if len(key_str) == 1:
        return key_str
    raise ValueError(
        f"Unknown key: '{key_str}'. Use: space, enter, esc, tab, f1-f12, or a single character."
    )


def _set_input_raw(fd: int) -> list:
    """
    Set terminal INPUT to raw mode WITHOUT touching output flags.

    tty.setraw() also clears OPOST which kills \\n→\\r\\n translation,
    causing the staircase effect in every other thread's print().
    Here we only touch input/local flags so output stays normal.
    """
    attrs = termios.tcgetattr(fd)
    saved = termios.tcgetattr(fd)

    # Input flags: disable flow ctrl, CR translation, etc.
    attrs[0] &= ~(termios.IXON | termios.ICRNL | termios.BRKINT
                  | termios.ISTRIP | termios.INPCK)
    # Output flags (attrs[1]): *** NOT TOUCHED *** — keeps OPOST/ONLCR intact
    # Local flags: disable line-buffering, echo, signals
    attrs[3] &= ~(termios.ICANON | termios.ECHO
                  | termios.ECHOE | termios.IEXTEN | termios.ISIG)
    # Fire on every single byte, no timer
    attrs[6][termios.VMIN]  = 1
    attrs[6][termios.VTIME] = 0

    termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
    return saved


def _read_key(fd: int) -> str:
    """Read one keypress (possibly a multi-byte escape sequence) from fd."""
    ch = os.read(fd, 1).decode("utf-8", errors="replace")
    if ch == "\x1b":
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            rest = os.read(fd, 8).decode("utf-8", errors="replace")
            return ch + rest
    return ch


class HotkeyHandler:
    def __init__(self, wake_key: str = None, sleep_key: str = None):
        wake_str  = wake_key  or os.getenv("WAKE_KEY",  "space")
        sleep_str = sleep_key or os.getenv("SLEEP_KEY", "enter")

        self._wake_key  = _parse_key(wake_str)
        self._sleep_key = _parse_key(sleep_str)

        self.wake_pressed  = threading.Event()
        self.sleep_pressed = threading.Event()

        self._stop_event = threading.Event()
        self._thread     = None

        print(f"[Hotkeys] Wake = {wake_str.upper()}  |  Sleep = {sleep_str.upper()}")

    # ─────────────────────────────────────────────────────────────────────────
    def _listen_loop(self):
        """
        Open /dev/tty as a private fd, set INPUT-ONLY raw mode once, poll forever.

        Why /dev/tty instead of sys.stdin:
          • select() on stdin in cooked mode only fires AFTER Enter — individual
            keys like Space are line-buffered and never seen.
          • /dev/tty is the controlling terminal, independent of Python's stdin.
          • With ICANON disabled, select() fires on every single keypress. ✓
          • Output flags are untouched → no staircase in other threads' prints. ✓
        """
        try:
            fd = os.open("/dev/tty", os.O_RDONLY)
        except OSError:
            print("[Hotkeys] Cannot open /dev/tty — hotkeys disabled.")
            return

        saved = _set_input_raw(fd)
        try:
            while not self._stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.1)
                if not r:
                    continue

                key = _read_key(fd)

                if key == "\x03":           # Ctrl+C passthrough
                    os.kill(os.getpid(), signal.SIGINT)
                    break

                if key == self._wake_key:
                    print("[Hotkey] Wake key pressed!")
                    self.wake_pressed.set()
                elif key == self._sleep_key:
                    print("[Hotkey] Sleep key pressed!")
                    self.sleep_pressed.set()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            os.close(fd)

    # ─────────────────────────────────────────────────────────────────────────
    def start(self):
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="HotkeyListener"
        )
        self._thread.start()
        print("[Hotkeys] Listener started (/dev/tty raw-input mode).")

    def stop(self):
        self._stop_event.set()

    def clear_wake(self):
        self.wake_pressed.clear()

    def clear_sleep(self):
        self.sleep_pressed.clear()
