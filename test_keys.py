import time
from pynput import keyboard

def on_press(key):
    print(f"Pressed: {key}")

listener = keyboard.Listener(on_press=on_press)
listener.daemon = True
listener.start()

time.sleep(5)
