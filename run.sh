#!/bin/bash
# Launcher for John AI Assistant

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

# Redirect stderr through a filter that drops harmless ALSA/PyAudio noise
exec python "$SCRIPT_DIR/assistant.py" "$@" 2> >(grep -v "ALSA lib\|Unknown PCM\|snd_func_refer\|_snd_config\|snd_config_expand\|pcm_oss\|pcm_a52\|pcm_usb" >&2)
