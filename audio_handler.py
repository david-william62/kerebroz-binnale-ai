import sounddevice as sd
import numpy as np
import os
import webrtcvad
import queue
from faster_whisper import WhisperModel

class AudioHandler:
    def __init__(self, wake_word="hey john"):
        self.wake_word = wake_word.lower()
        
        # Audio configuration
        self.RATE = 16000
        self.CHANNELS = 1
        self.CHUNK = 480  # 30ms chunk at 16kHz
        
        print("Loading Whisper model...")
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
             
        # VAD helps us detect when someone is speaking vs silence
        self.vad = webrtcvad.Vad(3) # Aggressiveness 3 is highest
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time, status):
        """This is called for each audio block."""
        if status:
            pass # Ignore overflows for now to avoid console spam
        # We need 16-bit PCM for VAD
        audio_data = (indata[:, 0] * 32767).astype(np.int16)
        self.audio_queue.put(audio_data)

    def _record_until_silence(self, max_silence_duration=1.0):
        """Records audio until silence is detected, and transcribes locally."""
        frames = []
        silence_frames = 0
        is_speaking = False
        max_silence_chunks = int(max_silence_duration * self.RATE / self.CHUNK)
        
        # Clear queue before starting
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        with sd.InputStream(samplerate=self.RATE, channels=self.CHANNELS, 
                            blocksize=self.CHUNK, callback=self._audio_callback):
            while True:
                chunk = self.audio_queue.get()
                
                # Check for speech in this chunk
                is_speech = self.vad.is_speech(chunk.tobytes(), self.RATE)
                
                if is_speech:
                    if not is_speaking:
                        print("\n(Listening...)", end="", flush=True)
                    is_speaking = True
                    silence_frames = 0
                    frames.append(chunk)

                elif is_speaking:
                    silence_frames += 1
                    frames.append(chunk)
                    
                    if silence_frames > max_silence_chunks:
                        break # Stopped speaking
                
                # Keep a tiny rolling window of audio before speaking starts so we don't clip the first word
                if not is_speaking:
                    frames.append(chunk)
                    if len(frames) > int(0.5 * self.RATE / self.CHUNK): # Keep 0.5 seconds
                        frames.pop(0)

        print("") # Clear the line after speaking
        
        if not frames:
            return ""
            
        audio_data = np.concatenate(frames)
        
        # Check if the audio is mostly just silence/noise even if VAD triggered
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 500:
             return ""

        # Convert to float32 for Whisper (-1.0 to 1.0)
        audio_data_f32 = audio_data.astype(np.float32) / 32768.0

        print("\r[Whisper Thinking...]", end="", flush=True)
        # Transcribe with faster-whisper
        segments, info = self.model.transcribe(audio_data_f32, beam_size=1)
        
        text = "".join([segment.text for segment in segments]).strip().lower()
        if text:
            print(f"\r[Whisper Heard]: {text}" + " " * 20)
        return text

    def listen_for_wake_word(self):
        """Listens for the wake word using faster-whisper."""
        print(f"\nListening for wake word: '{self.wake_word}'...")
        
        valid_wake_words = [
            self.wake_word, 
            "hey john", 
            "hey jon",
            "a john",
            "he john",
            "hey joan",
            "hey joe",
            "hey jaw",
            "hey jaw.",
            "hey john.",
            "hey john?",
            "hey john!"
        ]
        
        while True:
            text = self._record_until_silence(max_silence_duration=1.0)
            if not text:
                continue

            for ww in valid_wake_words:
                if ww in text:
                    print(f"\n[DEBUG] Wake word detected: '{text}'!")
                    return True

    def listen_for_query(self) -> str:
        """Listens for the user's query after the wake word is detected."""
        print("Listening for query...")
        text = self._record_until_silence(max_silence_duration=2.0)
        
        if text:
            print(f"User query: {text}")
            # Strip punctuation for cleaner passing to LLM (optional, but good for Whisper)
            return text
        else:
            print("Could not understand the query.")
            return ""
