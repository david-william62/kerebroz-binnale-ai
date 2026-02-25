import sounddevice as sd
import numpy as np
import os
import webrtcvad
import queue
import json
from vosk import Model, KaldiRecognizer

class AudioHandler:
    def __init__(self, wake_word="hey ansar"):
        self.wake_word = wake_word.lower()
        
        # Audio configuration
        self.RATE = 16000
        self.CHANNELS = 1
        self.CHUNK = 480  # 30ms chunk at 16kHz
        
        # Load Vosk Model
        model_path = os.path.join(os.path.dirname(__file__), "venv", "model")
        if not os.path.exists(model_path):
             print(f"Error: Vosk model not found at {model_path}. Please download it.")
        else:
             self.model = Model(model_path)
             self.recognizer = KaldiRecognizer(self.model, self.RATE)
             self.recognizer.SetWords(False)
             
        # VAD helps us detect when someone is speaking vs silence
        self.vad = webrtcvad.Vad(3) # Aggressiveness 3 is highest
        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time, status):
        """This is called for each audio block."""
        if status:
            pass # Ignore overflows for now to avoid console spam
        # We need 16-bit PCM for VAD and Vosk
        audio_data = (indata[:, 0] * 32767).astype(np.int16)
        self.audio_queue.put(audio_data)

    def _record_until_silence(self, max_silence_duration=1.5):
        """Records audio until silence is detected, and transcribes locally."""
        frames = []
        silence_frames = 0
        is_speaking = False
        max_silence_chunks = int(max_silence_duration * self.RATE / self.CHUNK)
        
        # Clear queue and recognizer before starting
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
            
        # Reset the Kaldi recognizer so it doesn't carry over words from previous queries
        self.recognizer = KaldiRecognizer(self.model, self.RATE)
            
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
                    # Feed audio to the recognizer piece by piece
                    self.recognizer.AcceptWaveform(chunk.tobytes())
                elif is_speaking:
                    silence_frames += 1
                    frames.append(chunk)
                    self.recognizer.AcceptWaveform(chunk.tobytes())
                    
                    if silence_frames > max_silence_chunks:
                        break # Stopped speaking
                
                # Keep a tiny rolling window of audio before speaking starts so we don't clip the first word
                if not is_speaking:
                    frames.append(chunk)
                    if len(frames) > int(0.5 * self.RATE / self.CHUNK): # Keep 0.5 seconds
                        dropped_chunk = frames.pop(0)
                    else:
                        self.recognizer.AcceptWaveform(chunk.tobytes())

        # Save to temporary WAV file
        if not frames:
            return ""
            
        audio_data = np.concatenate(frames)
        
        # Check if the audio is mostly just silence/noise even if VAD triggered
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 500:
             return ""

        # Get final result form vosk
        final_res = self.recognizer.FinalResult()
        
        try:
             text = json.loads(final_res).get("text", "")
             return text.strip().lower()
        except:
             return ""

    def listen_for_wake_word(self):
        """Continuously listens for the wake word using Vosk locally."""
        print(f"\nListening for wake word: '{self.wake_word}'...")
        while True:
            text = self._record_until_silence(max_silence_duration=1.0)
            if not text: continue
            
            print(f"[DEBUG] Local STT heard: '{text}'")
            
            # Check for the wake word or very common transcription errors
            valid_wake_words = [
                self.wake_word, 
                "hey answer", 
                "a answer",
                "hey and far",
                "hey and sir",
                "hey insar",
                "a insar"
            ]
            
            if any(ww in text for ww in valid_wake_words):
                print("Wake word detected!")
                return True

    def listen_for_query(self) -> str:
        """Listens for the user's query after the wake word is detected."""
        print("Listening for query...")
        text = self._record_until_silence(max_silence_duration=2.0)
        
        if text:
            print(f"User query: {text}")
            return text
        else:
            print("Could not understand the query.")
            return ""
