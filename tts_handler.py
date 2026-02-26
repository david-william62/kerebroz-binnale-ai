import threading
import queue
import re
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro as KokoroTTS

class TTSHandler:
    def __init__(self, voice="af_heart", speed=1.0, sample_rate=24000):
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate

        # Queue holds numpy audio arrays to be played
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.play_thread = None

        print("Loading Kokoro TTS model...")
        self.tts = KokoroTTS("venv/models/model.onnx","venv/voices/voices.bin")
        print("Kokoro TTS model loaded.")

    def process_llm_stream(self, response_stream):
        """
        Takes the streaming generator from the LLM, accumulates text into sentences,
        generates audio with Kokoro, and plays them back in order.
        """
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._play_audio_queue, daemon=True)
        self.play_thread.start()

        buffer = ""
        # Match sentence endings: period, exclamation, question mark followed by space or end
        sentence_endings = re.compile(r'(?<=[.!?])\s+')

        for chunk in response_stream:
            print(chunk, end="", flush=True)
            buffer += chunk

            parts = sentence_endings.split(buffer)

            # If we have more than one part, the first n-1 are complete sentences
            if len(parts) > 1:
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        self._generate_and_queue_audio(sentence)
                # Keep the incomplete tail in the buffer
                buffer = parts[-1]

        # Flush any remaining text
        buffer = buffer.strip()
        if buffer:
            self._generate_and_queue_audio(buffer)

        print()  # Newline after full response is printed

        # Sentinel to signal playback thread to stop
        self.audio_queue.put(None)

        if self.play_thread:
            self.play_thread.join()

        self.is_playing = False

    def _generate_and_queue_audio(self, text):
        """
        Generates audio for a sentence using Kokoro and puts the
        numpy audio array into the playback queue.
        """
        try:
            # v1.0 API: create() returns (samples, sample_rate) directly
            samples, sample_rate = self.tts.create(
                text,
                voice=self.voice,
                speed=self.speed,
                lang="en-us"
            )
            if samples is not None and len(samples) > 0:
                audio = np.array(samples, dtype=np.float32).flatten()
                self.sample_rate = sample_rate  # Use model's actual sample rate
                self.audio_queue.put(audio)
        except Exception as e:
            print(f"\n[TTS Error] Failed to generate audio for: '{text}'\n  Reason: {e}")

    def _play_audio_queue(self):
        """Background thread that continuously plays audio arrays from the queue."""
        while True:
            audio_array = self.audio_queue.get()
            
            # None acts as a sentinel value to signal the end of the stream
            if audio_array is None:
                self.audio_queue.task_done()
                break
                
            try:
                sd.play(audio_array, samplerate=self.sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self.audio_queue.task_done()