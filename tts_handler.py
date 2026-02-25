import asyncio
import edge_tts
import pygame
import os
import tempfile
import threading
import queue
import re

class TTSHandler:
    def __init__(self, voice="en-US-AriaNeural"):
        self.voice = voice
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.play_thread = None
        self.temp_dir = tempfile.mkdtemp()
        self.file_counter = 0
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()

    def process_llm_stream(self, response_stream):
        """
        Takes the streaming generator from the LLM, accumulates words into sentences,
        and queues them for TTS generation and playback.
        """
        self.is_playing = True
        self.play_thread = threading.Thread(target=self._play_audio_queue)
        self.play_thread.start()

        buffer = ""
        sentence_endings = re.compile(r'(?<=[.!?])\s')

        for chunk in response_stream:
            # Print the chunk to the console as it comes in
            print(chunk, end="", flush=True)
            buffer += chunk

            # Split buffer by sentence endings
            parts = sentence_endings.split(buffer)
            
            # If we have complete sentences in the buffer
            if len(parts) > 1:
                # All but the last part are complete sentences
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        self._generate_and_queue_audio(sentence)
                
                # The last part is the incomplete sentence, keep it in the buffer
                buffer = parts[-1]

        # Process any remaining text in the buffer after the stream ends
        buffer = buffer.strip()
        if buffer:
            self._generate_and_queue_audio(buffer)
            
        print() # Newline after the full text string is printed

        # Signal the playback thread that generation is done
        self.audio_queue.put(None)
        
        # Wait for all audio to finish playing
        if self.play_thread:
            self.play_thread.join()
        
        self.is_playing = False

    def _generate_and_queue_audio(self, text):
        """Generates audio for a sentence using edge-tts and adds it to the queue."""
        self.file_counter += 1
        output_file = os.path.join(self.temp_dir, f"audio_{self.file_counter}.mp3")
        
        # edge_tts requires asyncio, so we run it in a new event loop
        async def generate():
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_file)
            
        asyncio.run(generate())
        
        # Put the path to the generated audio file in the queue
        self.audio_queue.put(output_file)

    def _play_audio_queue(self):
        """Background thread that continuously plays audio files from the queue."""
        while True:
            audio_file = self.audio_queue.get()
            
            # None acts as a sentinel value to signal the end of the stream
            if audio_file is None:
                break
                
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                    
                # Clean up the temporary file after playing
                os.remove(audio_file)
                
            except Exception as e:
                print(f"Error playing audio: {e}")
            finally:
                self.audio_queue.task_done()
