import os
from dotenv import load_dotenv

# Load environment variables from .env file FIRST
load_dotenv()

from audio_handler import AudioHandler
from llm_handler import LLMHandler
from tts_handler import TTSHandler

def main():
    print("Initializing Ansar AI Assistant...")
    
    if not os.getenv("GEMINI_API_KEY"):
         print("Error: GEMINI_API_KEY environment variable not found.")
         print("Please create a .env file and add your GEMINI_API_KEY.")
         return

    try:
        audio = AudioHandler(wake_word="hey ansar")
        llm = LLMHandler()
        tts = TTSHandler(voice="en-US-AriaNeural") # Default a professional female voice
    except Exception as e:
        print(f"Failed to initialize components: {e}")
        return

    print("\nAnsar is ready! Say 'Hey Ansar' to start a conversation.\n")
    
    while True:
        try:
            # 1. Listen for the wake word
            if audio.listen_for_wake_word():
                
                # 2. Optionally, give a quick audio queue that it's listening
                # Example: tts._generate_and_queue_audio("Yes?")
                
                # 3. Listen for the user's query
                query = audio.listen_for_query()
                
                if query:
                    # 4. Send query to LLM and get a streaming response
                    response_stream = llm.generate_response_stream(query)
                    
                    # 5. Process the stream and stream text-to-speech
                    tts.process_llm_stream(response_stream)
                else:
                    print("No query detected. Going back to listening for wake word.")
                    
        except KeyboardInterrupt:
             print("\nExiting Ansar AI Assistant. Goodbye!")
             break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            break

if __name__ == "__main__":
    main()
