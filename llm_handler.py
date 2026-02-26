import os
from google import genai

class LLMHandler:
    def __init__(self):
        # We explicitly pass the API key loaded from the .env
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-2.5-flash"
        
        self.system_prompt = (
            "You are a general purpose search assistant named John, created by students of the Department of "
            "Computer and Information Science of UKF College of Engineering. "
            "Your response should have a professional tone and must be apt for the given query."
        )

        # We can use the chat session to maintain history if needed, 
        # or just generate responses for single queries.
        self.chat = self.client.chats.create(
            model=self.model_id,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                temperature=0.7,
            )
        )

    def generate_response_stream(self, query: str):
        """Sends the user's query and returns the streaming response."""
        try:
            print(f"Sending query to Gemini: '{query}'")
            # We use send_message_stream to get chunks as they are generated
            response_stream = self.chat.send_message_stream(query)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"Error communicating with Gemini: {e}")
            yield "I'm sorry, I encountered an error while trying to process your request."
