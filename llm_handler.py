import os
from google import genai
from google.genai import types

class LLMHandler:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-2.5-flash"

        self.system_prompt = (
            "You are a general purpose AI assistant named John, created by students of the Department of "
            "Computer and Information Science of UKF College of Engineering. "
            "You have access to Google Search and can retrieve real-time information including news, "
            "weather, prices, sports scores, and any current events. "
            "When a query requires up-to-date information, use your search capability to fetch it. "
            "Keep responses concise and conversational since they will be read aloud by a text-to-speech engine. "
            "Avoid markdown formatting, bullet points, or symbols — use plain spoken sentences only."
        )

        # Tool: Google Search grounding — gives Gemini live web access
        self.search_tool = types.Tool(google_search=types.GoogleSearch())

        # Chat session maintains conversation history across turns
        self.chat = self.client.chats.create(
            model=self.model_id,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                tools=[self.search_tool],
                temperature=0.7,
            )
        )

        print("[LLM] Gemini with Google Search grounding enabled ✓")

    def generate_response_stream(self, query: str):
        """Sends the user's query and yields response text chunks as they stream in."""
        try:
            print(f"[LLM] Query: '{query}'")
            response_stream = self.chat.send_message_stream(query)
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"[LLM] Error: {e}")
            yield "I'm sorry, I encountered an error while trying to process your request."
