import os
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads your .env file

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ No API key found — check your .env file")
else:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one short sentence."
    )
    print("✅ Gemini says:", response.text)