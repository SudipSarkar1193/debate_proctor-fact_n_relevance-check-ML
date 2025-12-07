import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    print("❌ Error: API_KEY not found in .env")
    exit()

genai.configure(api_key=api_key)

print(f"🔑 Checking models for API Key: {api_key[:5]}...")
print("-" * 30)

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ AVAILABLE: {m.name}")
except Exception as e:
    print(f"❌ Error listing models: {e}")