import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key: 
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables.")

chat_model = GoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=google_api_key,
    temperature=0.7
)

response = chat_model.invoke("What is the capital of France?")
print(response.content)
