from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Get the key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

# Create LLM instance with API key
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    api_key=api_key
)

# Invoke the model
result = llm.invoke("What is the capital of India?")
print(result)
