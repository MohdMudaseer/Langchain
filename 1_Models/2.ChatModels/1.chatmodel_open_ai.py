import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key: 
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables.")

chat_model = ChatOpenAI(
    model="gpt-4",
    api_key=openai_api_key,
    temperature=0.7
)

response = chat_model.invoke("What is the capital of France?")
print(response.content)
