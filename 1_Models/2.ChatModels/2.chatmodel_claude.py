import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not anthropic_api_key: 
    raise ValueError("❌ ANTHROPIC_API_KEY not found in environment variables.")

chat_model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=anthropic_api_key,
    temperature=0.7
)

response = chat_model.invoke("What is the capital of France?")
print(response.content)
