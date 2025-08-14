import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

huggingfacehub_access_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not huggingfacehub_access_token: 
    raise ValueError("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables.")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b",task="text-generation",huggingfacehub_api_token=huggingfacehub_access_token)

chat_model = ChatHuggingFace(
    llm=llm
)
response = chat_model.invoke("What is the capital of France?")
print(response.content)
