from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
chat_huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not chat_huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b", task="text-generation", huggingfacehub_api_token=chat_huggingface_api)
chat_model=ChatHuggingFace(llm=llm)
chat_history=[]
while True:
    user_input=input("User:")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        break
    output=chat_model.invoke(chat_history)
    chat_history.append(output.content)
    print(f"Chatbot: {output.content}")