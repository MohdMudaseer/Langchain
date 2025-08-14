import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,HuggingFaceEmbeddings

load_dotenv()

huggingfacehub_access_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not huggingfacehub_access_token: 
    raise ValueError("❌ HUGGINGFACEHUB_ACCESS_TOKEN not found in environment variables.")

embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents= [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector=embeddings.embed_documents(documents)
print(str(vector))