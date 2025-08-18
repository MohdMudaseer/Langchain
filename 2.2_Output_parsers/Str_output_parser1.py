from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation", huggingfacehub_api_token=huggingface_api)

model=ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

output_parser=StrOutputParser()

chain=template1 | model | output_parser | template2 | model | output_parser

response=chain.invoke({'topic': 'black hole'})

print(response)