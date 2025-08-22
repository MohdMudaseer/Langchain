from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation", huggingfacehub_api_token=huggingface_api)
model=ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
    template='Write the summary of the following poem: \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader=TextLoader("4_Indexes/1_Document_loaders/cricket.txt",encoding='utf-8')
text=loader.load()
print(len(text))
print(type(text))
print(type(text[0]))
print(text[0].page_content)
print(text[0].metadata)

chain=prompt1 | model | parser
print(chain.invoke({"poem":text[0].page_content}))