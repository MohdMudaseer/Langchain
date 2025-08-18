from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation", huggingfacehub_api_token=huggingface_api)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str=Field(description='Name of the person')
    age:int=Field(gt=18,description="Age of the person")
    city: str = Field(description='Name of the city the person belongs to')

parser=PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)

