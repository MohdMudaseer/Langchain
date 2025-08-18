from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation", huggingfacehub_api_token=huggingface_api)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

pydantic_parser=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template="classify the sentiment of the following text: {feedback} as either positive or negative \n{format_expression}",
    input_variables=["feedback"],
    partial_variables={'format_expression':pydantic_parser.get_format_instructions()}
)

classification_chain=prompt1 | model | pydantic_parser

prompt2=PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

conditional_chain=RunnableBranch(
    (lambda x:x.sentiment == "positive",prompt2 | model | parser),
    (lambda x:x.sentiment == "negative",prompt3 | model | parser),
    RunnableLambda(lambda x: "No appropriate response available.")
)
chain = classification_chain | conditional_chain

print(chain.invoke({'feedback': 'This is the worst phone'}))

chain.get_graph().print_ascii()