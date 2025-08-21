from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence,RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

huggingface_api=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not huggingface_api:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN not found")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b", task="text-generation", huggingfacehub_api_token=huggingface_api)
model=ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({'topic':'cricket'}))
