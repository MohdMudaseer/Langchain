from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

huggingface_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not huggingface_token:
    raise ValueError("HUGGINGFACEHUB_ACCESS_TOKEN is not set in the environment variables.")

llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b",task="text-generation",huggingfacehub_api_token=huggingface_token)

chat_model = ChatHuggingFace(llm=llm)

# stramlit UI

st.title("Research Assisstant")

st.header("Research Assistant")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

prompt_template = load_prompt("2_Prompts/prompt_template.json")

if st.button("summarize"):
    with st.spinner("Generating summary..."):
        chain= prompt_template | chat_model
        response = chain.invoke({
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        })
        output = response.content
        st.write(output)
