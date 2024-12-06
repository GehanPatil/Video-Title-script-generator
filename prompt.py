import os
import requests
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import LLMChain, SimpleSequentialChain,SequentialChain
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub

load_dotenv()

# Set Hugging Face Hub API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# App framework
st.title('Video Title & Script Generator')
prompt = st.text_input('Plug in your prompt here')
topic = prompt 

# Prompt Templates
title_template = PromptTemplate(
    input_variables= ["topic"],
    template= "Write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables= ['title'],
    template= "write me a youtube video script based on this title {title}"
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Set up the language model using the Hugging Face Hub repository
repo_id = "tiiuae/falcon-7b-instruct"
#repo_id = "gpt2-large"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.8, "max_new_tokens": 2000})
title_chain = LLMChain(prompt=title_template, llm=llm, verbose= True, output_key='title', memory=title_memory)
script_chain = LLMChain(prompt=script_template, llm=llm, verbose= True,output_key='script',memory=script_memory)
sequential_Chain = SequentialChain(chains=[title_chain,script_chain], input_variables=['topic'], output_variables=['title','script'], verbose=True)

if prompt:
    response = sequential_Chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)
