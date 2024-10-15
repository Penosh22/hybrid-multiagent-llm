import streamlit as st
from code import handle_query
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from tools.sentiment_analysis_tool import retail_sentiment_analysis
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.fundamental_analysis_tool import yf_fundamental_analysis
from langchain_openai import ChatOpenAI
from tools.search_tools import SearchTools  # Import the SearchTools class
from tools.news_tool import tradingview_news_tool  # Import the yahoo_news_tool
import faiss
import numpy as np
import openai
from crewai_tools import tool
from preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer
import joblib
import requests
from bs4 import BeautifulSoup
load_dotenv()

from datetime import datetime



# Streamlit app title
st.title("Financial Assistant Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User inputs OpenAI API key and model option
# User inputs OpenAI API key and model option
llm_api_key = st.sidebar.text_input("LLM API Key", type="password")

# Align model options with those expected in the initialize_llm function
model_option = st.sidebar.selectbox("Model Option", [
    "gpt-3.5-turbo-1106",  # For GPT-3
    "OpenAI GPT-4o Mini",  # For GPT-4
    "llama3-8b-8192"  # For Llama model
])

# React to user input
if prompt := st.chat_input("What is your query?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if llm_api_key:
        # Call handle_query function to get the dynamic response
        response = handle_query(user_query=prompt, model_option=model_option, api_key=llm_api_key)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # If API key is not provided, display a warning message
        response = "Please provide a valid LLM API Key."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

