import streamlit as st
from mysv2 import handle_query
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

# Initialize session state for conversation history if not already done
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Streamlit App Title
st.title("Financial Investment Assistant")

# Select the model option
model_option = st.selectbox('Choose the LLM model:', ['gpt-3.5-turbo-1106', 'OpenAI GPT-4o Mini'])

# API Key from environment
openai_api_key = os.getenv('OPENAI_API_KEY')

# Chat history display section
st.subheader("Chat History")
if st.session_state['conversation']:
    # Loop through chat history and display messages from user and assistant
    for i, chat in enumerate(st.session_state['conversation']):
        if chat["role"] == "user":
            st.markdown(f"**🧑 User {i//2 + 1}:** {chat['content']}")
        elif chat["role"] == "assistant":
            st.markdown(f"**🤖 Assistant {i//2 + 1}:** {chat['content']}")
else:
    st.write("No conversation history yet.")

# User Input Section
user_query = st.text_input("Enter your query (e.g., 'top 10 stocks to invest now'): ")

# Button to submit the query
if st.button("Send"):
    if user_query:
        # Append user query to conversation history
        st.session_state.conversation.append({"role": "user", "content": user_query})

        # Call the function to get the result
        with st.spinner("Processing..."):
            response = handle_query(user_query, model_option, openai_api_key)
        
        # Append the model response to conversation history
        st.session_state.conversation.append({"role": "assistant", "content": response})

# Button to regenerate the last response
if st.button("Regenerate Response"):
    if st.session_state['conversation']:
        # Regenerate the last response based on the last user query
        last_user_query = next((item['content'] for item in reversed(st.session_state['conversation']) if item['role'] == 'user'), None)
        if last_user_query:
            with st.spinner("Regenerating..."):
                regenerated_response = handle_query(last_user_query, model_option, openai_api_key)
            
            # Replace the last assistant response with the new one
            for i in range(len(st.session_state['conversation'])-1, -1, -1):
                if st.session_state['conversation'][i]['role'] == 'assistant':
                    st.session_state['conversation'][i]['content'] = regenerated_response
                    break
    else:
        st.warning("No conversation to regenerate.")
