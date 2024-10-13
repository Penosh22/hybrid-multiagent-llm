import streamlit as st
from mys import handle_query
import os
from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from tools.sentiment_analysis_tool import retail_sentiment_analysis
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.fundamental_analysis_tool import yf_fundamental_analysis
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools.browser_tool import BrowserTools  # Import the BrowserTools class
from tools.search_tools import SearchTools  # Import the SearchTools class
from tools.news_tool import tradingview_news_tool  # Import the yahoo_news_tool

load_dotenv()


# Streamlit App Title
st.title("Financial Investment Assistant")

# Select the model option
model_option = st.selectbox('Choose the LLM model:', ['gpt-3.5-turbo-1106', 'OpenAI GPT-4o Mini'])

# API Key from environment
openai_api_key = os.getenv('OPENAI_API_KEY')

# User Input Section
user_query = st.text_input("Enter your query (e.g., 'top 10 stocks to invest now'):")

# Button to run the query
if st.button("Get Analysis"):
    if user_query:
        # Call the function from financial_agent.py to get the result
        with st.spinner("Processing..."):
            response = handle_query(user_query, model_option, openai_api_key)
        
        # Display the result
        st.subheader("Investment Report:")
        st.write(response)
    else:
        st.warning("Please enter a query.")
