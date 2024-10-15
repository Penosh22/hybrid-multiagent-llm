import os
import faiss
import numpy as np
import openai
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from tools.sentiment_analysis_tool import retail_sentiment_analysis
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.fundamental_analysis_tool import yf_fundamental_analysis
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools.search_tools import SearchTools  # Import the SearchTools class
from tools.news_tool import tradingview_news_tool  # Import the yahoo_news_tool
from crewai_tools import tool
from preprocessing import preprocess_text
from sentence_transformers import SentenceTransformer
import joblib
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_tradingview_news(stock_symbol: str, max_articles: int = 5):
    """Fetch news articles from TradingView."""
    tvnewsdata = []
    url = f"https://news-headlines.tradingview.com/v2/view/headlines/symbol?client=web&lang=en&section=&streaming=true&symbol=NSE%3A{stock_symbol}"
    headers = {
        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        "Referer": "https://in.tradingview.com/",
        "sec-ch-ua-mobile": "?0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "sec-ch-ua-platform": '"Windows"'
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    story_paths = [item['storyPath'] for item in data['items']]
    
    limit = 0
    for path in story_paths:
        iurl = "https://in.tradingview.com" + path
        response2 = requests.get(iurl, headers=headers)
        soup = BeautifulSoup(response2.content, "html.parser")
        news_content = "\n".join(p.get_text(separator="\n") for p in soup.find_all('p')[:-23])
        tvnewsdata.append(news_content)
        limit += 1
        if limit == max_articles:
            break
            
    return tvnewsdata

@tool
def market_sentiment_analysis(stock_symbol: str, limit: int = 5):
    """
    Perform sentiment analysis on posts from news about a stock symbol.
    
    Args:
        stock_symbol (str): The stock symbol to search for, in tradingview example: "HDFCBANK","LTIM","RELIANCE". 
        limit (int): Number of posts to fetch from TradingView.
    
    Returns:
        dict: Sentiment counts for TradingView news.
    """
    print(stock_symbol)
    sentiments_counts = {}
    pipeline, label_encoder = joblib.load(r'E:\Data Science\Capstone\Mindyourstock\text_classification_pipeline.pkl')

    tv_news = get_tradingview_news(stock_symbol, limit)
    for post in tv_news:
        sentiment = pipeline.predict([str(post)])[0]
        sentiment = label_encoder.inverse_transform([sentiment])[0]
        
        if sentiment not in sentiments_counts:
            sentiments_counts[sentiment] = 0
        sentiments_counts[sentiment] += 1

    return sentiments_counts
# Model Selection
def initialize_llm(model_option, api_key):
    if model_option == 'gpt-3.5-turbo-1106':
        return ChatOpenAI(openai_api_key=api_key, model='gpt-3.5-turbo-1106', temperature=0.1)
    elif model_option == 'OpenAI GPT-4o Mini':
        return ChatOpenAI(openai_api_key=api_key, model='gpt-4o-mini', temperature=0.1)
    elif model_option == 'llama3-8b-8192':
        return ChatGroq(groq_api_key=api_key , model='groq/llama3-8b-8192', temperature=0.1)  # Correct provider here
    else:
        raise ValueError("Invalid model option selected")

# FAISS index initialization
embedding_dim = 768  
index_file = './faiss_index.index'

# Check if FAISS index already exists, else create a new one
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
else:
    index = faiss.IndexFlatL2(embedding_dim)

# Store for metadata (query-response pairs)
query_response_metadata = []

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

def get_embeddings(text):
    """Generate embeddings for the text using HuggingFace's model."""
    embedding = model.encode(text)
    return np.array(embedding)

def store_query_response(query, response):
    """Store the query and response in the FAISS vector DB."""
    # Generate embeddings for the query
    query_embedding = get_embeddings(query)
    
    # Add the embedding to the FAISS index
    index.add(np.array([query_embedding], dtype=np.float32))
    
    # Save the query and response in metadata
    query_response_metadata.append({
        'query': query,
        'response': response
    })
    
    # Persist the FAISS index and metadata
    faiss.write_index(index, index_file)
    with open('query_response_metadata.npy', 'wb') as f:
        np.save(f, query_response_metadata)

def load_metadata():
    """Load stored metadata."""
    if os.path.exists('query_response_metadata.npy'):
        with open('query_response_metadata.npy', 'rb') as f:
            return np.load(f, allow_pickle=True).tolist()
    return []

def find_similar_response(query):
    """Find a similar response from the FAISS index."""
    query_embedding = get_embeddings(query)
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), k=1)  # Search for the closest response
    
    if distances[0][0] < 1.0:  # A threshold for considering it a match (you can adjust this)
        similar_query = query_response_metadata[indices[0][0]]['query']
        similar_response = query_response_metadata[indices[0][0]]['response']
        return similar_response, similar_query
    return None, None

# Load metadata if it exists
query_response_metadata = load_metadata()



# Function to handle dynamic user queries and responses
def handle_query(user_query, model_option, api_key):
    # First check for a similar response
    stored_response, similar_query = find_similar_response(user_query)

    if stored_response is not None:
        return f"Found a similar response for '{similar_query}': {stored_response}"

    # If no stored response, proceed with agent processing
    llm = initialize_llm(model_option, api_key)

    # Tools Initialization
    retail_sentiment_tool = retail_sentiment_analysis
    serper_tool = SerperDevTool()
    market_sentiment_tool = market_sentiment_analysis
    yf_tech_tool = yf_tech_analysis
    yf_fundamental_tool = yf_fundamental_analysis
    news_tool = tradingview_news_tool

    # Conversation Agent Definition
    converser = Agent(
        role='Financial Data Analyst',
        goal='Dynamically respond to queries using available tools',
        verbose=True,
        memory=True,
        backstory="An expert in financial analysis with deep understanding of various analytic tools, you're adept at providing dynamic and insightful information.",
        tools=[news_tool, yf_fundamental_tool, yf_tech_tool, retail_sentiment_tool, market_sentiment_tool, serper_tool],
        llm=llm
    )
    
    # Agents Definitions
    researcher = Agent(
        role='Senior Stock Market Researcher',
        goal='Gather and analyze comprehensive data about stock_symbol mentioned in {user_query}',
        verbose=True,
        memory=True,
        backstory="With a Ph.D. in Financial Economics and 15 years of experience in equity research, you're known for your meticulous data collection and insightful analysis.",
        tools=[market_sentiment_tool, retail_sentiment_tool, serper_tool, news_tool],
        llm=llm
    )

    technical_analyst = Agent(
        role='Expert Technical Analyst',
        goal='Perform an in-depth technical analysis on stock_symbol mentioned in {user_query}',
        verbose=True,
        memory=True,
        backstory="As a Chartered Market Technician (CMT) with 15 years of experience, you have a keen eye for chart patterns and market trends.",
        tools=[yf_tech_tool],
        llm=llm
    )
    fundamental_analyst = Agent(
        role='Senior Fundamental Analyst',
        goal='Conduct a comprehensive fundamental analysis of stock_symbol mentioned in {user_query}',
        verbose=True,
        memory=True,
        backstory="With a CFA charter and 15 years of experience in value investing, you dissect financial statements and identify key value drivers.",
        tools=[yf_fundamental_tool],
        llm=llm
    )
    reporter = Agent(
        role='Chief Investment Strategist',
        goal='Synthesize all analyses to create a definitive investment report on stock_symbol mentioned in {user_query}',
        verbose=True,
        memory=True,
        backstory="As a seasoned investment strategist with 20 years of experience, you weave complex financial data into compelling investment narratives.",
        tools=[market_sentiment_tool, retail_sentiment_tool, serper_tool, yf_fundamental_tool, yf_tech_tool, news_tool],
        llm=llm
    )
    
    # Task for dynamic interaction
    dynamic_task = Task(
        description=(
            "Analyze and respond to the query: {user_query}. Include:\n"
            "tool output in the response"
        ),
        expected_output='Provide a dynamic, interactive response with the data you have available',
        agent=converser
    )
    
    # Task Definitions
    research_task = Task(
        description=(
            "Conduct research on stock symbol in the {user_query}. Your analysis should include:\n"
            "1. Current stock price and historical performance (5 years).\n"
            "2. Key financial metrics (P/E, EPS growth, revenue growth, margins).\n"
            "3. Recent news and press releases (1 month).\n"
            "4. Analyst ratings and price targets (min 3 analysts).\n"
            "5. sentiment analysis (100 posts).\n"
            "6. Major institutional holders and recent changes.\n"
            "7. Competitive landscape and market share.\n"
            "Use reputable financial websites for data."
        ),
        expected_output='A detailed 150-word research report with data sources and brief analysis.',
        agent=researcher
    )
    technical_analysis_task = Task(
        description=(
            "Perform technical analysis on stock_symbol mentioned in {user_query}. Include:\n"
            "1. 50-day and 200-day moving averages (1 year).\n"
            "2. Key support and resistance levels (3 each).\n"
            "3. RSI and MACD indicators.\n"
            "4. Volume analysis (3 months).\n"
            "5. Significant chart patterns (6 months).\n"
            "6. Fibonacci retracement levels.\n"
            "7. Comparison with sector's average.\n"
            "Use the yf_tech_analysis tool for data."
        ),
        expected_output='A 100-word technical analysis report with buy/sell/hold signals and annotated charts.',
        agent=technical_analyst
    )
    fundamental_analysis_task = Task(
        description=(
            "Conduct fundamental analysis of stock_symbol mentioned in {user_query}. Include:\n"
            "1. Review last 3 years of financial statements.\n"
            "2. Key ratios (P/E, P/B, P/S, PEG, Debt-to-Equity, etc.).\n"
            "3. Comparison with main competitors and industry averages.\n"
            "4. Revenue and earnings growth trends.\n"
            "5. Management effectiveness (ROE, capital allocation).\n"
            "6. Competitive advantages and market position.\n"
            "7. Growth catalysts and risks (2-3 years).\n"
            "8. DCF valuation model with assumptions.\n"
            "Use yf_fundamental_analysis tool for data."
        ),
        expected_output='A 100-word fundamental analysis report with buy/hold/sell recommendation and key metrics summary.',
        agent=fundamental_analyst
    )
    report_task = Task(
        description=(
            "Create an investment report on stock_symbol mentioned in {user_query}. Include:\n"
            "1. Executive Summary: Investment recommendation.\n"
            "2. Company Snapshot: Key facts.\n"
            "3. Financial Highlights: Top metrics and peer comparison.\n"
            "4. Technical Analysis: Key findings.\n"
            "5. Fundamental Analysis: Top strengths and concerns.\n"
            "6. Risk and Opportunity: Major risk and growth catalyst.\n"
            "7. Sentiment: Key takeaway from sentiment analysis, including the number of positive, negative and neutral comments and total comments.\n"
            "8. Investment Thesis: Bull and bear cases.\n"
            "9. Price Target: 12-month forecast.\n"
        ),
        expected_output='A 600-word investment report with clear sections, key insights.',
        agent=reporter
    )
    
    # Crew Definition and Kickoff for Dynamic Interaction
    crew = Crew(
        agents=[converser, researcher, technical_analyst, fundamental_analyst, reporter],
        tasks=[dynamic_task, research_task, technical_analysis_task, fundamental_analysis_task, report_task],
        process=Process.sequential,  # Using a sequential process for dynamic interaction
        cache=True
    )

    result = crew.kickoff(inputs={
        'user_query': user_query
    })
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('./crew_results', exist_ok=True)
    file_path = f"./crew_results/crew_result_{current_time}.pdf"
    result_str = str(result)
    with open(file_path, 'w') as file:
        file.write(result_str)
    
    # Store the result in FAISS
    final_response = result
    
    # Store the query and response in FAISS
    store_query_response(user_query, final_response)
    return final_response