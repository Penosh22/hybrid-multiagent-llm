import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from tools.sentiment_analysis_tool import retail_sentiment_analysis
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.fundamental_analysis_tool import yf_fundamental_analysis
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools.browser_tool import BrowserTools  # Import the BrowserTools class
from tools.search_tools import SearchTools  # Import the SearchTools class
from tools.news_tool import tradingview_news_tool  # Import the yahoo_news_tool
from crewai_tools import tool

load_dotenv()

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
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
        stock_symbol (str): The stock symbol to search for.
        limit (int): Number of posts to fetch from TradingView.
    
    Returns:
        dict: Sentiment counts for TradingView news.
    """
    sentiments_counts = {}
    pipeline, label_encoder = joblib.load(r'E:\Data Science\Capstone\repository\financial_Agent\text_classification_pipeline.pkl')

    tv_news = get_tradingview_news(stock_symbol, limit)
    for post in tv_news:
        sentiment = pipeline.predict([str(post)])[0]
        sentiment = label_encoder.inverse_transform([sentiment])[0]
        
        if sentiment not in sentiments_counts:
            sentiments_counts[sentiment] = 0
        sentiments_counts[sentiment] += 1

    return sentiments_counts
# Model Selection
def initialize_llm(model_option, openai_api_key):
    if model_option == 'gpt-3.5-turbo-1106':
        return ChatOpenAI(openai_api_key=openai_api_key, model='gpt-3.5-turbo-1106', temperature=0.1)
    elif model_option == 'OpenAI GPT-4o Mini':
        return ChatOpenAI(openai_api_key=openai_api_key, model='gpt-4o-mini', temperature=0.1)
    else:
        raise ValueError("Invalid model option selected")
# Function to handle dynamic user queries and responses
def handle_query(user_query, model_option, openai_api_key):
    llm = initialize_llm(model_option, openai_api_key)

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
        verbose=False,
        memory=True,
        backstory="An expert in financial analysis with deep understanding of various analytic tools, you're adept at providing dynamic and insightful information.",
        tools=[yf_fundamental_tool,yf_tech_tool,retail_sentiment_tool,market_sentiment_tool,serper_tool,news_tool],
        llm=llm
    )
    # Agents Definitions
    researcher = Agent(
        role='Senior Stock Market Researcher',
        goal='Gather and analyze comprehensive data about stock_symbol mentioned in {user_query}',
        verbose=False,
        memory=True,
        backstory="With a Ph.D. in Financial Economics and 15 years of experience in equity research, you're known for your meticulous data collection and insightful analysis.",
        tools=[retail_sentiment_tool, serper_tool, news_tool],
        llm=llm
    )

    technical_analyst = Agent(
        role='Expert Technical Analyst',
        goal='Perform an in-depth technical analysis on stock_symbol mentioned in {user_query}',
        verbose=False,
        memory=True,
        backstory="As a Chartered Market Technician (CMT) with 15 years of experience, you have a keen eye for chart patterns and market trends.",
        tools=[yf_tech_tool],
        llm=llm
    )
    fundamental_analyst = Agent(
        role='Senior Fundamental Analyst',
        goal='Conduct a comprehensive fundamental analysis of stock_symbol mentioned in {user_query}',
        verbose=False,
        memory=True,
        backstory="With a CFA charter and 15 years of experience in value investing, you dissect financial statements and identify key value drivers.",
        tools=[yf_fundamental_tool],
        llm=llm
    )
    reporter = Agent(
        role='Chief Investment Strategist',
        goal='Synthesize all analyses to create a definitive investment report on stock_symbol mentioned in {user_query}',
        verbose=False,
        memory=True,
        backstory="As a seasoned investment strategist with 20 years of experience, you weave complex financial data into compelling investment narratives.",
        tools=[retail_sentiment_tool, serper_tool, yf_fundamental_tool, yf_tech_tool,news_tool ],
        llm=llm
    )
    

    #, serper_tool, yf_tech_tool, yf_fundamental_tool, YahooFinanceNewsTool(), scrape_website_tool, search_internet_tool, search_news_tool, yahoo_finance_news_tool
    # Task for dynamic interaction
    dynamic_task = Task(
        description=(
            "Analyze and respond to the query: {user_query}. Include:\n"
            "1. Current stock performance.\n"
            "2. Key financial metrics.\n"
            "3. Recent news.\n"
            "4. Analyst ratings.\n"
            "5. Reddit sentiment.\n"
            "6. Technical analysis.\n"
            "7. Fundamental analysis.\n"
            "8. Website content scraping and summarization if relevant.\n"
            "9. Internet search results.\n"
            "10. News search results.\n"
            "11. Yahoo Finance news results.\n"
            "Utilize all available tools to gather and process this information."
        ),
        expected_output='Provide a dynamic, interactive response with comprehensive insights.',
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
            "5. Reddit sentiment analysis (100 posts).\n"
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
            "7. Reddit Sentiment: Key takeaway from sentiment analysis, including the number of positive, negative and neutral comments and total comments.\n"
            "8. Investment Thesis: Bull and bear cases.\n"
            "9. Price Target: 12-month forecast.\n"
        ),
        expected_output='A 600-word investment report with clear sections, key insights.',
        agent=reporter
    )

    # Crew Definition and Kickoff for Dynamic Interaction
    crew = Crew(
        agents=[converser,researcher, technical_analyst, fundamental_analyst, reporter],
        tasks=[dynamic_task,research_task, technical_analysis_task, fundamental_analysis_task, report_task],
        process=Process.sequential,  # Using a sequential process for dynamic interaction
        cache=True
    )

    result = crew.kickoff(inputs={'user_query': user_query})
    os.makedirs('./crew_results', exist_ok=True)
    file_path = f"./crew_results/crew_result.md"
    result_str = str(result)
    with open(file_path, 'w') as file:
        file.write(result_str)
    
    tool_output = result

    # Use LLM to generate a natural language response based on the tool output
    natural_language_response = llm.invoke(f"Based on the following analysis, please give a detailed analysis of the output for actionable insights and analytical decision making:\n{tool_output}")

    # Access the content of the AIMessage directly
    final_response = natural_language_response.content  # Changed this line

    return final_response

