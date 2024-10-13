import requests
from bs4 import BeautifulSoup
from crewai_tools import tool


# Tool to fetch news articles from TradingView
@tool
def tradingview_news_tool(stock_symbol: str, max_articles: int = 5):
    """
    Get the news articles from TradingView about a stock symbol.

    Args:
        stock_symbol (str): The stock symbol to search for.
        max_articles (int): The maximum number of articles to fetch.

    Returns:
        list: A list containing the news articles from TradingView.
    """
    tvnewsdata = []


    
    # URL and headers from the curl command
    url = f"https://news-headlines.tradingview.com/v2/view/headlines/symbol?client=web&lang=en&section=&streaming=true&symbol=NSE%3A{stock_symbol}"
    headers = {
        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
        "Referer": "https://in.tradingview.com/",
        "sec-ch-ua-mobile": "?0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "sec-ch-ua-platform": '"Windows"'
    }
    
    # Send the GET request
    response = requests.get(url, headers=headers)
    data = response.json()

    # Extract story paths
    story_paths = [item['storyPath'] for item in data['items']]

    limit = 0
    for path in story_paths:
        iurl = "https://in.tradingview.com" + path
        response2 = requests.get(iurl, headers=headers)
        soup = BeautifulSoup(response2.content, "html.parser")

        # Extract article content
        news_content = ""
        for p in soup.find_all('p')[:-23]:
            news_content += p.get_text(separator="\n") + "\n"

        tvnewsdata.append(news_content)
        limit += 1
        if limit == max_articles:
            break
            
    return tvnewsdata

