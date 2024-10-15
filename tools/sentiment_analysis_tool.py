import os
import torch
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from crewai_tools import tool

# Download hf model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text.
    """
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
    labels = ["negative", "neutral", "positive"]
    label = labels[scores.argmax()]
    return label

def get_tradingview_posts(stock_symbol, limit=10):
    """
    Get the latest posts from TradingView Minds section for a given stock symbol.
    """
    url = f"https://in.tradingview.com/symbols/NSE-{stock_symbol}/minds/"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        divs = soup.find_all("div", class_="message-3bvBdNT_ symbol-page-tab-3bvBdNT_")
        posts = []
        count = 0
        for div in divs:
            span_content = div.find("span", class_="mind-body-ast-tree-3bvBdNT_ symbol-page-tab-3bvBdNT_")
            if span_content:
                text = span_content.get_text(separator="\n")
                posts.append(text)
                count += 1
                if count == limit:
                    break
        return posts
    else:
        print(f"Failed to fetch TradingView page. Status code: {response.status_code}")
        return []

@tool
def retail_sentiment_analysis(stock_symbol: str, limit: int = 10):
    """
    Perform sentiment analysis on posts from TradingView Minds about a stock symbol.
    
    Args:
        stock_symbol (str): The stock symbol to search for in tradingview ex: "HDFCBANK","LTIM","RELIANCE".
        limit (int): Number of posts to fetch from TradingView.
    
    Returns:
        dict: Sentiment counts for TradingView posts.
    """
    sentiments_counts = {'neutral': 0, 'negative': 0, 'positive': 0}
    
    # Analyze TradingView posts
    tv_posts = get_tradingview_posts(stock_symbol, limit)
    for post in tv_posts:
        sentiment = analyze_sentiment(post)
        sentiments_counts[sentiment] += 1
    
    return sentiments_counts

