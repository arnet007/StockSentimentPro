import pandas as pd
import numpy as np
import re
import requests
import streamlit as st
from datetime import datetime, timedelta
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    """
    Cleans text by removing special characters, URLs, and extra whitespace
    
    Parameters:
    text (str): Text to clean
    
    Returns:
    str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to analyze sentiment
def analyze_sentiment(text):
    """
    Analyzes sentiment of text using TextBlob and VADER
    
    Parameters:
    text (str): Text to analyze
    
    Returns:
    dict: Dictionary containing sentiment scores
    """
    if not text:
        return {
            'compound': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'sentiment': 'neutral'
        }
    
    # Clean text
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return {
            'compound': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'sentiment': 'neutral'
        }
    
    # TextBlob sentiment
    blob = TextBlob(cleaned_text)
    textblob_polarity = blob.sentiment.polarity
    
    # VADER sentiment
    vader_scores = sia.polarity_scores(cleaned_text)
    
    # Combine scores (giving more weight to VADER)
    compound = vader_scores['compound']
    
    # Determine sentiment label
    if compound >= 0.05:
        sentiment = 'positive'
    elif compound <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'compound': compound,
        'positive': vader_scores['pos'],
        'negative': vader_scores['neg'],
        'neutral': vader_scores['neu'],
        'sentiment': sentiment
    }

# Function to get news for a stock
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_news(ticker, days=7, max_articles=10):
    """
    Gets news articles for a stock from Yahoo Finance
    
    Parameters:
    ticker (str): Stock ticker symbol
    days (int): Number of days to look back
    max_articles (int): Maximum number of articles to return
    
    Returns:
    pandas.DataFrame: DataFrame containing news articles
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return pd.DataFrame(), "No news found for this stock."
        
        # Convert to DataFrame
        news_df = pd.DataFrame(news)
        
        # Keep only relevant columns
        if 'title' in news_df.columns and 'link' in news_df.columns:
            news_df = news_df[['title', 'publisher', 'link', 'providerPublishTime']]
            
            # Convert timestamp to datetime
            news_df['date'] = pd.to_datetime(news_df['providerPublishTime'], unit='s')
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days)
            news_df = news_df[news_df['date'] >= cutoff_date]
            
            # Sort by date (newest first)
            news_df = news_df.sort_values('date', ascending=False)
            
            # Limit number of articles
            news_df = news_df.head(max_articles)
            
            # Add sentiment analysis
            news_df['sentiment'] = news_df['title'].apply(lambda x: analyze_sentiment(x)['sentiment'])
            news_df['compound'] = news_df['title'].apply(lambda x: analyze_sentiment(x)['compound'])
            
            return news_df, None
        else:
            return pd.DataFrame(), "Invalid news data format."
    except Exception as e:
        return pd.DataFrame(), f"Error fetching news: {str(e)}"

# Function to get tweets for a stock (mock function as Twitter API requires authentication)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_tweets(ticker, days=2, max_tweets=20):
    """
    Simulates getting tweets for a stock based on news data
    
    Parameters:
    ticker (str): Stock ticker symbol
    days (int): Number of days to look back
    max_tweets (int): Maximum number of tweets to return
    
    Returns:
    pandas.DataFrame: DataFrame containing tweets
    """
    # Since we can't actually use the Twitter API without authentication,
    # we'll generate simulated tweets based on news articles
    news_df, error = get_stock_news(ticker, days=days, max_articles=max_tweets)
    
    if error:
        return pd.DataFrame(), error
    
    if news_df.empty:
        return pd.DataFrame(), "No news found to generate tweet data."
    
    # Create tweet dataframe
    tweets = []
    
    for _, row in news_df.iterrows():
        # Create a simulated tweet based on the news title
        tweet_text = f"{row['title']} #{ticker.replace('.', '')}"
        
        # Analyze sentiment
        sentiment = analyze_sentiment(tweet_text)
        
        tweets.append({
            'text': tweet_text,
            'date': row['date'] - timedelta(minutes=np.random.randint(0, 60*24)),  # Random time offset
            'sentiment': sentiment['sentiment'],
            'compound': sentiment['compound'],
            'retweets': np.random.randint(0, 100),
            'likes': np.random.randint(0, 500)
        })
    
    tweets_df = pd.DataFrame(tweets)
    tweets_df = tweets_df.sort_values('date', ascending=False)
    
    return tweets_df, None

# Function to get sentiment summary
def get_sentiment_summary(ticker):
    """
    Gets sentiment summary for a stock
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    dict: Dictionary containing sentiment summary
    """
    # Get news and tweets
    news_df, news_error = get_stock_news(ticker, days=7)
    tweets_df, tweets_error = get_stock_tweets(ticker, days=2)
    
    # Initialize summary
    summary = {
        'overall_sentiment': 'neutral',
        'news_sentiment': 'neutral',
        'social_sentiment': 'neutral',
        'news_count': 0,
        'tweets_count': 0,
        'positive_pct': 0,
        'negative_pct': 0,
        'neutral_pct': 0,
        'errors': []
    }
    
    # Add any errors
    if news_error:
        summary['errors'].append(f"News error: {news_error}")
    
    if tweets_error:
        summary['errors'].append(f"Social media error: {tweets_error}")
    
    # Calculate news sentiment
    if not news_df.empty and 'sentiment' in news_df.columns:
        news_count = len(news_df)
        summary['news_count'] = news_count
        
        positive_news = len(news_df[news_df['sentiment'] == 'positive'])
        negative_news = len(news_df[news_df['sentiment'] == 'negative'])
        neutral_news = len(news_df[news_df['sentiment'] == 'neutral'])
        
        # Calculate news sentiment
        if positive_news > negative_news and positive_news > neutral_news:
            summary['news_sentiment'] = 'positive'
        elif negative_news > positive_news and negative_news > neutral_news:
            summary['news_sentiment'] = 'negative'
        else:
            summary['news_sentiment'] = 'neutral'
    
    # Calculate social media sentiment
    if not tweets_df.empty and 'sentiment' in tweets_df.columns:
        tweets_count = len(tweets_df)
        summary['tweets_count'] = tweets_count
        
        positive_tweets = len(tweets_df[tweets_df['sentiment'] == 'positive'])
        negative_tweets = len(tweets_df[tweets_df['sentiment'] == 'negative'])
        neutral_tweets = len(tweets_df[tweets_df['sentiment'] == 'neutral'])
        
        # Calculate social sentiment
        if positive_tweets > negative_tweets and positive_tweets > neutral_tweets:
            summary['social_sentiment'] = 'positive'
        elif negative_tweets > positive_tweets and negative_tweets > neutral_tweets:
            summary['social_sentiment'] = 'negative'
        else:
            summary['social_sentiment'] = 'neutral'
    
    # Calculate overall sentiment
    total_items = summary['news_count'] + summary['tweets_count']
    
    if total_items > 0:
        # Combine all sentiment data
        all_sentiments = []
        
        if not news_df.empty and 'sentiment' in news_df.columns:
            all_sentiments.extend(news_df['sentiment'].tolist())
        
        if not tweets_df.empty and 'sentiment' in tweets_df.columns:
            all_sentiments.extend(tweets_df['sentiment'].tolist())
        
        # Calculate percentages
        positive_count = all_sentiments.count('positive')
        negative_count = all_sentiments.count('negative')
        neutral_count = all_sentiments.count('neutral')
        
        summary['positive_pct'] = (positive_count / total_items) * 100
        summary['negative_pct'] = (negative_count / total_items) * 100
        summary['neutral_pct'] = (neutral_count / total_items) * 100
        
        # Determine overall sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            summary['overall_sentiment'] = 'positive'
        elif negative_count > positive_count and negative_count > neutral_count:
            summary['overall_sentiment'] = 'negative'
        else:
            summary['overall_sentiment'] = 'neutral'
    
    return summary
