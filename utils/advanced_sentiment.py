import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# Download NLTK data if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """
    Preprocess the text for better sentiment analysis
    
    Parameters:
    text (str): Text to preprocess
    
    Returns:
    str: Preprocessed text
    """
    # Convert to string if not already
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def enhanced_sentiment_analysis(text):
    """
    Performs enhanced sentiment analysis using a combination of models
    
    Parameters:
    text (str): Text to analyze
    
    Returns:
    dict: Dictionary containing sentiment scores and labels
    """
    if not text or not isinstance(text, str):
        return {
            'compound': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 1,
            'sentiment': 'neutral'
        }
    
    # Preprocess the text
    text = preprocess_text(text)
    
    # Use TextBlob for sentiment analysis
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity
    
    # Use VADER for sentiment analysis
    vader_scores = sia.polarity_scores(text)
    
    # Combine the scores (weighting VADER more heavily as it's better for social media)
    compound_score = vader_scores['compound'] * 0.7 + textblob_polarity * 0.3
    
    # Determine sentiment label
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
        
    return {
        'compound': compound_score,
        'positive': vader_scores['pos'],
        'negative': vader_scores['neg'],
        'neutral': vader_scores['neu'],
        'subjectivity': textblob_subjectivity,
        'sentiment': sentiment
    }

def get_sentiment_stats(df, text_column='text'):
    """
    Calculate sentiment statistics for a dataframe
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing text data
    text_column (str): Column name containing the text
    
    Returns:
    dict: Dictionary containing sentiment statistics
    """
    if df.empty or text_column not in df.columns:
        return {
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'total_count': 0,
            'positive_pct': 0,
            'negative_pct': 0,
            'neutral_pct': 0,
            'avg_compound': 0,
            'avg_positive': 0,
            'avg_negative': 0,
            'avg_neutral': 0,
            'primary_sentiment': 'neutral'
        }
    
    # Apply sentiment analysis to each text
    sentiments = df[text_column].apply(enhanced_sentiment_analysis)
    
    # Extract sentiment labels and scores
    df['sentiment'] = sentiments.apply(lambda x: x['sentiment'])
    df['compound'] = sentiments.apply(lambda x: x['compound'])
    df['positive'] = sentiments.apply(lambda x: x['positive'])
    df['negative'] = sentiments.apply(lambda x: x['negative'])
    df['neutral'] = sentiments.apply(lambda x: x['neutral'])
    
    # Calculate counts
    total_count = len(df)
    positive_count = len(df[df['sentiment'] == 'positive'])
    negative_count = len(df[df['sentiment'] == 'negative'])
    neutral_count = len(df[df['sentiment'] == 'neutral'])
    
    # Calculate percentages
    positive_pct = (positive_count / total_count) * 100 if total_count > 0 else 0
    negative_pct = (negative_count / total_count) * 100 if total_count > 0 else 0
    neutral_pct = (neutral_count / total_count) * 100 if total_count > 0 else 0
    
    # Calculate averages
    avg_compound = df['compound'].mean() if 'compound' in df.columns else 0
    avg_positive = df['positive'].mean() if 'positive' in df.columns else 0
    avg_negative = df['negative'].mean() if 'negative' in df.columns else 0
    avg_neutral = df['neutral'].mean() if 'neutral' in df.columns else 0
    
    # Determine primary sentiment
    if positive_count > negative_count and positive_count > neutral_count:
        primary_sentiment = 'positive'
    elif negative_count > positive_count and negative_count > neutral_count:
        primary_sentiment = 'negative'
    else:
        primary_sentiment = 'neutral'
    
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'total_count': total_count,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct,
        'neutral_pct': neutral_pct,
        'avg_compound': avg_compound,
        'avg_positive': avg_positive,
        'avg_negative': avg_negative,
        'avg_neutral': avg_neutral,
        'primary_sentiment': primary_sentiment
    }