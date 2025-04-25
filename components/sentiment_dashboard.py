import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.sentiment_analysis import (
    get_stock_news,
    get_stock_tweets,
    get_sentiment_summary
)
from utils.stock_data import format_ticker

# Define sentiment colors
SENTIMENT_COLORS = {
    "positive": "#2e7d32",  # Green
    "negative": "#c62828",  # Red
    "neutral": "#757575"    # Gray
}

def render_sentiment_dashboard():
    """Renders the stock sentiment analysis dashboard"""
    
    st.subheader("Stock Sentiment Analysis")
    
    # Instruction
    st.markdown("""
    This dashboard analyzes sentiment from news and social media for stocks. 
    Select a stock to see its sentiment analysis.
    """)
    
    # Stock selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Market selection for sentiment analysis
        market_options = ["India (NSE)", "India (BSE)", "US"]
        sentiment_market = st.selectbox(
            "Select Market for Sentiment Analysis", 
            market_options,
            index=0,
            key="sentiment_market_select"
        )
        
        # Default tickers based on market
        if sentiment_market == "India (NSE)":
            default_tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                "HINDUNILVR.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS", "KOTAKBANK.NS"
            ]
        elif sentiment_market == "India (BSE)":
            default_tickers = [
                "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO", 
                "HINDUNILVR.BO", "SBIN.BO", "AXISBANK.BO", "BAJFINANCE.BO", "KOTAKBANK.BO"
            ]
        else:  # US
            default_tickers = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
                "TSLA", "NVDA", "BRK-B", "JPM", "JNJ"
            ]
        
        # Ticker selection
        sentiment_ticker = st.selectbox(
            "Select Stock", 
            default_tickers,
            index=0,
            key="sentiment_ticker_select"
        )
        
        # Custom ticker input
        custom_sentiment_ticker = st.text_input(
            "Or enter custom ticker symbol", 
            value="",
            help="For NSE stocks, add .NS suffix. For BSE stocks, add .BO suffix.",
            key="custom_sentiment_ticker"
        )
        
        if custom_sentiment_ticker:
            sentiment_ticker = custom_sentiment_ticker
            
        # Data source options
        st.subheader("Data Sources")
        show_news = st.checkbox("Show News", value=True, key="show_news")
        show_social = st.checkbox("Show Social Media", value=True, key="show_social")
        
        # Time frame
        st.subheader("Time Frame")
        days_options = [1, 3, 7, 14, 30]
        days_back = st.select_slider(
            "Days to look back", 
            options=days_options,
            value=7,
            key="days_back_slider"
        )
        
        # Analyze button
        analyze_button = st.button("Analyze Sentiment", key="analyze_sentiment_button")
        
        if analyze_button or 'sentiment_summary' not in st.session_state:
            with st.spinner("Analyzing sentiment..."):
                # Get sentiment summary
                sentiment_summary = get_sentiment_summary(sentiment_ticker)
                st.session_state.sentiment_summary = sentiment_summary
                
                # Get news and tweets
                news_df, news_error = get_stock_news(sentiment_ticker, days=days_back, max_articles=20)
                tweets_df, tweets_error = get_stock_tweets(sentiment_ticker, days=days_back, max_tweets=30)
                
                st.session_state.news_df = news_df
                st.session_state.news_error = news_error
                st.session_state.tweets_df = tweets_df
                st.session_state.tweets_error = tweets_error
    
    with col2:
        if 'sentiment_summary' in st.session_state:
            summary = st.session_state.sentiment_summary
            
            # Add custom CSS for better text handling
            st.markdown("""
            <style>
            .sentiment-card {
                border-radius: 10px;
                padding: 10px;
                color: white;
                text-align: center;
                margin-bottom: 10px;
                word-wrap: break-word;
            }
            .sentiment-card h3 {
                font-size: 1.1rem;
                margin-bottom: 5px;
                white-space: normal;
                word-wrap: break-word;
            }
            .sentiment-card h2 {
                font-size: 1.4rem;
                margin-top: 0;
                white-space: normal;
                word-wrap: break-word;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display overall sentiment with improved layout
            sentiment_col1, sentiment_col2, sentiment_col3 = st.columns([1, 1, 1])
            
            with sentiment_col1:
                overall_color = SENTIMENT_COLORS.get(summary['overall_sentiment'], SENTIMENT_COLORS['neutral'])
                st.markdown(f"""
                <div class="sentiment-card" style="background-color:{overall_color};">
                    <h3>Overall Sentiment</h3>
                    <h2>{summary['overall_sentiment'].upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with sentiment_col2:
                news_color = SENTIMENT_COLORS.get(summary['news_sentiment'], SENTIMENT_COLORS['neutral'])
                st.markdown(f"""
                <div class="sentiment-card" style="background-color:{news_color};">
                    <h3>News Sentiment</h3>
                    <h2>{summary['news_sentiment'].upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with sentiment_col3:
                social_color = SENTIMENT_COLORS.get(summary['social_sentiment'], SENTIMENT_COLORS['neutral'])
                st.markdown(f"""
                <div class="sentiment-card" style="background-color:{social_color};">
                    <h3>Social Media</h3>
                    <h2>{summary['social_sentiment'].upper()}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            # Display sentiment distribution
            st.subheader("Sentiment Distribution")
            
            # Create pie chart of sentiment distribution
            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[
                    summary['positive_pct'],
                    summary['negative_pct'],
                    summary['neutral_pct']
                ],
                hole=.3,
                marker_colors=[
                    SENTIMENT_COLORS['positive'],
                    SENTIMENT_COLORS['negative'],
                    SENTIMENT_COLORS['neutral']
                ]
            )])
            
            fig.update_layout(
                title=f"Sentiment Distribution for {sentiment_ticker}",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display errors if any
            if summary['errors']:
                with st.expander("Errors"):
                    for error in summary['errors']:
                        st.warning(error)
    
    # Display news and tweets in tabs
    if 'news_df' in st.session_state and 'tweets_df' in st.session_state:
        tab1, tab2 = st.tabs(["News Analysis", "Social Media Analysis"])
        
        with tab1:
            if show_news:
                news_df = st.session_state.news_df
                news_error = st.session_state.news_error
                
                if news_error:
                    st.warning(news_error)
                    
                if not news_df.empty:
                    st.subheader(f"News Sentiment Analysis for {sentiment_ticker}")
                    # Show more articles for better analysis
                    article_count = len(news_df)
                    st.write(f"Showing {article_count} news articles from the past {days_back} days")
                    
                    # Add info about sentiment distribution
                    positive_count = len(news_df[news_df['sentiment'] == 'positive'])
                    negative_count = len(news_df[news_df['sentiment'] == 'negative'])
                    neutral_count = len(news_df[news_df['sentiment'] == 'neutral'])
                    
                    st.write(f"Sentiment distribution: {positive_count} positive, {negative_count} negative, {neutral_count} neutral articles")
                    
                    # News sentiment over time
                    if 'date' in news_df.columns and 'compound' in news_df.columns:
                        # Sort by date
                        news_df = news_df.sort_values('date')
                        
                        # Create line chart of sentiment over time
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=news_df['date'],
                            y=news_df['compound'],
                            mode='lines+markers',
                            name='Sentiment Score',
                            marker=dict(
                                color=news_df['compound'].apply(
                                    lambda x: SENTIMENT_COLORS['positive'] if x > 0.05 
                                             else SENTIMENT_COLORS['negative'] if x < -0.05 
                                             else SENTIMENT_COLORS['neutral']
                                )
                            )
                        ))
                        
                        fig.update_layout(
                            title="News Sentiment Over Time",
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score (-1 to 1)",
                            height=300,
                            yaxis=dict(range=[-1, 1])
                        )
                        
                        # Add horizontal lines for reference
                        fig.add_shape(type="line", x0=news_df['date'].min(), y0=0, x1=news_df['date'].max(), y1=0,
                                    line=dict(color="gray", width=1, dash="dash"))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # News table
                    st.subheader("Recent News Articles")
                    
                    # Create a cleaner table layout for news articles
                    st.markdown("""
                    <style>
                    .news-item {
                        border-left: 5px solid;
                        padding-left: 10px;
                        margin-bottom: 15px;
                        background-color: #f9f9f9;
                        border-radius: 3px;
                        padding: 10px 10px 10px 15px;
                        word-wrap: break-word;
                    }
                    .news-title {
                        font-size: 1.1rem;
                        font-weight: bold;
                        margin-bottom: 5px;
                        word-wrap: break-word;
                    }
                    .news-meta {
                        font-size: 0.85rem;
                        color: #666;
                        margin-top: 5px;
                        margin-bottom: 5px;
                    }
                    .news-sentiment {
                        font-weight: bold;
                        padding: 2px 6px;
                        border-radius: 3px;
                        color: white;
                        display: inline-block;
                        margin-right: 5px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    for _, row in news_df.iterrows():
                        # Determine sentiment color
                        sentiment_color = SENTIMENT_COLORS.get(row['sentiment'], SENTIMENT_COLORS['neutral'])
                        sentiment_text = row['sentiment'].upper()
                        
                        # Format date safely
                        try:
                            date_str = row['date'].strftime('%Y-%m-%d %H:%M')
                        except:
                            date_str = "N/A"
                        
                        # Safe link handling
                        link = row.get('link', '#')
                        if not link or link == '':
                            link = '#'
                            link_text = ''
                        else:
                            link_text = f'<a href="{link}" target="_blank">Read more</a>'
                        
                        st.markdown(f"""
                        <div class="news-item" style="border-left-color: {sentiment_color};">
                            <div class="news-title">{row['title']}</div>
                            <div class="news-meta">
                                <span class="news-sentiment" style="background-color: {sentiment_color};">{sentiment_text}</span>
                                <strong>Source:</strong> {row.get('publisher', 'Unknown')} | <strong>Date:</strong> {date_str}
                            </div>
                            <div>{link_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No news articles found for {sentiment_ticker} in the past {days_back} days.")
            else:
                st.info("Enable 'Show News' to see news sentiment analysis.")
                
        with tab2:
            if show_social:
                tweets_df = st.session_state.tweets_df
                tweets_error = st.session_state.tweets_error
                
                if tweets_error:
                    st.warning(tweets_error)
                    
                if not tweets_df.empty:
                    st.subheader(f"Social Media Sentiment Analysis for {sentiment_ticker}")
                    
                    # Show comprehensive post count information
                    tweet_count = len(tweets_df)
                    st.write(f"Showing {tweet_count} social media posts from the past {days_back} days")
                    
                    # Add info about sentiment distribution
                    positive_count = len(tweets_df[tweets_df['sentiment'] == 'positive'])
                    negative_count = len(tweets_df[tweets_df['sentiment'] == 'negative'])
                    neutral_count = len(tweets_df[tweets_df['sentiment'] == 'neutral'])
                    
                    st.write(f"Sentiment distribution: {positive_count} positive, {negative_count} negative, {neutral_count} neutral posts")
                    
                    # Social sentiment over time
                    if 'date' in tweets_df.columns and 'compound' in tweets_df.columns:
                        # Sort by date
                        tweets_df = tweets_df.sort_values('date')
                        
                        # Create line chart of sentiment over time
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=tweets_df['date'],
                            y=tweets_df['compound'],
                            mode='lines+markers',
                            name='Sentiment Score',
                            marker=dict(
                                color=tweets_df['compound'].apply(
                                    lambda x: SENTIMENT_COLORS['positive'] if x > 0.05 
                                             else SENTIMENT_COLORS['negative'] if x < -0.05 
                                             else SENTIMENT_COLORS['neutral']
                                )
                            )
                        ))
                        
                        fig.update_layout(
                            title="Social Media Sentiment Over Time",
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score (-1 to 1)",
                            height=300,
                            yaxis=dict(range=[-1, 1])
                        )
                        
                        # Add horizontal lines for reference
                        fig.add_shape(type="line", x0=tweets_df['date'].min(), y0=0, x1=tweets_df['date'].max(), y1=0,
                                    line=dict(color="gray", width=1, dash="dash"))
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Sentiment distribution by engagement
                    st.subheader("Sentiment by Engagement")
                    
                    # Create bubble chart
                    fig = px.scatter(
                        tweets_df,
                        x='retweets',
                        y='likes',
                        size='likes',
                        color='sentiment',
                        color_discrete_map=SENTIMENT_COLORS,
                        hover_name='text',
                        size_max=40,
                        opacity=0.7
                    )
                    
                    fig.update_layout(
                        title="Sentiment by Engagement",
                        xaxis_title="Retweets",
                        yaxis_title="Likes",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Social media posts
                    st.subheader("Recent Social Media Posts")
                    
                    # Sort by engagement (likes + retweets)
                    tweets_df['engagement'] = tweets_df['likes'] + tweets_df['retweets']
                    sorted_tweets = tweets_df.sort_values('engagement', ascending=False)
                    
                    # Show more social media posts (up to 50 instead of just 10)
                    for _, row in sorted_tweets.head(50).iterrows():
                        # Determine sentiment color
                        sentiment_color = SENTIMENT_COLORS.get(row['sentiment'], SENTIMENT_COLORS['neutral'])
                        sentiment_text = row['sentiment'].upper()
                        
                        # Format date safely
                        try:
                            date_str = row['date'].strftime('%Y-%m-%d %H:%M')
                        except:
                            date_str = "N/A"
                        
                        st.markdown(f"""
                        <div class="news-item" style="border-left-color: {sentiment_color};">
                            <div class="news-title">{row['text']}</div>
                            <div class="news-meta">
                                <span class="news-sentiment" style="background-color: {sentiment_color};">{sentiment_text}</span>
                                <strong>Date:</strong> {date_str} | <strong>Likes:</strong> {row['likes']} | <strong>Retweets:</strong> {row['retweets']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No social media posts found for {sentiment_ticker} in the past {days_back} days.")
            else:
                st.info("Enable 'Show Social Media' to see social media sentiment analysis.")
