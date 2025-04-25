import yfinance as yf
import pandas as pd
import time
import streamlit as st
from datetime import datetime, timedelta

# Cache stock data to minimize API calls
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(ticker, period="1mo", interval="1d"):
    """
    Fetches stock data from Yahoo Finance API
    
    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period to fetch data for (1d, 5d, 1mo, 6mo, ytd, 1y, 5y, max)
    interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
    pandas.DataFrame: DataFrame with stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None, "No data found for this ticker. Please check the symbol."
        
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Get stock information
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_stock_info(ticker):
    """
    Fetches stock information like company name, market cap, P/E ratio, etc.
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    dict: Dictionary containing stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        stock_info = {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "peRatio": info.get("trailingPE", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "logo_url": info.get("logo_url", None),
        }
        
        # Format market cap in trillions/billions/millions
        if stock_info["marketCap"] != "N/A" and stock_info["marketCap"] is not None:
            market_cap = float(stock_info["marketCap"])
            if market_cap >= 1e12:
                stock_info["marketCapFormatted"] = f"{market_cap/1e12:.2f} T"
            elif market_cap >= 1e9:
                stock_info["marketCapFormatted"] = f"{market_cap/1e9:.2f} B"
            elif market_cap >= 1e6:
                stock_info["marketCapFormatted"] = f"{market_cap/1e6:.2f} M"
            else:
                stock_info["marketCapFormatted"] = f"{market_cap:.2f}"
        else:
            stock_info["marketCapFormatted"] = "N/A"
            
        # Format dividend yield as percentage
        if stock_info["dividendYield"] != "N/A" and stock_info["dividendYield"] is not None:
            stock_info["dividendYieldFormatted"] = f"{stock_info['dividendYield']*100:.2f}%"
        else:
            stock_info["dividendYieldFormatted"] = "N/A"
            
        return stock_info, None
    except Exception as e:
        return None, f"Error fetching stock information: {str(e)}"

# Search for stocks
@st.cache_data(ttl=86400)  # Cache for 24 hours
def search_stocks(query):
    """
    Searches for stocks based on a query string
    
    Parameters:
    query (str): Search query
    
    Returns:
    list: List of matching stock tickers
    """
    try:
        tickers = yf.Tickers(query)
        matching_tickers = []
        
        for ticker in tickers.tickers:
            info = ticker.info
            if 'symbol' in info:
                matching_tickers.append({
                    'symbol': info['symbol'],
                    'name': info.get('longName', info.get('shortName', 'Unknown')),
                    'exchange': info.get('exchange', 'Unknown')
                })
        
        return matching_tickers, None
    except Exception as e:
        return [], f"Error searching for stocks: {str(e)}"

# Map period to time delta for date range
def map_period_to_delta(period):
    """
    Maps a period string to a time delta
    
    Parameters:
    period (str): Period string (1d, 5d, 1mo, 6mo, ytd, 1y, 5y, max)
    
    Returns:
    datetime.timedelta: Time delta corresponding to the period
    """
    today = datetime.now()
    
    if period == "1d":
        return timedelta(days=1)
    elif period == "5d":
        return timedelta(days=5)
    elif period == "1mo":
        return timedelta(days=30)
    elif period == "6mo":
        return timedelta(days=180)
    elif period == "ytd":
        return today - datetime(today.year, 1, 1)
    elif period == "1y":
        return timedelta(days=365)
    elif period == "5y":
        return timedelta(days=5*365)
    elif period == "max":
        return timedelta(days=50*365)  # Arbitrary large value
    else:
        return timedelta(days=30)  # Default to 1 month

# Get comparable stocks
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_comparable_stocks(ticker):
    """
    Gets comparable stocks based on sector/industry
    
    Parameters:
    ticker (str): Stock ticker symbol
    
    Returns:
    list: List of comparable stock tickers
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        sector = info.get('sector', None)
        industry = info.get('industry', None)
        
        if not sector or not industry:
            return [], "Sector/industry information not available for this stock."
        
        # For simplicity, we'll return a predefined list of stocks for each exchange
        # In a real implementation, this would be more sophisticated
        if "NSE" in ticker or ".NS" in ticker:
            return [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"
            ], None
        elif "BSE" in ticker or ".BO" in ticker:
            return [
                "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO"
            ], None
        else:
            return [
                "AAPL", "MSFT", "AMZN", "GOOGL", "FB"
            ], None
    except Exception as e:
        return [], f"Error getting comparable stocks: {str(e)}"

# Format stock ticker for Yahoo Finance
def format_ticker(ticker, exchange=None):
    """
    Formats a stock ticker symbol for Yahoo Finance
    
    Parameters:
    ticker (str): Raw ticker symbol
    exchange (str): Exchange (NSE, BSE, etc.)
    
    Returns:
    str: Formatted ticker symbol
    """
    ticker = ticker.upper().strip()
    
    # Check if already formatted
    if ".NS" in ticker or ".BO" in ticker or "^" in ticker:
        return ticker
    
    # Format based on exchange
    if exchange:
        if exchange.upper() == "NSE":
            return f"{ticker}.NS"
        elif exchange.upper() == "BSE":
            return f"{ticker}.BO"
            
    # Default to no suffix
    return ticker
