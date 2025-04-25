import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from utils.stock_data import (
    get_stock_data, 
    get_stock_info, 
    search_stocks, 
    format_ticker, 
    map_period_to_delta,
    get_comparable_stocks
)

# Default tickers for different markets
DEFAULT_TICKERS = {
    "India (NSE)": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "HDFC.NS", "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS"
    ],
    "India (BSE)": [
        "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO", 
        "HDFC.BO", "HINDUNILVR.BO", "SBIN.BO", "BAJFINANCE.BO", "KOTAKBANK.BO"
    ],
    "US": [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "TSLA", "NVDA", "BRK-B", "JPM", "JNJ"
    ],
    "Indices": [
        "^NSEI", "^BSESN", "^GSPC", "^DJI", "^IXIC", 
        "^FTSE", "^N225", "^HSI", "^GDAXI", "^FCHI"
    ]
}

# Define time periods available
TIME_PERIODS = {
    "1D": {"period": "1d", "interval": "5m"},
    "5D": {"period": "5d", "interval": "15m"},
    "1M": {"period": "1mo", "interval": "1d"},
    "6M": {"period": "6mo", "interval": "1d"},
    "YTD": {"period": "ytd", "interval": "1d"},
    "1Y": {"period": "1y", "interval": "1d"},
    "5Y": {"period": "5y", "interval": "1wk"},
    "MAX": {"period": "max", "interval": "1mo"}
}

# Define chart types
CHART_TYPES = ["Candlestick", "Line", "OHLC", "Area"]

def render_data_dashboard():
    """Renders the stock data dashboard"""

    # Set up layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Select Stock")
        
        # Market selection
        market = st.selectbox(
            "Select Market", 
            list(DEFAULT_TICKERS.keys()),
            index=0,
            key="market_select"
        )
        
        # Ticker selection
        selected_ticker = st.selectbox(
            "Select Stock", 
            DEFAULT_TICKERS[market],
            index=0,
            key="ticker_select"
        )
        
        # Custom ticker input
        custom_ticker = st.text_input(
            "Or enter custom ticker symbol", 
            value="",
            help="For NSE stocks, add .NS suffix (e.g., RELIANCE.NS). For BSE stocks, add .BO suffix."
        )
        
        if custom_ticker:
            selected_ticker = custom_ticker
            
        # Time period selection
        period_key = st.selectbox(
            "Select Time Period", 
            list(TIME_PERIODS.keys()),
            index=2,  # Default to 1M
            key="period_select"
        )
        
        period_config = TIME_PERIODS[period_key]
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select Chart Type", 
            CHART_TYPES,
            index=0,
            key="chart_type_select"
        )
        
        # Get stock data and info
        with st.spinner("Fetching stock data..."):
            data, error = get_stock_data(
                selected_ticker, 
                period=period_config["period"], 
                interval=period_config["interval"]
            )
            
            if error:
                st.error(error)
                return
                
            info, info_error = get_stock_info(selected_ticker)
            
            if info_error:
                st.warning(info_error)
                
        # Display stock information
        if info:
            # Use markdown for company name to ensure it wraps properly
            st.markdown(f"### {info.get('name', selected_ticker)}")
            
            # Use 2 columns for metrics with better text wrapping
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric(
                    "Market Cap", 
                    info.get("marketCapFormatted", "N/A"),
                    help="Total market value of the company's outstanding shares"
                )
                st.metric(
                    "P/E Ratio", 
                    f"{info.get('peRatio', 'N/A')}",
                    help="Price to Earnings ratio"
                )
                
            with metrics_col2:
                # Format the 52-week high to ensure it fits
                fiftyTwoWeekHighValue = info.get('fiftyTwoWeekHigh', 'N/A')
                currency = info.get('currency', '')
                st.metric(
                    "52W High", 
                    f"{fiftyTwoWeekHighValue} {currency}" if fiftyTwoWeekHighValue != "N/A" else "N/A",
                    help="Highest price in the last 52 weeks"
                )
                st.metric(
                    "Dividend Yield", 
                    info.get("dividendYieldFormatted", "N/A"),
                    help="Annual dividend as percentage of share price"
                )
            
            # Use more compact display for additional information
            st.markdown(f"<div style='font-size:0.9em'>Exchange: <b>{info.get('exchange', 'N/A')}</b> | Sector: <b>{info.get('sector', 'N/A')}</b></div>", 
                        unsafe_allow_html=True)
            
            # Comparison section
            st.subheader("Compare with")
            comparable_stocks, comp_error = get_comparable_stocks(selected_ticker)
            
            if comp_error:
                st.warning(comp_error)
            
            if comparable_stocks:
                compare_with = st.multiselect(
                    "Select stocks to compare", 
                    comparable_stocks,
                    default=[comparable_stocks[0]] if comparable_stocks else [],
                    key="compare_select"
                )
                
    with col2:
        # Main chart area
        if data is not None and not data.empty:
            st.subheader(f"{info.get('name', selected_ticker)} Stock Chart")
            
            # Get the last price and calculate change
            last_close = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else data['Close'].iloc[0]
            price_change = last_close - prev_close
            price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
            
            # Display current price and change
            price_col, change_col, _, vol_col = st.columns([1, 1, 1, 1])
            
            with price_col:
                st.metric(
                    "Current Price", 
                    f"{last_close:.2f} {info.get('currency', '')}",
                )
                
            with change_col:
                st.metric(
                    "Change", 
                    f"{price_change:.2f} ({price_change_pct:.2f}%)",
                    delta=price_change
                )
                
            with vol_col:
                if 'Volume' in data.columns:
                    last_volume = data['Volume'].iloc[-1]
                    vol_str = f"{last_volume:,.0f}"
                    st.metric("Volume", vol_str)
            
            # Create figure
            fig = None
            
            if chart_type == "Candlestick":
                fig = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=selected_ticker
                )])
            elif chart_type == "Line":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['Close'],
                    mode='lines',
                    name=selected_ticker
                ))
            elif chart_type == "OHLC":
                fig = go.Figure(data=[go.Ohlc(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=selected_ticker
                )])
            elif chart_type == "Area":
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=data['Close'],
                    fill='tozeroy',
                    name=selected_ticker
                ))
                
            # Add comparison stocks if selected
            if 'compare_with' in locals() and compare_with:
                for comp_ticker in compare_with:
                    comp_data, comp_error = get_stock_data(
                        comp_ticker,
                        period=period_config["period"],
                        interval=period_config["interval"]
                    )
                    
                    if comp_error or comp_data is None or comp_data.empty:
                        st.warning(f"Could not fetch data for {comp_ticker}: {comp_error}")
                        continue
                        
                    if chart_type == "Line" or chart_type == "Area":
                        # Normalize data for comparison (first value = 100)
                        base_value = comp_data['Close'].iloc[0]
                        normalized_data = (comp_data['Close'] / base_value) * 100
                        
                        fig.add_trace(go.Scatter(
                            x=comp_data.index,
                            y=normalized_data,
                            mode='lines',
                            name=comp_ticker
                        ))
            
            # Update layout
            fig.update_layout(
                title=f"{info.get('name', selected_ticker)} - {period_key} Chart",
                xaxis_title="Date",
                yaxis_title=f"Price ({info.get('currency', '')})",
                height=600,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display volume chart
            if 'Volume' in data.columns:
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume"
                ))
                
                volume_fig.update_layout(
                    title="Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=250
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
        else:
            st.warning("No data available for the selected stock and time period.")
