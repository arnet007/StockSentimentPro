import streamlit as st
from components.data_dashboard import render_data_dashboard
from components.sentiment_dashboard import render_sentiment_dashboard
import time

st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

def main():
    # App title and description
    st.title("Stock Market Dashboard")
    st.markdown("""
    This dashboard provides real-time stock data and sentiment analysis for Indian and global markets.
    """)
    
    # Tabs for different dashboard views
    tab1, tab2 = st.tabs(["Stock Data", "Sentiment Analysis"])
    
    with tab1:
        render_data_dashboard()
    
    with tab2:
        render_sentiment_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>Data sourced from Yahoo Finance. Sentiment analysis from various public sources.</p>
        <p>© 2023 Stock Market Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
