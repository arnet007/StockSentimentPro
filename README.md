
# Stock Market Dashboard

A real-time stock market dashboard built with Streamlit that provides stock data visualization and sentiment analysis for Indian and global markets.

## Features

- **Stock Data Dashboard**
  - Real-time stock price tracking
  - Interactive price charts (Candlestick, Line, OHLC, Area)
  - Multiple timeframe analysis
  - Volume analysis
  - Stock comparison tools
  - Support for NSE, BSE, and US markets

- **Sentiment Analysis Dashboard**
  - News sentiment analysis
  - Social media sentiment tracking
  - Sentiment distribution visualization
  - Comprehensive sentiment metrics
  - Historical sentiment trends

## Requirements

The project uses Poetry for dependency management. Main dependencies include:

- Python 3.11+
- Streamlit
- yfinance
- pandas
- plotly
- nltk
- textblob
- numpy
- openai
- trafilatura

## Quick Start

1. Clone the repository
2. Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will be accessible at `http://0.0.0.0:5000`

## Project Structure

```
├── .streamlit/          # Streamlit configuration
├── components/          # Dashboard components
├── utils/              # Utility functions
├── app.py              # Main application
└── pyproject.toml      # Project dependencies
```

## Dashboard Components

### Data Dashboard
- Real-time stock data visualization
- Multiple chart types
- Technical indicators
- Market information

### Sentiment Dashboard
- News analysis
- Social media sentiment
- Sentiment metrics
- Trend analysis

## Configuration

The dashboard can be configured through `.streamlit/config.toml`:

- Server settings
- Theme customization
- Chart preferences
- Data refresh intervals

## License

MIT License

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Streamlit
- Sentiment analysis using NLTK and TextBlob
