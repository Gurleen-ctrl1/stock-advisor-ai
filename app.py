import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
import requests
import time
import random
from transformers import pipeline  # For local Hugging Face models

# Set page configuration
st.set_page_config(
    page_title="Stock Recommender & Advisor",
    page_icon="💰",
    layout="wide"
)

# Enhanced Stock Data Manager Class
class StockDataManager:
    def __init__(self):
        self.request_delay = 1.5  # Base delay for yfinance
        self.max_retries = 3
        self.backoff_factor = 2
        self.last_request_time = 0

    def wait_for_rate_limit(self):
        """Implement smart rate limiting for yfinance"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        delay = self.request_delay

        if time_since_last < delay:
            sleep_time = delay - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def validate_ticker(self, ticker):
        """Quickly validate if a ticker exists by attempting to fetch basic info."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info  # Fetch basic info to validate ticker
            return True
        except Exception:
            return False

    def get_stock_data_with_retry(self, ticker, period="1y"):
        """Fetch stock data from yfinance with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.wait_for_rate_limit()
                if attempt > 0:
                    jitter = random.uniform(0.5, 1.5)
                    time.sleep(attempt * self.backoff_factor + jitter)

                stock = yf.Ticker(ticker)
                history = stock.history(period=period)
                if history.empty:
                    st.warning(f"No historical data available for {ticker} with period {period}")
                    return None

                try:
                    income_statement = stock.income_stmt
                    balance_sheet = stock.balance_sheet
                    cash_flow = stock.cashflow
                    info = stock.info
                except Exception:
                    st.warning(f"Some financial data for {ticker} unavailable: Limited data will be shown")
                    income_statement = pd.DataFrame()
                    balance_sheet = pd.DataFrame()
                    cash_flow = pd.DataFrame()
                    info = {}

                return {
                    "history": history,
                    "income_statement": income_statement,
                    "balance_sheet": balance_sheet,
                    "cash_flow": cash_flow,
                    "info": info
                }

            except Exception as e:
                error_msg = str(e).lower()
                st.warning(f"Exception occurred: {e}")  # Debug logging
                if "too many requests" in error_msg or "rate limit" in error_msg:
                    wait_time = (2 ** attempt) * self.backoff_factor + random.uniform(1, 3)
                    st.warning(f"Rate limited for {ticker}. Waiting {wait_time:.1f} seconds before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                elif "not found" in error_msg or "invalid" in error_msg:
                    st.error(f"Ticker {ticker} not found")
                    return None
                else:
                    if attempt < self.max_retries - 1:
                        st.warning(f"Error fetching {ticker}: {e}. Retrying...")
                        time.sleep((attempt + 1) * 2)
                        continue
                    st.error(f"Error fetching {ticker}: {e}")
                    return None

        st.error(f"Failed to fetch data for {ticker} after {self.max_retries} attempts")
        return None

# Initialize the stock data manager
@st.cache_resource
def get_stock_manager():
    return StockDataManager()

# Define navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Stock Analysis", "Financial Advisor Chat"])

# Currency conversion for Stock Analysis page
if 'currency' not in st.session_state:
    st.session_state.currency = 'USD'

def get_exchange_rate(to_currency='INR'):
    """Fetch exchange rate for the specified currency from USD."""
    try:
        if f'exchange_rate_{to_currency}' not in st.session_state:
            response = requests.get("https://open.er-api.com/v6/latest/USD")
            data = response.json()
            if 'rates' in data and to_currency in data['rates']:
                st.session_state[f'exchange_rate_{to_currency}'] = data['rates'][to_currency]
            else:
                raise ValueError(f"Currency {to_currency} not found in API response")
        return st.session_state[f'exchange_rate_{to_currency}']
    except Exception as e:
        st.warning(f"Using approximate exchange rate for {to_currency} due to API error: {e}")
        # Updated fallback rates (approximate, as of June 2025)
        fallback_rates = {
            'INR': 83.5,  # Updated to reflect slight INR depreciation
            'EUR': 0.93,  # Adjusted based on assumed June 2025 rate
            'AUD': 1.52   # Adjusted based on assumed June 2025 rate
        }
        return fallback_rates.get(to_currency, 1.0)

def convert_currency(amount, to_currency='USD', from_currency='USD'):
    """Convert amount from the specified currency to the target currency."""
    if to_currency == from_currency or not isinstance(amount, (int, float)):
        return amount
    # Convert from_currency to USD first
    if from_currency != 'USD':
        rate_to_usd = 1 / get_exchange_rate(from_currency)
        amount_usd = amount * rate_to_usd
    else:
        amount_usd = amount
    # Convert from USD to to_currency
    if to_currency == 'USD':
        return amount_usd
    exchange_rate = get_exchange_rate(to_currency)
    return amount_usd * exchange_rate

def format_currency(amount, currency='USD'):
    """Format amount with the appropriate currency symbol."""
    currency_symbols = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'AUD': 'A$'
    }
    symbol = currency_symbols.get(currency, '')
    if isinstance(amount, (int, float)):
        if abs(amount) >= 1_000_000_000:  # Billions
            return f"{symbol}{amount/1_000_000_000:,.2f}B"
        elif abs(amount) >= 1_000_000:  # Millions
            return f"{symbol}{amount/1_000_000:,.2f}M"
        return f"{symbol}{amount:,.2f}"
    return f"{symbol}{amount}"

if page == "Stock Analysis":
    st.sidebar.title("Currency Settings")
    currency_option = st.sidebar.selectbox(
        "Select Currency",
        ("USD", "INR", "EUR", "AUD"),
        index=['USD', 'INR', 'EUR', 'AUD'].index(st.session_state.currency)
    )

    if currency_option != st.session_state.currency:
        st.session_state.currency = currency_option
        st.rerun()

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the LLM for financial advice
@st.cache_resource
from transformers import pipeline

def load_model():
    try:
        return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        print("⚠️ TinyLlama failed, switching to distilgpt2:", e)
        return pipeline("text-generation", model="distilgpt2")


# Stock Analysis Functions
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data with retry logic using yfinance."""
    manager = get_stock_manager()
    return manager.get_stock_data_with_retry(ticker, period)

def calculate_financial_metrics(stock_data):
    """Calculate financial metrics."""
    metrics = {}

    if stock_data is None:
        return metrics

    history = stock_data["history"]
    if not history.empty:
        try:
            current_price = history['Close'].iloc[-1]
            metrics["Current Price"] = round(current_price, 2)

            if len(history) > 20:
                start_price = history['Close'].iloc[0]
                price_change = ((current_price - start_price) / start_price) * 100
                metrics["Price Change (%)"] = round(price_change, 2)

                high_52week = history['High'].max()
                low_52week = history['Low'].min()
                metrics["52W High"] = round(high_52week, 2)
                metrics["52W Low"] = round(low_52week, 2)

                avg_volume = history['Volume'].mean()
                metrics["Avg Volume"] = int(avg_volume)

        except Exception as e:
            st.warning(f"Error calculating price metrics: {e}")

    info = stock_data.get("info", {})
    if info:
        try:
            # Handle Market Cap
            if info.get("marketCap"):
                metrics["Market Cap"] = info["marketCap"]

            if info.get("trailingPE"):
                metrics["P/E Ratio"] = round(info["trailingPE"], 2)

            if info.get("dividendYield"):
                metrics["Dividend Yield (%)"] = round(info["dividendYield"] * 100, 2)

            if info.get("beta"):
                metrics["Beta"] = round(info["beta"], 2)

        except Exception:
            pass

    # yfinance-specific financials
    income = stock_data["income_statement"]
    balance = stock_data["balance_sheet"]

    if not income.empty and not balance.empty:
        try:
            latest_income = income.iloc[:, 0]
            latest_balance = balance.iloc[:, 0]

            if 'Total Revenue' in latest_income:
                revenue = latest_income['Total Revenue']
                metrics["Revenue"] = revenue
            elif info.get("totalRevenue"):
                revenue = info["totalRevenue"]
                metrics["Revenue"] = revenue
            else:
                st.warning("Revenue data unavailable or incomplete in yfinance. Using approximate value.")
                metrics["Revenue"] = "N/A"

            if 'Net Income' in latest_income:
                net_income = latest_income['Net Income']
                metrics["Net Income"] = net_income

                if revenue > 0 and isinstance(revenue, (int, float)):
                    net_margin = (net_income / revenue) * 100
                    metrics["Net Margin (%)"] = round(net_margin, 2)

            if 'Net Income' in latest_income and 'Total Stockholder Equity' in latest_balance:
                net_income = latest_income['Net Income']
                equity = latest_balance['Total Stockholder Equity']
                if equity > 0:
                    roe = (net_income / equity) * 100
                    metrics["ROE (%)"] = round(roe, 2)

            if 'Total Debt' in latest_balance and 'Total Stockholder Equity' in latest_balance:
                debt = latest_balance.get('Total Debt', 0)
                equity = latest_balance['Total Stockholder Equity']
                if equity > 0:
                    debt_to_equity = (debt / equity) * 100
                    metrics["Debt-to-Equity (%)"] = round(debt_to_equity, 2)

        except Exception:
            pass

    return metrics

@st.cache_data(ttl=3600)
def plot_stock_price_history(histories, tickers, currency='USD', native_currencies=None):
    """Plot the stock price history for multiple stocks."""
    if not histories or not tickers:
        return None

    fig = go.Figure()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # Colors for multiple stocks

    for idx, (ticker, history) in enumerate(zip(tickers, histories)):
        if history is None or history.empty:
            continue

        close_prices = history['Close']
        native_currency = native_currencies[idx]
        if currency != native_currency:
            close_prices = close_prices.apply(lambda x: convert_currency(x, currency, native_currency))

        fig.add_trace(go.Scatter(
            x=history.index,
            y=close_prices,
            mode='lines',
            name=f"{ticker} Close Price",
            line=dict(color=colors[idx % len(colors)], width=2)
        ))

    currency_symbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'AUD': 'A$'}
    currency_symbol = currency_symbols.get(currency, '$')
    fig.update_layout(
        title="Stock Price History Comparison",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@st.cache_data(ttl=3600)
def plot_volume_chart(histories, tickers):
    """Plot the trading volume history for multiple stocks."""
    if not histories or not tickers:
        return None

    fig = go.Figure()
    colors = ['gray', 'lightblue', 'lightgreen', 'lightcoral', 'violet', 'lightbrown', 'lightpink', 'darkgray']

    for idx, (ticker, history) in enumerate(zip(tickers, histories)):
        if history is None or history.empty:
            continue

        fig.add_trace(go.Bar(
            x=history.index,
            y=history['Volume'],
            name=f"{ticker} Volume",
            marker_color=colors[idx % len(colors)],
            opacity=0.6
        ))

    fig.update_layout(
        title="Trading Volume Comparison",
        xaxis_title="Date",
        yaxis_title="Volume (Shares)",
        hovermode="x unified",
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@st.cache_data(ttl=3600)
def plot_moving_averages(histories, tickers, currency='USD', native_currencies=None):
    """Plot 50-day and 200-day moving averages with closing price for multiple stocks."""
    if not histories or not tickers or any(len(history) < 200 for history in histories if history is not None):
        return None

    fig = go.Figure()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    ma_colors = ['lightblue', 'lightcoral', 'lightgreen', 'violet', 'skyblue', 'salmon', 'lime', 'orchid']

    for idx, (ticker, history) in enumerate(zip(tickers, histories)):
        if history is None or history.empty:
            continue

        close_prices = history['Close']
        native_currency = native_currencies[idx]
        if currency != native_currency:
            close_prices = close_prices.apply(lambda x: convert_currency(x, currency, native_currency))

        ma50 = close_prices.rolling(window=50).mean()
        ma200 = close_prices.rolling(window=200).mean()

        fig.add_trace(go.Scatter(
            x=history.index,
            y=close_prices,
            mode='lines',
            name=f"{ticker} Close Price",
            line=dict(color=colors[idx % len(colors)], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=history.index,
            y=ma50,
            mode='lines',
            name=f"{ticker} 50-Day MA",
            line=dict(color=ma_colors[idx % len(ma_colors)], width=1.5, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=history.index,
            y=ma200,
            mode='lines',
            name=f"{ticker} 200-Day MA",
            line=dict(color=ma_colors[idx % len(ma_colors)], width=1.5, dash='dot')
        ))

    currency_symbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'AUD': 'A$'}
    currency_symbol = currency_symbols.get(currency, '$')
    fig.update_layout(
        title="Stock Price with Moving Averages Comparison",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@st.cache_data(ttl=3600)
def plot_candlestick_chart(history, ticker, currency='USD', native_currency='USD'):
    """Plot a candlestick chart for price movements for a single stock."""
    if history is None or history.empty:
        return None

    prices = history[['Open', 'High', 'Low', 'Close']].copy()
    if currency != native_currency:
        for col in prices.columns:
            prices[col] = prices[col].apply(lambda x: convert_currency(x, currency, native_currency))

    fig = go.Figure(data=[go.Candlestick(
        x=history.index,
        open=prices['Open'],
        high=prices['High'],
        low=prices['Low'],
        close=prices['Close'],
        name=f"{ticker} OHLC",
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    currency_symbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'AUD': 'A$'}
    currency_symbol = currency_symbols.get(currency, '$')
    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        hovermode="x unified",
        xaxis_rangeslider_visible=False
    )
    return fig

def get_financial_advice_llm(query):
    """Generate financial advice using open source LLM."""
    try:
        model = load_llm_model()
        if model is None:
            return "I'm sorry, the language model could not be loaded. Please try again later."

        prompt = f"""
        <|system|>
        You are a knowledgeable financial advisor assistant. Provide clear, accurate, and helpful advice on personal finance topics.
        Keep responses concise but informative. Don't recommend specific investments or make promises about returns.
        </|system|>

        <|user|>
        {query}
        </|user|>

        <|assistant|>
        """

        response = model(prompt, max_length=400, temperature=0.7, num_return_sequences=1)
        generated_text = response[0]['generated_text']

        if "<|assistant|>" in generated_text:
            assistant_response = generated_text.split("<|assistant|>")[-1].strip()
            if "<|" in assistant_response:
                assistant_response = assistant_response.split("<|")[0].strip()
        else:
            assistant_response = generated_text.replace(prompt, "").strip()

        if len(assistant_response) < 20:
            return "I'm sorry, I couldn't generate a specific response to your question. Please try rephrasing or asking something else about personal finance."

        return assistant_response

    except Exception as e:
        st.error(f"Error with LLM service: {e}")
        return "I'm sorry, I couldn't process your request due to a technical issue. Please try again or ask a different question."

# Page Content
if page == "Stock Analysis":
    st.subheader("📈 Stock Analysis")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker_input = st.text_input(
            "Enter Stock Ticker Symbols (comma-separated, e.g., AAPL, RELIANCE, TSLA)",
            value="RELIANCE",
            help="Enter one or more ticker symbols separated by commas."
        )
        # Parse the input into a list of tickers
        tickers = [ticker.strip().upper() for ticker in ticker_input.split(",") if ticker.strip()]
    with col2:
        # Add an option for no exchange (for US stocks)
        exchange_options = ["None", "NSE", "BSE"]
        exchange = st.selectbox(
            "Exchange (None for US stocks like AAPL)",
            exchange_options,
            index=1  # Default to NSE since RELIANCE is the default ticker
        )
    with col3:
        period_options = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "YTD", "Max"]
        period_display = st.selectbox(
            "Select Period",
            period_options,
            index=period_options.index("1Y")  # Default to 1Y
        )
        # Map display period to yfinance period
        period_map = {
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
            "2Y": "2y",
            "5Y": "5y",
            "YTD": "ytd",
            "Max": "max"
        }
        period = period_map[period_display]

    if tickers:
        # Validate ticker format
        for ticker in tickers:
            if not re.match(r'^[A-Z0-9]{1,10}$', ticker):
                st.error(f"Invalid ticker symbol: {ticker}. Please use valid symbols (e.g., AAPL, RELIANCE, TSLA).")
                st.stop()

        # Validate tickers exist
        manager = get_stock_manager()
        valid_tickers = []
        for ticker in tickers:
            yf_ticker = ticker
            if exchange == "NSE" and not ticker.endswith(".NS"):
                yf_ticker = f"{ticker}.NS"
            elif exchange == "BSE" and not ticker.endswith(".BO"):
                yf_ticker = f"{ticker}.BO"
            
            with st.spinner(f"Validating ticker {ticker}..."):
                if manager.validate_ticker(yf_ticker):
                    valid_tickers.append(ticker)
                else:
                    st.warning(f"Ticker {ticker} is not valid or not found. Skipping...")

        if not valid_tickers:
            st.error("No valid tickers provided. Please enter valid ticker symbols.")
            st.stop()

        with st.spinner(f"Fetching data for {', '.join(valid_tickers)}..."):
            # Adjust tickers for yfinance based on exchange
            yf_tickers = []
            native_currencies = []
            for ticker in valid_tickers:
                yf_ticker = ticker
                if exchange == "NSE" and not ticker.endswith(".NS"):
                    yf_ticker = f"{ticker}.NS"
                elif exchange == "BSE" and not ticker.endswith(".BO"):
                    yf_ticker = f"{ticker}.BO"
                yf_tickers.append(yf_ticker)
                native_currency = 'INR' if exchange in ["NSE", "BSE"] else 'USD'
                native_currencies.append(native_currency)

            # Fetch data for all tickers
            stock_data_list = []
            for yf_ticker in yf_tickers:
                st.info(f"Fetching {yf_ticker} data for period {period_display}...")
                stock_data = get_stock_data(yf_ticker, period)
                stock_data_list.append(stock_data)

            if any(stock_data is not None for stock_data in stock_data_list):
                st.success(f"Successfully retrieved data for {', '.join(ticker for ticker, data in zip(valid_tickers, stock_data_list) if data)}")

                # Extract histories for charting
                histories = [stock_data["history"] if stock_data else None for stock_data in stock_data_list]

                # Determine if Moving Averages tab should be shown (requires at least 200 days)
                period_days = {
                    "1M": 30,
                    "3M": 90,
                    "6M": 180,
                    "1Y": 365,
                    "2Y": 730,
                    "5Y": 1825,
                    "YTD": (pd.Timestamp.now() - pd.Timestamp.now().replace(month=1, day=1)).days,
                    "Max": float('inf')
                }
                days_in_period = period_days[period_display]
                show_moving_averages = days_in_period >= 200

                # Define tabs based on whether Moving Averages should be shown
                if show_moving_averages:
                    tabs = st.tabs([
                        "Price History",
                        "Trading Volume",
                        "Moving Averages",
                        "Candlestick Charts"
                    ])
                else:
                    tabs = st.tabs([
                        "Price History",
                        "Trading Volume",
                        "Candlestick Charts"
                    ])

                # Price History Tab
                with tabs[0]:
                    with st.spinner("Generating price history chart..."):
                        history_chart = plot_stock_price_history(histories, valid_tickers, st.session_state.currency, native_currencies)
                        if history_chart:
                            st.plotly_chart(history_chart, use_container_width=True)
                        else:
                            st.warning("Could not display price history chart.")

                # Trading Volume Tab
                with tabs[1]:
                    with st.spinner("Generating volume chart..."):
                        volume_chart = plot_volume_chart(histories, valid_tickers)
                        if volume_chart:
                            st.plotly_chart(volume_chart, use_container_width=True)
                        else:
                            st.warning("Could not display volume chart.")

                # Moving Averages Tab (if applicable)
                if show_moving_averages:
                    with tabs[2]:
                        with st.spinner("Generating moving averages chart..."):
                            ma_chart = plot_moving_averages(histories, valid_tickers, st.session_state.currency, native_currencies)
                            if ma_chart:
                                st.plotly_chart(ma_chart, use_container_width=True)
                            else:
                                st.warning("Could not display moving averages chart. Ensure at least 200 days of data are available.")
                    candlestick_tab = tabs[3]
                else:
                    candlestick_tab = tabs[2]

                # Candlestick Charts Tab
                with candlestick_tab:
                    for ticker, stock_data, native_currency in zip(valid_tickers, stock_data_list, native_currencies):
                        if stock_data and not stock_data["history"].empty:
                            st.subheader(f"Candlestick Chart for {ticker}")
                            with st.spinner(f"Generating candlestick chart for {ticker}..."):
                                candlestick_chart = plot_candlestick_chart(stock_data["history"], ticker, st.session_state.currency, native_currency)
                                if candlestick_chart:
                                    st.plotly_chart(candlestick_chart, use_container_width=True)
                                else:
                                    st.warning(f"Could not display candlestick chart for {ticker}.")
                        else:
                            st.warning(f"No data available to display candlestick chart for {ticker}.")

                # Key Financial Metrics
                st.subheader("Key Financial Metrics Comparison")
                all_metrics = []
                for ticker, stock_data in zip(valid_tickers, stock_data_list):
                    if stock_data:
                        metrics = calculate_financial_metrics(stock_data)
                        metrics["Ticker"] = ticker
                        all_metrics.append(metrics)

                if all_metrics:
                    # Create a DataFrame for comparison
                    metrics_df = pd.DataFrame(all_metrics)
                    metrics_df.set_index("Ticker", inplace=True)

                    # Convert currency for applicable columns
                    for ticker, native_currency in zip(valid_tickers, native_currencies):
                        for col in metrics_df.columns:
                            if "%" not in col and col not in ["Avg Volume", "P/E Ratio", "Beta"]:
                                metrics_df.loc[ticker, col] = convert_currency(
                                    metrics_df.loc[ticker, col],
                                    st.session_state.currency,
                                    native_currency
                                ) if isinstance(metrics_df.loc[ticker, col], (int, float)) else metrics_df.loc[ticker, col]

                    # Format the DataFrame for display
                    display_df = metrics_df.copy()
                    for col in display_df.columns:
                        if "%" not in col and col not in ["Avg Volume", "P/E Ratio", "Beta"]:
                            display_df[col] = display_df[col].apply(lambda x: format_currency(x, st.session_state.currency) if isinstance(x, (int, float)) else x)
                        else:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

                    # Color-code Price Change (%)
                    def color_price_change(val):
                        if isinstance(val, (int, float)):
                            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                            return f'color: {color}'
                        return ''

                    styled_df = display_df.style.applymap(color_price_change, subset=["Price Change (%)"])
                    st.dataframe(styled_df, use_container_width=True)

                    # Export Data Button
                    csv = display_df.to_csv(index=True)
                    st.download_button(
                        label="Download Metrics as CSV",
                        data=csv,
                        file_name="stock_metrics.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Could not calculate financial metrics for the selected stocks.")

                # Company Information
                st.subheader("Company Information")
                for ticker, stock_data in zip(valid_tickers, stock_data_list):
                    if stock_data and stock_data.get("info", {}).get("longBusinessSummary"):
                        st.markdown(f"**{ticker}**")
                        st.info(stock_data["info"]["longBusinessSummary"])
                    else:
                        st.info(f"No company information available for {ticker}.")

            else:
                st.error(f"Could not retrieve data for any of the selected tickers. Please check the ticker symbols.")

elif page == "Financial Advisor Chat":
    st.subheader("💬 Financial Advisor Chat")

    st.write("""
    Ask me anything about personal finance, investing, retirement planning, or financial terms!

    Note: I'm using a small language model that may have limitations. For complex financial advice,
    consult with a certified financial advisor.
    """)

    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['message']}", unsafe_allow_html=True)
            else:
                st.markdown(f"**Advisor:** {chat['message']}", unsafe_allow_html=True)

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question:", key="user_message")
        submit_chat = st.form_submit_button("Send")

    with st.expander("Example questions you can ask"):
        st.markdown("""
        - What is a Roth IRA?
        - How much should I save for retirement?
        - What's a good investment strategy for beginners?
        - Should I pay off debt or invest?
        - What is an ETF?
        - How should I build an emergency fund?
        """)

    if submit_chat and user_input:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        with st.spinner("Thinking..."):
            response = get_financial_advice_llm(user_input)
        st.session_state.chat_history.append({"role": "assistant", "message": response})
        st.rerun()
