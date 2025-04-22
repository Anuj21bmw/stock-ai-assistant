import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import os
import json
import time
import io
from datetime import datetime, timedelta

# --- SETTINGS ---
EXCEL_PATH = 'stock_queries.xlsx'
USE_AI = True
OPENAI_API_KEY = "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"  # <--- Replace with your actual OpenAI key

# --- INIT ---
client = OpenAI(api_key=OPENAI_API_KEY)

if not os.path.exists(EXCEL_PATH):
    df = pd.DataFrame(columns=['Prompt', 'Result'])
    df.to_excel(EXCEL_PATH, index=False)

st.set_page_config(page_title="📈 Advanced Stock AI Assistant", layout="wide")
st.title("📈 Excel Stock AI Assistant")

st.sidebar.header("Settings")
use_ai = st.sidebar.checkbox("Use AI Parsing (OpenAI)", value=USE_AI)
default_ticker = st.sidebar.text_input("Default Ticker (if not specified)", value="AAPL")
default_period = st.sidebar.selectbox("Default Time Period", 
                                     options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                                     index=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache for stock data to avoid repeated API calls
if "stock_cache" not in st.session_state:
    st.session_state.stock_cache = {}

# Define the AI parsing function using OpenAI
def parse_query_with_ai(prompt):
    if not use_ai or not OPENAI_API_KEY or OPENAI_API_KEY == "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":
        return fallback_parse_query(prompt)
    
    try:
        system_message = """
        You are a stock market assistant that extracts structured information from user queries.
        Parse the following query and extract relevant parameters in JSON format with these fields:
        - ticker: Stock symbol (e.g., AAPL, MSFT)
        - period: Time period (e.g., 1d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)
        - analysis_type: One of [price_history, financial_metrics, company_info, comparison, prediction]
        - metrics: List of metrics to analyze [price, volume, pe_ratio, market_cap, revenue, profit, dividend, etc.]
        - comparison_tickers: List of tickers to compare with (if any)
        - chart_type: One of [line, candle, bar, area, None]
        - additional_info: Any other relevant details
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        parsed_data = json.loads(response.choices[0].message.content)
        
        # Apply default values for missing fields
        if not parsed_data.get('ticker'):
            parsed_data['ticker'] = default_ticker
        if not parsed_data.get('period'):
            parsed_data['period'] = default_period
        if not parsed_data.get('analysis_type'):
            parsed_data['analysis_type'] = 'price_history'
        if not parsed_data.get('chart_type'):
            parsed_data['chart_type'] = 'line'
            
        return parsed_data
    
    except Exception as e:
        st.warning(f"AI parsing error: {str(e)}. Using fallback parsing.")
        return fallback_parse_query(prompt)

# Simple fallback parsing when AI is not available
def fallback_parse_query(prompt):
    prompt = prompt.lower()
    
    # Extract ticker
    tickers = {
        "amazon": "AMZN", "tesla": "TSLA", "apple": "AAPL", 
        "microsoft": "MSFT", "google": "GOOGL", "facebook": "META",
        "netflix": "NFLX", "nvidia": "NVDA", "amd": "AMD"
    }
    
    ticker = default_ticker
    for company, symbol in tickers.items():
        if company in prompt:
            ticker = symbol
            break
    
    # Extract period
    period = default_period
    if "day" in prompt or "24 hour" in prompt:
        period = "1d"
    elif "week" in prompt:
        period = "1wk"
    elif "month" in prompt and "6" not in prompt and "3" not in prompt:
        period = "1mo"
    elif "3 month" in prompt or "quarter" in prompt:
        period = "3mo"
    elif "6 month" in prompt or "half year" in prompt:
        period = "6mo"
    elif "year" in prompt and "2" not in prompt and "5" not in prompt and "10" not in prompt:
        period = "1y"
    elif "2 year" in prompt:
        period = "2y"
    elif "5 year" in prompt:
        period = "5y"
    elif "10 year" in prompt or "decade" in prompt:
        period = "10y"
    elif "all" in prompt or "max" in prompt:
        period = "max"
    
    # Determine analysis type
    analysis_type = "price_history"  # default
    if any(term in prompt for term in ["compare", "versus", "vs"]):
        analysis_type = "comparison"
    elif any(term in prompt for term in ["revenue", "profit", "income", "earnings", "financial"]):
        analysis_type = "financial_metrics"
    elif any(term in prompt for term in ["about", "info", "what is", "tell me"]):
        analysis_type = "company_info"
    elif any(term in prompt for term in ["predict", "forecast", "future", "will", "expect"]):
        analysis_type = "prediction"
    
    # Chart type
    chart_type = "line"
    if "candle" in prompt:
        chart_type = "candle"
    elif "bar" in prompt:
        chart_type = "bar"
    elif "area" in prompt:
        chart_type = "area"
    
    # Find comparison tickers
    comparison_tickers = []
    for company, symbol in tickers.items():
        if company != ticker.lower() and company in prompt:
            comparison_tickers.append(symbol)
    
    return {
        "ticker": ticker,
        "period": period,
        "analysis_type": analysis_type,
        "metrics": ["price"],
        "comparison_tickers": comparison_tickers,
        "chart_type": chart_type,
        "additional_info": ""
    }

# Get cached stock data or fetch new data
def get_stock_data(ticker, period):
    cache_key = f"{ticker}_{period}"
    
    # Check if we have fresh cached data
    if cache_key in st.session_state.stock_cache:
        cached_time, cached_data = st.session_state.stock_cache[cache_key]
        # Use cached data if it's less than 1 hour old
        if time.time() - cached_time < 3600:
            return cached_data
    
    # Fetch new data
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        # Cache the new data with current timestamp
        st.session_state.stock_cache[cache_key] = (time.time(), hist)
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

# Generate price history chart
def create_price_chart(ticker, period, chart_type="line"):
    hist = get_stock_data(ticker, period)
    
    if hist.empty:
        return None, "No data available"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if chart_type == "line":
        hist['Close'].plot(ax=ax)
    elif chart_type == "candle":
        # Simple candlestick-like chart
        up = hist[hist.Close >= hist.Open]
        down = hist[hist.Close < hist.Open]
        
        # Plot up candles
        ax.bar(up.index, up.Close - up.Open, bottom=up.Open, width=0.8, color='green', alpha=0.5)
        ax.bar(up.index, up.High - up.Close, bottom=up.Close, width=0.1, color='green', alpha=0.5)
        ax.bar(up.index, up.Open - up.Low, bottom=up.Low, width=0.1, color='green', alpha=0.5)
        
        # Plot down candles
        ax.bar(down.index, down.Open - down.Close, bottom=down.Close, width=0.8, color='red', alpha=0.5)
        ax.bar(down.index, down.High - down.Open, bottom=down.Open, width=0.1, color='red', alpha=0.5)
        ax.bar(down.index, down.Close - down.Low, bottom=down.Low, width=0.1, color='red', alpha=0.5)
    elif chart_type == "bar":
        hist['Close'].plot(kind='bar', ax=ax)
    elif chart_type == "area":
        hist['Close'].plot(kind='area', ax=ax, alpha=0.5)
    
    ax.set_title(f"{ticker} Stock Price - {period}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    plt.grid(True)
    plt.tight_layout()
    
    # Calculate some basic statistics
    current_price = hist['Close'].iloc[-1]
    start_price = hist['Close'].iloc[0]
    price_change = current_price - start_price
    percent_change = (price_change / start_price) * 100
    
    stats = f"""
    {ticker} Summary ({period}):
    - Current Price: ${current_price:.2f}
    - Change: ${price_change:.2f} ({percent_change:.2f}%)
    - High: ${hist['High'].max():.2f}
    - Low: ${hist['Low'].min():.2f}
    - Avg Volume: {int(hist['Volume'].mean()):,}
    """
    
    return fig, stats

# Compare multiple stocks
def compare_stocks(main_ticker, comparison_tickers, period):
    if not comparison_tickers:
        return create_price_chart(main_ticker, period)
    
    # Add the main ticker to the list for comparison
    all_tickers = [main_ticker] + comparison_tickers
    
    # Fetch data for all tickers
    price_data = {}
    for ticker in all_tickers:
        hist = get_stock_data(ticker, period)
        if not hist.empty:
            # Normalize to percentage change from first day for fair comparison
            price_data[ticker] = hist['Close'] / hist['Close'].iloc[0] * 100
    
    if not price_data:
        return None, "No data available"
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for ticker, prices in price_data.items():
        prices.plot(ax=ax, label=ticker)
    
    ax.set_title(f"Comparison of {', '.join(all_tickers)} - {period}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Price (%)')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Generate comparison statistics
    stats = f"Comparison Summary ({period}):\n"
    for ticker in all_tickers:
        if ticker in price_data:
            start_val = price_data[ticker].iloc[0]
            end_val = price_data[ticker].iloc[-1]
            percent_change = end_val - start_val  # Already normalized to percentage
            stats += f"- {ticker}: {percent_change:.2f}% change\n"
    
    return fig, stats

# Generate financial metrics table
def get_financial_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Get key statistics
        info = stock.info
        
        # Get income statement data
        income_stmt = stock.income_stmt
        
        # Create a metrics table
        metrics = {
            "Market Cap": info.get("marketCap", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Avg Volume": info.get("averageVolume", "N/A"),
        }
        
        # Format the metrics
        formatted_metrics = {}
        for key, value in metrics.items():
            if key == "Market Cap" and isinstance(value, (int, float)):
                formatted_metrics[key] = f"${value/1e9:.2f}B"
            elif key == "Dividend Yield" and isinstance(value, (int, float)):
                formatted_metrics[key] = f"{value*100:.2f}%"
            elif isinstance(value, (int, float)):
                formatted_metrics[key] = f"{value:.2f}"
            else:
                formatted_metrics[key] = value
        
        # Create a pandas DataFrame for display
        metrics_df = pd.DataFrame(formatted_metrics.items(), columns=["Metric", "Value"])
        
        # Generate a text summary
        summary = f"""
        Financial Summary for {ticker}:
        
        Market Cap: {formatted_metrics.get("Market Cap", "N/A")}
        P/E Ratio: {formatted_metrics.get("PE Ratio", "N/A")}
        EPS: {formatted_metrics.get("EPS", "N/A")}
        Dividend Yield: {formatted_metrics.get("Dividend Yield", "N/A")}
        52-Week Range: {formatted_metrics.get("52 Week Low", "N/A")} - {formatted_metrics.get("52 Week High", "N/A")}
        Average Volume: {formatted_metrics.get("Avg Volume", "N/A")}
        """
        
        # Check if we have revenue and income data
        if not income_stmt.empty:
            # Create a figure for financial chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot revenue and income
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue']
                ax.bar(revenue.index.astype(str), revenue/1e9, alpha=0.7, label='Revenue')
            
            if 'Net Income' in income_stmt.index:
                income = income_stmt.loc['Net Income']
                ax.plot(income.index.astype(str), income/1e9, marker='o', color='red', label='Net Income')
                
            ax.set_title(f"{ticker} Financial Performance")
            ax.set_xlabel('Year')
            ax.set_ylabel('Billion USD')
            ax.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            return metrics_df, summary, fig
        
        return metrics_df, summary, None
    
    except Exception as e:
        st.error(f"Error fetching financial data for {ticker}: {str(e)}")
        return pd.DataFrame(), f"Could not retrieve financial data for {ticker}", None

# Get company information
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant company information
        company_data = {
            "Name": info.get("shortName", ticker),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
            "Employees": info.get("fullTimeEmployees", "N/A"),
            "Business Summary": info.get("longBusinessSummary", "No information available")
        }
        
        # Format as text
        info_text = f"""
        # {company_data['Name']} ({ticker})
        
        **Sector:** {company_data['Sector']}
        **Industry:** {company_data['Industry']}
        **Country:** {company_data['Country']}
        **Employees:** {company_data['Employees']:,} (if available)
        **Website:** {company_data['Website']}
        
        ## Business Summary
        
        {company_data['Business Summary']}
        """
        
        return info_text
    
    except Exception as e:
        return f"Could not retrieve company information for {ticker}: {str(e)}"

# Save response to Excel
def save_to_excel(prompt, result):
    try:
        df = pd.read_excel(EXCEL_PATH)
        new_row = {
            'Prompt': prompt, 
            'Result': result
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(EXCEL_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to Excel: {str(e)}")
        return False

# Handle different types of analysis
def handle_query(query_data, original_prompt):
    ticker = query_data["ticker"]
    period = query_data["period"]
    analysis_type = query_data["analysis_type"]
    chart_type = query_data["chart_type"]
    comparison_tickers = query_data["comparison_tickers"]
    
    result_content = ""
    fig = None
    is_visualization = False
    
    # Check if this is a simple price query
    if "today" in original_prompt.lower() and "price" in original_prompt.lower():
        # Get the latest stock price
        try:
            stock = yf.Ticker(ticker)
            today_data = stock.history(period="1d")
            
            if not today_data.empty:
                current_price = today_data['Close'].iloc[-1]
                open_price = today_data['Open'].iloc[-1]
                high_price = today_data['High'].iloc[-1]
                low_price = today_data['Low'].iloc[-1]
                
                # If asking for prediction as well
                if "predict" in original_prompt.lower() or "next day" in original_prompt.lower() or "tomorrow" in original_prompt.lower():
                    # Simple naive prediction based on recent trend
                    recent_data = stock.history(period="5d")
                    if len(recent_data) >= 3:
                        avg_change = recent_data['Close'].pct_change().mean()
                        predicted_price = current_price * (1 + avg_change)
                        result_content = f"""
                        {ticker} Current Price: ${current_price:.2f}
                        Today's Range: ${low_price:.2f} - ${high_price:.2f}
                        
                        Predicted Price (Next Day): ${predicted_price:.2f}
                        (Based on recent average daily change of {avg_change*100:.2f}%)
                        
                        Note: This is a simple prediction based on recent performance, not financial advice.
                        """
                    else:
                        result_content = f"{ticker} Current Price: ${current_price:.2f}\nNot enough data for prediction."
                else:
                    result_content = f"""
                    {ticker} Current Price: ${current_price:.2f}
                    Open: ${open_price:.2f}
                    High: ${high_price:.2f}
                    Low: ${low_price:.2f}
                    """
            else:
                result_content = f"Could not retrieve today's price for {ticker}."
                
        except Exception as e:
            result_content = f"Error retrieving price data for {ticker}: {str(e)}"
    
    # Check if user wants a visualization
    elif "chart" in original_prompt.lower() or "plot" in original_prompt.lower() or "graph" in original_prompt.lower() or "visualize" in original_prompt.lower() or "show" in original_prompt.lower():
        is_visualization = True
        
        if analysis_type == "price_history":
            fig, result_content = create_price_chart(ticker, period, chart_type)
        
        elif analysis_type == "comparison":
            fig, result_content = compare_stocks(ticker, comparison_tickers, period)
        
        elif analysis_type == "prediction":
            # For predictions, we'll use historical data with projection
            hist = get_stock_data(ticker, period)
            if not hist.empty:
                last_price = hist['Close'].iloc[-1]
                
                # Simple "prediction" based on recent trend
                trend = "upward" if hist['Close'].iloc[-1] > hist['Close'].iloc[-20] else "downward"
                
                result_content = f"""
                Prediction for {ticker} based on historical data:
                
                Current price: ${last_price:.2f}
                Recent trend: {trend}
                """
                
                # Create a chart with historical data and projection
                fig, ax = plt.subplots(figsize=(10, 6))
                hist['Close'].plot(ax=ax, label='Historical')
                
                # Add a projection line
                last_date = hist.index[-1]
                future_dates = [last_date + timedelta(days=i*7) for i in range(1, 5)]
                
                if len(hist) >= 30:
                    recent_trend = (hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / 30
                    future_values = [hist['Close'].iloc[-1] + recent_trend * i * 7 for i in range(1, 5)]
                    
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Projected': future_values
                    }).set_index('Date')
                    
                    future_df['Projected'].plot(ax=ax, style='--', color='red', alpha=0.7, label='Projection')
                    
                ax.set_title(f"{ticker} Price Projection")
                ax.set_xlabel('Date')
                ax.set_ylabel('Price (USD)')
                ax.legend()
                plt.grid(True)
                plt.tight_layout()
        
        elif analysis_type == "financial_metrics":
            metrics_df, text_summary, metrics_fig = get_financial_metrics(ticker)
            result_content = text_summary
            fig = metrics_fig
    
    # For all other types of queries
    else:
        if analysis_type == "price_history":
            # Get the latest stock information
            hist = get_stock_data(ticker, period)
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
                pct_change = (change / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
                
                result_content = f"""
                {ticker} Latest Price: ${current_price:.2f}
                Change: ${change:.2f} ({pct_change:.2f}%)
                Period: {period}
                High: ${hist['High'].max():.2f}
                Low: ${hist['Low'].min():.2f}
                """
        
        elif analysis_type == "company_info":
            result_content = get_company_info(ticker)
            
        elif analysis_type == "financial_metrics":
            metrics_df, text_summary, _ = get_financial_metrics(ticker)
            result_content = text_summary
            
    # For visualization requests, create an image buffer to save in Excel
    excel_result = result_content
    if is_visualization and fig is not None:
        # Add a note that a visualization was created
        excel_result = result_content + "\n\n[Visualization was generated and displayed in the app]"

    # Save to Excel (original prompt and result)
    save_to_excel(original_prompt, excel_result)
    
    return fig, result_content

# Main app interface
st.title("📈 Excel Stock AI Assistant")

# Add two columns layout
col1, col2 = st.columns(2)

with col1:
    st.header("Enter Your Query")
    user_input = st.text_area("Type your stock-related question:", height=100, 
                            placeholder="Examples:\n- Show me a chart of Apple's stock for the last 6 months\n- What's today's price of Tesla and predicted price for tomorrow?\n- Compare Amazon and Microsoft stock performance")
    
    submit_button = st.button("Generate Result", type="primary")

# Process when button is clicked
if submit_button and user_input:
    # Parse the query
    with st.spinner("Analyzing your request..."):
        query_data = parse_query_with_ai(user_input)
    
    # Process the query and get response
    with st.spinner(f"Processing data for {query_data['ticker']}..."):
        fig, result_content = handle_query(query_data, user_input)
    
    # Display the response
    with col2:
        st.header("Result")
        st.markdown(result_content)
        if fig:
            st.pyplot(fig)
    
    # Show success message
    st.success("✅ Result generated and saved to Excel!")
    
    # Update messages for chat history if needed
    if "messages" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": result_content})

# Display Excel contents (showing just the two columns)
st.header("Excel Data Log")
try:
    excel_data = pd.read_excel(EXCEL_PATH)
    st.dataframe(excel_data, use_container_width=True)
    
    # Add download button
    excel_buffer = io.BytesIO()
    excel_data.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="Download Excel File",
        data=excel_buffer,
        file_name="stock_queries.xlsx",
        mime="application/vnd.ms-excel"
    )
except Exception as e:
    st.error(f"Error reading Excel file: {str(e)}")