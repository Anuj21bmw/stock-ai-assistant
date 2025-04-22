# Excel Stock AI Assistant

A powerful Streamlit application that processes natural language queries about stocks and provides intelligent responses including visualizations, price data, predictions, and company information - all while automatically logging interactions to Excel.

## Features

- **Natural Language Processing**: Ask questions in plain English about any stock
- **Automatic Excel Logging**: All queries and responses are saved to a two-column Excel file
- **Real-Time Stock Data**: Fetches the latest stock information using Yahoo Finance
- **AI-Powered Query Understanding**: Uses OpenAI to parse and understand complex stock-related questions
- **Multiple Response Types**:
  - Current stock prices and statistics
  - Price predictions based on historical data
  - Interactive stock price charts with various visualization options
  - Multi-stock comparisons
  - Company information and financial metrics

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Anuj21bmw/stock-ai-assistant.git
cd excel-stock-assistant
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Add your OpenAI API key in the `app.py` file:
```python
OPENAI_API_KEY = "your-openai-key-here"
```

4. Run the application:
```bash
streamlit run app.py
```

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- yfinance
- matplotlib
- OpenAI Python SDK
- openpyxl

## Usage

1. Enter your stock-related question in the left column
2. Click "Generate Result" to process your query
3. View the result in the right column
4. The Excel data log at the bottom shows all previous queries and results
5. Download the Excel file using the provided button

## Example Queries

- **Simple price information**:
  - "What's today's price of Tesla?"
  - "Show me the current price of AAPL"

- **Price predictions**:
  - "What's today's price of MSFT and predicted price for tomorrow?"
  - "Predict the next day price for Amazon"

- **Visualizations**:
  - "Show me a chart of Netflix stock for the last 6 months"
  - "Plot Apple's stock price over the past year"
  - "Create a candlestick chart for AMD"

- **Comparisons**:
  - "Compare Tesla and Ford stock performance"
  - "Show me a chart comparing Amazon, Microsoft and Google"

- **Company information**:
  - "Tell me about Nvidia as a company"
  - "What is Disney's business model?"

- **Financial metrics**:
  - "What are the financial metrics for Meta?"
  - "Show me revenue data for Intel"

## How It Works

1. **Query Parsing**: The application uses AI to understand the intent and extract parameters from natural language queries
2. **Data Retrieval**: Stock data is fetched from Yahoo Finance based on the extracted parameters
3. **Processing**: The application processes the data according to the query type
4. **Visualization**: For chart requests, matplotlib is used to create appropriate visualizations
5. **Excel Logging**: All interactions are saved to an Excel file with two columns (Prompt and Result)

## Customization

You can modify the application by:

- Changing the default ticker and time period in the sidebar
- Adjusting the chart styles and visualizations
- Adding new analysis types in the `handle_query` function
- Expanding the AI parsing capabilities to handle more complex queries

## Limitations

- Predictions are based on simple historical trends and should not be used for financial advice
- The system requires an internet connection to fetch real-time stock data
- Query parsing accuracy depends on the OpenAI model and your API key
- Very specific or technical financial queries may require additional customization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web application framework
- [Yahoo Finance API](https://pypi.org/project/yfinance/) for stock data
- [OpenAI](https://openai.com/) for natural language processing capabilities
- [Matplotlib](https://matplotlib.org/) for data visualization