# Options Strategy Analyzer

A comprehensive web application for analyzing both bullish and bearish options strategies. This application provides a user-friendly interface to input stock tickers and date ranges, then generates detailed HTML reports showing the best options combinations for both market directions.

## Features

- **Multi-Strategy Support**: Analyze both bullish and bearish risk reversal strategies
- **Web Interface**: Clean, modern UI for easy input of analysis parameters
- **Real-time Analysis**: Runs sophisticated Python analysis engines in the background
- **Professional Reports**: Generates detailed HTML reports with strategy recommendations
- **No Page Refresh**: Results are displayed dynamically without reloading the page
- **Error Handling**: Comprehensive error handling and user feedback
- **Responsive Design**: Works perfectly on desktop and mobile devices

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   Make sure you have the following files in your project directory:
   - `app.py` - Flask backend server
   - `analysis_engine.py` - Analysis logic for both strategies
   - `templates/home.html` - Strategy selection page
   - `templates/bullish_analyzer.html` - Bullish strategy analyzer
   - `templates/bearish_analyzer.html` - Bearish strategy analyzer
   - `requirements.txt` - Python dependencies

## Running the Application

1. **Start the Flask Server**:
   ```bash
   python app.py
   ```

2. **Access the Web Interface**:
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Using the Application**:
   - **Home Page**: Choose between Bullish or Bearish strategy analysis
   - **Bullish Analysis**: Long OTM Call + Short OTM Put for upward movement
   - **Bearish Analysis**: Long OTM Put + Short OTM Call for downward movement
   - Enter a stock ticker symbol (e.g., AAPL, TSLA, SPY)
   - Set minimum and maximum days to expiration (DTE)
   - Click "Run Analysis" to generate the report
   - View the results directly on the page

## Deployment on Render

This application is configured for deployment on a platform like Render.

### 1. Create a Redis Instance
Before deploying your web service, you need a Redis cache.
- In the Render dashboard, go to **New > Redis**.
- Choose a plan (the free plan is sufficient for this app).
- After creation, find your new Redis instance and copy the **Internal Connection URL**. It will look like `redis://red-....`.

### 2. Deploy the Web Service
- In the Render dashboard, go to **New > Web Service** and connect your GitHub repository.
- During setup, go to the **Environment** section.
- Add a new **Environment Variable**:
  - **Key**: `REDIS_URL`
  - **Value**: Paste the Internal Connection URL you copied from your Redis instance.
- Render will automatically install dependencies from `requirements.txt` and use the `Procfile` to start the application with `gunicorn`.

### 3. Verify Caching
- After deployment, go to the **Logs** tab for your web service.
- The first time you run an analysis, you will see a log message: `--- CACHE MISS ---`.
- If you run the exact same analysis again within 5 minutes, you will **not** see that message, confirming the result was served from the cache.

## How It Works

1. **Home Page**: Users select their market outlook (bullish or bearish)
2. **Strategy Pages**: Dedicated interfaces for each strategy type
3. **Backend**: Flask server receives requests and calls the appropriate analysis engine
4. **Analysis**: Sophisticated algorithms process options data and calculate optimal combinations
5. **Results**: Returns formatted HTML reports with detailed strategy recommendations

## File Structure

```
pythonProject2/
├── app.py                      # Flask backend server
├── analysis_engine.py          # Analysis logic for both strategies
├── templates/
│   ├── home.html              # Strategy selection page
│   ├── bullish_analyzer.html  # Bullish strategy interface
│   └── bearish_analyzer.html  # Bearish strategy interface
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── LONG.py                   # Original script (for reference)
```

## Strategy Details

### Bullish Risk Reversal
- **Structure**: Long OTM Call + Short OTM Put
- **Market Outlook**: Expecting upward price movement
- **Risk**: Limited to short put strike if assigned
- **Reward**: Unlimited upside potential
- **Vega Target**: Positive net vega for volatility exposure

### Bearish Risk Reversal
- **Structure**: Long OTM Put + Short OTM Call
- **Market Outlook**: Expecting downward price movement
- **Risk**: Unlimited if stock rises above short call strike
- **Reward**: Limited to long put strike if assigned
- **Vega Target**: Negative or near-zero net vega

## Technical Details

- **Backend**: Flask web framework with multiple routes
- **Caching**: Production-ready Redis cache via Flask-Caching to prevent API rate-limiting.
- **Frontend**: Vanilla JavaScript with modern CSS
- **Analysis**: Uses yfinance for options data, scipy for Black-Scholes calculations
- **Styling**: Responsive design with strategy-specific color schemes
- **Algorithms**: Advanced ranking systems for both bullish and bearish strategies

## Troubleshooting

- **Port Already in Use**: If port 5000 is busy, Flask will automatically try the next available port
- **Missing Dependencies**: Run `pip install -r requirements.txt` to install all required packages
- **Network Issues**: Ensure you have an internet connection for fetching options data via yfinance

## Notes

- The analysis examines the first 3 available expirations within your specified date range
- Results are ranked by strategy-specific criteria (delta, vega, efficiency scores)
- The application includes comprehensive error handling for invalid inputs and network issues
- Both strategies use sophisticated filtering and validation algorithms 