import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO

# -------------------------------
# Black-Scholes Functions
# -------------------------------

def d1(S, K, T, r, sigma, q=0):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_vega(S, K, T, r, sigma, q=0):
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T)

def bs_delta(S, K, T, r, sigma, option_type='call', q=0):
    D1 = d1(S, K, T, r, sigma, q)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(D1)
    else:
        return np.exp(-q * T) * (norm.cdf(D1) - 1)

# -------------------------------
# Fetch Options Data
# -------------------------------

def get_options_data(ticker, expiration):
    try:
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration)
    except Exception as e:
        print(f"Error fetching data for {ticker} at expiration {expiration}: {e}")
        return None

    calls = opt_chain.calls.copy()
    calls['type'] = 'call'

    puts = opt_chain.puts.copy()
    puts['type'] = 'put'

    options = pd.concat([calls, puts], ignore_index=True)
    options['expiration'] = expiration
    options = options[['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'impliedVolatility', 'type', 'expiration']]
    options = options.dropna(subset=['impliedVolatility'])
    options = options[(options['volume'] > 0) & (options['openInterest'] > 0)]
    options.reset_index(drop=True, inplace=True)
    return options

# -------------------------------
# Mispricing Detection
# -------------------------------

def analyze_mispricing(options, underlying_price, risk_free_rate=0.035, dividend_yield=0.0,
                       vega_neutral_tol=0.1, put_iv_threshold=0.05, call_iv_threshold=0.05,
                       max_iv_threshold=0.15, max_vega_tol=0.3, iv_step=0.01, vega_step=0.05):
    expiration_date = pd.to_datetime(options['expiration'].iloc[0])
    today = pd.to_datetime(datetime.now().date())
    T = max((expiration_date - today).days / 365, 1/365)

    puts = options[options['type'] == 'put'].copy()
    calls = options[options['type'] == 'call'].copy()
    avg_put_iv = puts['impliedVolatility'].mean()
    avg_call_iv = calls['impliedVolatility'].mean()

    for df in [puts, calls]:
        df['vega'] = df.apply(lambda row: bs_vega(underlying_price, row['strike'], T, risk_free_rate, row['impliedVolatility'], dividend_yield), axis=1)
        df['delta'] = df.apply(lambda row: bs_delta(underlying_price, row['strike'], T, risk_free_rate, row['impliedVolatility'], row['type'], dividend_yield), axis=1)

    current_put_thresh = put_iv_threshold
    current_call_thresh = call_iv_threshold
    current_vega_tol = vega_neutral_tol

    pairs_df = pd.DataFrame()

    while current_put_thresh <= max_iv_threshold and current_call_thresh <= max_iv_threshold and current_vega_tol <= max_vega_tol:
        puts['iv_diff'] = puts['impliedVolatility'] - avg_put_iv
        overpriced_puts = puts[puts['iv_diff'] >= current_put_thresh].copy()

        calls['iv_diff'] = avg_call_iv - calls['impliedVolatility']
        underpriced_calls = calls[calls['iv_diff'] >= current_call_thresh].copy()

        pairs = []
        for _, put in overpriced_puts.iterrows():
            for _, call in underpriced_calls.iterrows():
                combined_vega = call['vega'] - put['vega']
                if abs(combined_vega) <= current_vega_tol:
                    score = put['iv_diff'] + call['iv_diff']
                    net_delta = call['delta'] - put['delta']
                    initial_credit = put['bid'] - call['ask']

                    pairs.append({
                        'put_symbol': put['contractSymbol'],
                        'put_strike': put['strike'],
                        'put_bid': put['bid'],
                        'put_iv': put['impliedVolatility'],
                        'put_vega': put['vega'],
                        'call_symbol': call['contractSymbol'],
                        'call_strike': call['strike'],
                        'call_ask': call['ask'],
                        'call_iv': call['impliedVolatility'],
                        'call_vega': call['vega'],
                        'net_vega': combined_vega,
                        'net_delta': net_delta,
                        'initial_credit': initial_credit,
                        'mispricing_score': score,
                        'expiration': put['expiration']
                    })

        pairs_df = pd.DataFrame(pairs)
        if not pairs_df.empty:
            print(f"Found pairs with thresholds: put_iv >= {current_put_thresh:.3f}, call_iv >= {current_call_thresh:.3f}, vega_tol <= {current_vega_tol:.3f}")
            pairs_df.sort_values(by='mispricing_score', ascending=False, inplace=True)
            return pairs_df

        print(f"No pairs found for thresholds put_iv >= {current_put_thresh:.3f}, call_iv >= {current_call_thresh:.3f}, vega_tol <= {current_vega_tol:.3f}. Relaxing thresholds...")

        current_put_thresh += iv_step
        current_call_thresh += iv_step
        current_vega_tol += vega_step

    print("No suitable pairs found after relaxing thresholds.")
    return None


# -------------------------------
# Dashboard Exporter
# -------------------------------

def export_dashboard_html(df, underlying_price, ticker, expiration):
    best_pair = df.sort_values(by=["mispricing_score", "initial_credit"], ascending=[False, False]).iloc[0]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(df["put_strike"], df["put_iv"], label="Put IV", marker="o")
    plt.plot(df["call_strike"], df["call_iv"], label="Call IV", marker="o")
    plt.axvline(x=underlying_price, color="gray", linestyle="--", label=f"Underlying Price (${underlying_price:.2f})")
    plt.scatter([best_pair["put_strike"]], [best_pair["put_iv"]], color="red", s=100, zorder=5, label="Best Put")
    plt.scatter([best_pair["call_strike"]], [best_pair["call_iv"]], color="green", s=100, zorder=5, label="Best Call")
    plt.text(best_pair["put_strike"], best_pair["put_iv"] + 0.005, f'{best_pair["put_iv"]:.3f}', color='red')
    plt.text(best_pair["call_strike"], best_pair["call_iv"] + 0.005, f'{best_pair["call_iv"]:.3f}', color='green')
    plt.title(f"{ticker} IV Skew - Exp {expiration}")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    html = f"""
    <html>
    <head>
        <title>{ticker} Options Dashboard</title>
        <style>
            body {{ font-family: Arial; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{ticker} Mispriced Vega Dashboard</h1>
        <h2>Overview</h2>
        <p><strong>Underlying Price:</strong> ${underlying_price:.2f}<br>
           <strong>Expiration Date:</strong> {expiration}<br>
           <strong>Strategy:</strong> Sell Put with High IV, Buy Call with Low IV (Vega Neutral)</p>

        <h2>Best Trade Recommendation</h2>
        <ul>
            <li><strong>Put:</strong> {best_pair['put_symbol']} (Strike: {best_pair['put_strike']}, IV: {best_pair['put_iv']:.3f}, Vega: {best_pair['put_vega']:.2f})</li>
            <li><strong>Call:</strong> {best_pair['call_symbol']} (Strike: {best_pair['call_strike']}, IV: {best_pair['call_iv']:.3f}, Vega: {best_pair['call_vega']:.2f})</li>
            <li><strong>Initial Credit:</strong> ${best_pair['initial_credit']:.2f}</li>
            <li><strong>Net Vega:</strong> {best_pair['net_vega']:.4f}</li>
            <li><strong>Net Delta:</strong> {best_pair['net_delta']:.4f}</li>
            <li><strong>Mispricing Score:</strong> {best_pair['mispricing_score']:.4f}</li>
        </ul>

        <h2>Top Mispriced Vega Pairs</h2>
        {df.to_html(index=False)}

        <h2>Implied Volatility Skew</h2>
        <img src="data:image/png;base64,{plot_base64}" alt="IV Skew Chart"/>
    </body>
    </html>
    """

    html_path = os.path.join(os.path.dirname(__file__), f"{ticker.lower()}_options_dashboard.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard exported to: {html_path}")

# -------------------------------
# Main Execution Function
# -------------------------------

def run_dashboard(ticker, expiration):
    print(f"Fetching data for {ticker} expiring on {expiration}...")
    options = get_options_data(ticker, expiration)
    if options is None or options.empty:
        print("No option data available.")
        return

    underlying_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
    print(f"Current underlying price: {underlying_price:.2f}")

    pairs_df = analyze_mispricing(options, underlying_price)
    if pairs_df is None or pairs_df.empty:
        print("No mispriced Vega pairs identified.")
        return

    export_dashboard_html(pairs_df.head(5), underlying_price, ticker, expiration)

# -------------------------------
# Run Script with Input Prompt
# -------------------------------

if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., SRPT): ").upper()
    expiration = input("Enter expiration date (YYYY-MM-DD): ")
    run_dashboard(ticker, expiration)
