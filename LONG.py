import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
import webbrowser
import os

warnings.filterwarnings('ignore')


# -------------------------------
# Black-Scholes and Data Functions
# -------------------------------
def d1(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0: return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_vega(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0: return 0
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T) / 100


def bs_delta(S, K, T, r, sigma, option_type='call', q=0):
    if T <= 0: return 1.0 if S > K and option_type == 'call' else (-1.0 if S < K and option_type == 'put' else 0.0)
    D1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(D1) if option_type == 'call' else np.exp(-q * T) * (norm.cdf(D1) - 1)


def get_options_data(ticker, expiration, underlying_price):
    try:
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration)
    except Exception as e:
        print(f"    - Error fetching data for {ticker} at {expiration}: {e}")
        return pd.DataFrame(), pd.DataFrame()

    def process_df(input_df):
        if input_df is None or input_df.empty: return pd.DataFrame()
        df = input_df.copy()
        df.dropna(subset=['impliedVolatility', 'bid', 'ask'], inplace=True)
        if df.empty: return pd.DataFrame()
        mask = (((df['volume'] > 0) | (df['openInterest'] > 0)) & (df['impliedVolatility'] > 0.01) & (df['bid'] > 0) & (
                    (df['ask'] - df['bid']) / df['ask'] < 0.6))
        filtered_df = df[mask].copy()
        if not filtered_df.empty:
            filtered_df['moneyness'] = filtered_df['strike'] / underlying_price
            filtered_df['expiration'] = expiration
        return filtered_df

    calls = process_df(opt_chain.calls)
    puts = process_df(opt_chain.puts)
    return calls, puts


# -------------------------------
# Smarter Strategy Engine
# -------------------------------
def analyze_bullish_risk_reversal(calls, puts, underlying_price, expiration_date, risk_free_rate=0.045,
                                  dividend_yield=0.0):
    today = pd.to_datetime(datetime.now().date())
    exp_date = pd.to_datetime(expiration_date)
    T = max((exp_date - today).days / 365.0, 1 / (365 * 24))
    days_to_exp = (exp_date - today).days
    print(f"    - Time to expiration: {days_to_exp} days ({T:.4f} years)")

    for df, option_type in [(calls, 'call'), (puts, 'put')]:
        if df.empty: continue
        df['vega'] = df.apply(
            lambda row: bs_vega(underlying_price, row['strike'], T, risk_free_rate, row['impliedVolatility'],
                                dividend_yield), axis=1)
        df['delta'] = df.apply(
            lambda row: bs_delta(underlying_price, row['strike'], T, risk_free_rate, row['impliedVolatility'],
                                 option_type, dividend_yield), axis=1)

    otm_calls = calls[calls['strike'] > underlying_price].copy()
    otm_puts = puts[puts['strike'] < underlying_price].copy()

    max_strike_distance = underlying_price * 0.75
    otm_calls = otm_calls[otm_calls['strike'] < underlying_price + max_strike_distance]
    otm_puts = otm_puts[otm_puts['strike'] > underlying_price - max_strike_distance]

    if otm_calls.empty or otm_puts.empty:
        print("    - No suitable OTM calls and puts found to build strategies.")
        return []

    combinations = []
    for _, call in otm_calls.iterrows():
        for _, put in otm_puts.iterrows():
            if call['strike'] <= put['strike']: continue
            combo = create_strategy_combination(call, put)
            if combo and is_valid_bullish_combo(combo):
                combo.update({'strategy_type': 'Bullish Risk Reversal', 'expiration': expiration_date,
                              'days_to_exp': days_to_exp})
                combinations.append(combo)
    return combinations


def create_strategy_combination(call_row, put_row):
    try:
        net_cost = call_row['ask'] - put_row['bid']
        strike_diff = call_row['strike'] - put_row['strike']
        efficiency = -net_cost / strike_diff if strike_diff > 0 else 0
        return {
            'long_call_strike': call_row['strike'], 'short_put_strike': put_row['strike'],
            'net_cost': net_cost, 'iv_advantage': put_row['impliedVolatility'] - call_row['impliedVolatility'],
            'net_delta': call_row['delta'] - put_row['delta'], 'net_vega': call_row['vega'] - put_row['vega'],
            'max_loss_down': put_row['strike'] - (put_row['bid'] - call_row['ask']),
            'breakeven': call_row['strike'] + net_cost,
            'efficiency': efficiency
        }
    except (KeyError, TypeError):
        return None


def is_valid_bullish_combo(combo):
    if not combo: return False
    if abs(combo['net_cost']) > 20: return False
    if combo['net_delta'] <= 0.1 or combo['net_vega'] <= 0: return False
    return True


def rank_combinations(combinations):
    if not combinations: return []
    df = pd.DataFrame(combinations)

    def safe_normalize(series, reverse=False):
        if series.std() == 0 or len(series) < 2: return pd.Series([0.5] * len(series), index=series.index)
        norm = (series - series.min()) / (series.max() - series.min())
        return 1 - norm if reverse else norm

    df['delta_score'] = safe_normalize(df['net_delta'])
    df['vega_score'] = safe_normalize(df['net_vega'])
    df['efficiency_score'] = safe_normalize(df['efficiency'])
    df['total_score'] = (df['delta_score'] * 0.40 + df['efficiency_score'] * 0.40 + df['vega_score'] * 0.20)
    return df.sort_values('total_score', ascending=False)


# -------------------------------
# Enhanced Dashboard with Vega Focus
# -------------------------------
def create_bullish_dashboard(results, ticker, analysis_summary):
    if results is None or results.empty: return ""
    best = results.iloc[0]
    cost_display = f"${abs(best['net_cost']):.2f} {'CREDIT' if best['net_cost'] < 0 else 'DEBIT'}"
    cost_color = 'positive' if best['net_cost'] < 0 else 'negative'

    summary_html = "<ul>"
    for exp, count in analysis_summary.items():
        summary_html += f"<li><b>{exp}:</b> Found {count} valid trades</li>"
    summary_html += "</ul>"

    num_results = len(results)
    table_title = f"Top {min(num_results, 5)} Combinations (Max 3 Per Expiration)"

    html_template = """
    <!DOCTYPE html><html><head><title>{ticker} Bullish Strategy Report</title><style>body{{font-family:system-ui,sans-serif;margin:20px;background:#f9f9f9;color:#333}}.container{{max-width:1100px;margin:0 auto;background:white;padding:30px;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.08)}}h1{{color:#27ae60;text-align:center}}h2{{border-bottom:2px solid #2ecc71;padding-bottom:8px}}table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#2ecc71;color:white;text-align:center}}td{{text-align:center}}.strategy-box{{background:#f0fff4;border:1px solid #27ae60;padding:20px;margin:20px 0;border-radius:8px}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin:20px 0}}.metric{{background:#ecf0f1;padding:15px;border-radius:5px;text-align:center}}.positive{{color:#27ae60;font-weight:bold}}.negative{{color:#c0392b;font-weight:bold}}.danger{{background:#fff0f0;border:1px solid #c0392b;padding:15px;border-radius:5px}}</style></head><body><div class="container"><h1>🐂 {ticker} Bullish Risk Reversal Report</h1><div class="strategy-box"><h2>🎯 Top Recommended Trade</h2><div class="metric-grid"><div class="metric"><strong>Expiration</strong><br>{best_exp} ({best_dte} days)</div><div class="metric"><strong>Strikes</strong><br>Long Call: ${best_call_K:.2f}<br>Short Put: ${best_put_K:.2f}</div><div class="metric"><strong>Net Cost</strong><br><span class="{cost_color}">{cost_display}</span></div><div class="metric"><strong>Breakeven</strong><br>${best_breakeven:.2f}</div><div class="metric"><strong>Net Vega</strong><br><span class="positive">{best_vega:.3f}</span></div><div class="metric"><strong>Efficiency</strong><br><span class="positive">{best_efficiency:.1%}</span></div></div></div><div class="danger"><h3>⚠️ Strategy Overview & Risk</h3><p>A Bullish Risk Reversal (Long OTM Call, Short OTM Put) creates a synthetic long stock position with low or zero cost. The primary risk is the short put. If the stock price falls below ${best_put_K:.2f}, you may be assigned 100 shares per contract at that price. Maximum loss is up to ${max_loss:.2f} per share if the stock goes to zero.</p></div><h2>🔎 Analysis Summary</h2><div>{summary_html}</div><h2>📊 {table_title}</h2><table><tr><th>Rank</th><th>Expiration</th><th>Strikes (Call/Put)</th><th>Net Cost</th><th>Net Vega</th><th>Efficiency</th><th>Score</th></tr>{table_rows}</table></div></body></html>
    """
    table_rows = ""
    for i, row in results.head(5).iterrows():
        cost_txt = f"${abs(row['net_cost']):.2f} {'CR' if row['net_cost'] < 0 else 'DB'}"
        table_rows += f"<tr><td>{i + 1}</td><td>{row['expiration']}</td><td>${row['long_call_strike']:.2f} / ${row['short_put_strike']:.2f}</td><td>{cost_txt}</td><td>{row['net_vega']:.3f}</td><td>{row['efficiency']:.1%}</td><td>{row['total_score']:.3f}</td></tr>"

    return html_template.format(
        ticker=ticker,
        best_exp=best['expiration'],
        best_dte=best['days_to_exp'],
        best_call_K=best['long_call_strike'],
        best_put_K=best['short_put_strike'],
        cost_color=cost_color,
        cost_display=cost_display,
        best_breakeven=best['breakeven'],
        best_vega=best['net_vega'],
        best_efficiency=best['efficiency'],
        max_loss=best['max_loss_down'],
        summary_html=summary_html,
        table_title=table_title,
        table_rows=table_rows
    )


# -------------------------------
# Main Execution
# -------------------------------
def run_bullish_analysis(ticker):
    print(f"🚀 Starting Bullish Analysis for {ticker}")
    try:
        while True:
            try:
                min_dte = int(input("Enter MINIMUM days to expiration (e.g., 30): "))
                max_dte = int(input("Enter MAXIMUM days to expiration (e.g., 180): "))
                if 0 <= min_dte < max_dte:
                    break
                else:
                    print("Invalid range. Minimum DTE must be less than Maximum DTE.")
            except ValueError:
                print("Invalid input. Please enter whole numbers.")

        stock = yf.Ticker(ticker)
        underlying_price = stock.history(period='1d')['Close'].iloc[-1]
        print(f"📊 Current {ticker} price: ${underlying_price:.2f}")

        today = datetime.now()
        valid_expirations = [exp for exp in stock.options if
                             min_dte <= (datetime.strptime(exp, "%Y-%m-%d") - today).days <= max_dte]

        if not valid_expirations:
            print(f"❌ No expirations found between {min_dte} and {max_dte} days.")
            return

        exp_to_analyze = valid_expirations[:3]
        print(
            f"🎯 Found {len(valid_expirations)} expirations. Analyzing the first {len(exp_to_analyze)}: {exp_to_analyze}")

        all_combinations = []
        analysis_summary = {}
        for expiration in exp_to_analyze:
            print(f"\n⚡ Analyzing expiration: {expiration}...")
            calls, puts = get_options_data(ticker, expiration, underlying_price)
            if calls.empty or puts.empty:
                print(f"    - No suitable OTM options data found after cleaning.")
                analysis_summary[expiration] = 0
                continue

            combinations = analyze_bullish_risk_reversal(calls, puts, underlying_price, expiration)
            analysis_summary[expiration] = len(combinations)
            if combinations:
                print(f"    ✅ Found {len(combinations)} potential combinations.")
                all_combinations.extend(combinations)
            else:
                print(f"    - No valid combinations met the strategy criteria.")

        if not all_combinations:
            print("\n❌ Analysis complete. No strategies were found in the specified range.")
            return

        print(f"\n🌐 Ranking {len(all_combinations)} total combinations...")
        ranked_results = rank_combinations(all_combinations)

        print(f"📊 Filtering results for diversity (max 3 per expiration)...")
        final_results = ranked_results.groupby('expiration').head(3).sort_values('total_score',
                                                                                 ascending=False).reset_index(drop=True)

        if final_results.empty:
            print("\n❌ No valid strategies remained after filtering.")
            return

        html_content = create_bullish_dashboard(final_results, ticker, analysis_summary)
        filename = f"{ticker.lower()}_bullish_report.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"\n✅ Report saved: {filename}")

        print("🚀 Opening report in your browser...")
        try:
            webbrowser.get('chrome').open_new_tab('file://' + os.path.realpath(filename))
        except webbrowser.Error:
            print("Could not find Google Chrome. Opening in default browser instead.")
            webbrowser.open_new_tab('file://' + os.path.realpath(filename))

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🐂 Bullish Risk Reversal Strategy Analyzer")
    print("==========================================================")
    ticker_input = input("Enter a ticker symbol (e.g., AAPL, TSLA, SPY): ").upper().strip()
    if ticker_input:
        run_bullish_analysis(ticker_input)