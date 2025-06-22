import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings('ignore')


# -------------------------------
# Black-Scholes and Data Functions
# -------------------------------
def d1(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0: return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_vega(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0: return 0
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T) / 100


def bs_delta(S, K, T, r, sigma, option_type='call', q=0.0):
    if T <= 0: return 1.0 if S > K and option_type == 'call' else (-1.0 if S < K and option_type == 'put' else 0.0)
    D1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(D1) if option_type == 'call' else np.exp(-q * T) * (norm.cdf(D1) - 1)


def get_options_data(ticker, expiration, underlying_price):
    try:
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration)
    except Exception as e:
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
        return []

    combinations = []
    for _, call in otm_calls.iterrows():
        for _, put in otm_puts.iterrows():
            if call['strike'] <= put['strike']: continue
            combo = create_bullish_strategy_combination(call, put)
            if combo and is_valid_bullish_combo(combo):
                combo.update({'strategy_type': 'Bullish Risk Reversal', 'expiration': expiration_date,
                              'days_to_exp': days_to_exp})
                combinations.append(combo)
    return combinations


def analyze_bearish_risk_reversal(calls, puts, underlying_price, expiration_date, risk_free_rate=0.045,
                                  dividend_yield=0.0):
    today = pd.to_datetime(datetime.now().date())
    exp_date = pd.to_datetime(expiration_date)
    T = max((exp_date - today).days / 365.0, 1 / (365 * 24))
    days_to_exp = (exp_date - today).days

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
        return []

    combinations = []
    for _, put in otm_puts.iterrows():
        for _, call in otm_calls.iterrows():
            if put['strike'] >= call['strike']: continue
            combo = create_bearish_strategy_combination(put, call)
            if combo and is_valid_bearish_combo(combo):
                combo.update({'strategy_type': 'Bearish Risk Reversal', 'expiration': expiration_date,
                              'days_to_exp': days_to_exp})
                combinations.append(combo)
    return combinations


def create_bullish_strategy_combination(call_row, put_row):
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


def create_bearish_strategy_combination(put_row, call_row):
    try:
        net_cost = put_row['ask'] - call_row['bid']
        strike_diff = call_row['strike'] - put_row['strike']
        efficiency = -net_cost / strike_diff if strike_diff > 0 else 0
        return {
            'long_put_strike': put_row['strike'], 'short_call_strike': call_row['strike'],
            'net_cost': net_cost, 'iv_advantage': call_row['impliedVolatility'] - put_row['impliedVolatility'],
            'net_delta': put_row['delta'] - call_row['delta'], 'net_vega': put_row['vega'] - call_row['vega'],
            'max_loss_up': call_row['strike'] + (call_row['bid'] - put_row['ask']),
            'breakeven': put_row['strike'] - net_cost,
            'efficiency': efficiency
        }
    except (KeyError, TypeError):
        return None


def is_valid_bullish_combo(combo):
    if not combo: return False
    if abs(combo['net_cost']) > 20: return False
    if combo['net_delta'] <= 0.1 or combo['net_vega'] <= 0: return False
    return True


def is_valid_bearish_combo(combo):
    if not combo: return False
    if abs(combo['net_cost']) > 20: return False
    if combo['net_delta'] >= -0.1 or combo['net_vega'] > 0.01: return False
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


def rank_bearish_combinations(combinations):
    if not combinations: return []
    df = pd.DataFrame(combinations)

    def safe_normalize(series, reverse=False):
        if series.std() == 0 or len(series) < 2: return pd.Series([0.5] * len(series), index=series.index)
        norm = (series - series.min()) / (series.max() - series.min())
        return 1 - norm if reverse else norm

    # For bearish, we want negative delta (more negative is better), low absolute vega, and good efficiency
    df['delta_score'] = safe_normalize(df['net_delta'], reverse=True)  # More negative delta is better
    df['vega_score'] = safe_normalize(df['net_vega'].abs(), reverse=True)  # Lower absolute vega is better
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


def create_bearish_dashboard(results, ticker, analysis_summary):
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
    <!DOCTYPE html><html><head><title>{ticker} Bearish Strategy Report</title><style>body{{font-family:system-ui,sans-serif;margin:20px;background:#f9f9f9;color:#333}}.container{{max-width:1100px;margin:0 auto;background:white;padding:30px;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.08)}}h1{{color:#e74c3c;text-align:center}}h2{{border-bottom:2px solid #e74c3c;padding-bottom:8px}}table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#e74c3c;color:white;text-align:center}}td{{text-align:center}}.strategy-box{{background:#fdf2f2;border:1px solid #e74c3c;padding:20px;margin:20px 0;border-radius:8px}}.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;margin:20px 0}}.metric{{background:#ecf0f1;padding:15px;border-radius:5px;text-align:center}}.positive{{color:#27ae60;font-weight:bold}}.negative{{color:#c0392b;font-weight:bold}}.danger{{background:#fff0f0;border:1px solid #c0392b;padding:15px;border-radius:5px}}</style></head><body><div class="container"><h1>🐻 {ticker} Bearish Risk Reversal Report</h1><div class="strategy-box"><h2>🎯 Top Recommended Trade</h2><div class="metric-grid"><div class="metric"><strong>Expiration</strong><br>{best_exp} ({best_dte} days)</div><div class="metric"><strong>Strikes</strong><br>Long Put: ${best_put_K:.2f}<br>Short Call: ${best_call_K:.2f}</div><div class="metric"><strong>Net Cost</strong><br><span class="{cost_color}">{cost_display}</span></div><div class="metric"><strong>Breakeven</strong><br>${best_breakeven:.2f}</div><div class="metric"><strong>Net Vega</strong><br><span class="negative">{best_vega:.3f}</span></div><div class="metric"><strong>Efficiency</strong><br><span class="positive">{best_efficiency:.1%}</span></div></div></div><div class="danger"><h3>⚠️ Strategy Overview & Risk</h3><p>A Bearish Risk Reversal (Long OTM Put, Short OTM Call) creates a synthetic short stock position with low or zero cost. The primary risk is the short call. If the stock price rises above ${best_call_K:.2f}, you may be assigned 100 shares per contract at that price. Maximum loss is unlimited if the stock continues to rise.</p></div><h2>🔎 Analysis Summary</h2><div>{summary_html}</div><h2>📊 {table_title}</h2><table><tr><th>Rank</th><th>Expiration</th><th>Strikes (Put/Call)</th><th>Net Cost</th><th>Net Vega</th><th>Efficiency</th><th>Score</th></tr>{table_rows}</table></div></body></html>
    """
    table_rows = ""
    for i, row in results.head(5).iterrows():
        cost_txt = f"${abs(row['net_cost']):.2f} {'CR' if row['net_cost'] < 0 else 'DB'}"
        table_rows += f"<tr><td>{i + 1}</td><td>{row['expiration']}</td><td>${row['long_put_strike']:.2f} / ${row['short_call_strike']:.2f}</td><td>{cost_txt}</td><td>{row['net_vega']:.3f}</td><td>{row['efficiency']:.1%}</td><td>{row['total_score']:.3f}</td></tr>"

    return html_template.format(
        ticker=ticker,
        best_exp=best['expiration'],
        best_dte=best['days_to_exp'],
        best_put_K=best['long_put_strike'],
        best_call_K=best['short_call_strike'],
        cost_color=cost_color,
        cost_display=cost_display,
        best_breakeven=best['breakeven'],
        best_vega=best['net_vega'],
        best_efficiency=best['efficiency'],
        max_loss=best['max_loss_up'],
        summary_html=summary_html,
        table_title=table_title,
        table_rows=table_rows
    )


# -------------------------------
# Main Analysis Functions
# -------------------------------
def generate_bullish_report_html(ticker: str, min_dte: int, max_dte: int, cache) -> str:
    """
    Generate a bullish risk reversal analysis report for the given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        cache: Flask-Caching cache object
    
    Returns:
        HTML string containing the analysis report
    """
    @cache.memoize(timeout=300)
    def _generate_bullish_report(ticker: str, min_dte: int, max_dte: int) -> str:
        logging.info(f"--- CACHE MISS --- Running bullish analysis for {ticker} ({min_dte}-{max_dte} DTE)")
        try:
            stock = yf.Ticker(ticker)
            underlying_price = stock.history(period='1d')['Close'].iloc[-1]

            today = datetime.now()
            valid_expirations = [exp for exp in stock.options if
                                 min_dte <= (datetime.strptime(exp, "%Y-%m-%d") - today).days <= max_dte]

            if not valid_expirations:
                return "<p>No expirations found in the specified date range.</p>"

            exp_to_analyze = valid_expirations[:3]

            all_combinations = []
            analysis_summary = {}
            for expiration in exp_to_analyze:
                calls, puts = get_options_data(ticker, expiration, underlying_price)
                if calls.empty or puts.empty:
                    analysis_summary[expiration] = 0
                    continue

                combinations = analyze_bullish_risk_reversal(calls, puts, underlying_price, expiration)
                analysis_summary[expiration] = len(combinations)
                if combinations:
                    all_combinations.extend(combinations)

            if not all_combinations:
                return "<p>No valid strategies were found in the specified range.</p>"

            ranked_results = rank_combinations(all_combinations)

            final_results = ranked_results.groupby('expiration').head(3).sort_values('total_score',
                                                                                     ascending=False).reset_index(drop=True)

            if final_results.empty:
                return "<p>No valid strategies remained after filtering.</p>"

            html_content = create_bullish_dashboard(final_results, ticker, analysis_summary)
            return html_content
            
        except Exception as e:
            return f"<p>An error occurred: {e}</p>"
    
    return _generate_bullish_report(ticker, min_dte, max_dte)


def generate_bearish_report_html(ticker: str, min_dte: int, max_dte: int, cache) -> str:
    """
    Generate a bearish risk reversal analysis report for the given ticker and date range.
    
    Args:
        ticker: Stock ticker symbol
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        cache: Flask-Caching cache object
    
    Returns:
        HTML string containing the analysis report
    """
    @cache.memoize(timeout=300)
    def _generate_bearish_report(ticker: str, min_dte: int, max_dte: int) -> str:
        logging.info(f"--- CACHE MISS --- Running bearish analysis for {ticker} ({min_dte}-{max_dte} DTE)")
        try:
            stock = yf.Ticker(ticker)
            underlying_price = stock.history(period='1d')['Close'].iloc[-1]

            today = datetime.now()
            valid_expirations = [exp for exp in stock.options if
                                 min_dte <= (datetime.strptime(exp, "%Y-%m-%d") - today).days <= max_dte]

            if not valid_expirations:
                return "<p>No expirations found in the specified date range.</p>"

            exp_to_analyze = valid_expirations[:3]

            all_combinations = []
            analysis_summary = {}
            for expiration in exp_to_analyze:
                calls, puts = get_options_data(ticker, expiration, underlying_price)
                if calls.empty or puts.empty:
                    analysis_summary[expiration] = 0
                    continue

                combinations = analyze_bearish_risk_reversal(calls, puts, underlying_price, expiration)
                analysis_summary[expiration] = len(combinations)
                if combinations:
                    all_combinations.extend(combinations)

            if not all_combinations:
                return "<p>No valid strategies were found in the specified range.</p>"

            ranked_results = rank_bearish_combinations(all_combinations)

            final_results = ranked_results.groupby('expiration').head(3).sort_values('total_score',
                                                                                     ascending=False).reset_index(drop=True)

            if final_results.empty:
                return "<p>No valid strategies remained after filtering.</p>"

            html_content = create_bearish_dashboard(final_results, ticker, analysis_summary)
            return html_content
            
        except Exception as e:
            return f"<p>An error occurred: {e}</p>"
    
    return _generate_bearish_report(ticker, min_dte, max_dte)


# Legacy function for backward compatibility
def generate_report_html(ticker: str, min_dte: int, max_dte: int) -> str:
    return generate_bullish_report_html(ticker, min_dte, max_dte, None) 