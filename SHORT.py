# CORRECTED Long Put + Short Call Strategy Analyzer
# This is a BEARISH DIRECTIONAL strategy with IV risk hedging, NOT income generation!

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# -------------------------------
# Black-Scholes Functions
# -------------------------------

def d1(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0:
        return 0
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma, q=0):
    if T <= 0:
        return max(S - K, 0)
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def bs_put_price(S, K, T, r, sigma, q=0):
    if T <= 0:
        return max(K - S, 0)
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * np.exp(-q * T) * norm.cdf(-D1)


def bs_vega(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0:
        return 0
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T) / 100


def bs_delta(S, K, T, r, sigma, option_type='call', q=0):
    if T <= 0:
        return 0
    D1 = d1(S, K, T, r, sigma, q)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(D1)
    else:
        return np.exp(-q * T) * (norm.cdf(D1) - 1)


def bs_theta(S, K, T, r, sigma, option_type='call', q=0):
    if T <= 0:
        return 0
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        return (-S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(D2)
                + q * S * np.exp(-q * T) * norm.cdf(D1)) / 365
    else:
        return (-S * np.exp(-q * T) * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-D2)
                - q * S * np.exp(-q * T) * norm.cdf(-D1)) / 365


# -------------------------------
# Enhanced Options Data Fetching
# -------------------------------

def get_options_data(ticker, expiration, underlying_price):
    """Get and clean options data"""
    try:
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration)
    except Exception as e:
        print(f"Error fetching data for {ticker} at expiration {expiration}: {e}")
        return None, None

    calls = opt_chain.calls.copy()
    puts = opt_chain.puts.copy()

    # Clean data
    for df in [calls, puts]:
        df.dropna(subset=['impliedVolatility', 'bid', 'ask'], inplace=True)
        df = df[(df['volume'] > 0) | (df['openInterest'] > 0)]
        df = df[df['impliedVolatility'] > 0.01]
        df = df[df['bid'] > 0]
        df = df[(df['ask'] - df['bid']) / df['ask'] < 0.6]  # Reasonable spreads

    # Add calculated fields
    calls['moneyness'] = calls['strike'] / underlying_price
    puts['moneyness'] = puts['strike'] / underlying_price
    calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
    puts['mid_price'] = (puts['bid'] + puts['ask']) / 2

    # Add expiration to each row
    calls['expiration'] = expiration
    puts['expiration'] = expiration

    return calls, puts


# -------------------------------
# CORRECTED Strategy Analysis
# -------------------------------

def analyze_bearish_iv_hedge_strategy(calls, puts, underlying_price, expiration_date,
                                      risk_free_rate=0.045, dividend_yield=0.0):
    """
    Analyze Long Put + Short Call as a BEARISH DIRECTIONAL strategy with IV hedge
    This is NOT income generation - it's a leveraged bearish bet!
    """

    today = pd.to_datetime(datetime.now().date())
    exp_date = pd.to_datetime(expiration_date)
    T = max((exp_date - today).days / 365, 1 / 365)
    days_to_exp = (exp_date - today).days

    print(f"   Time to expiration: {days_to_exp} days ({T:.3f} years)")

    # Calculate Greeks for all options
    for df, opt_type in [(calls, 'call'), (puts, 'put')]:
        df['vega'] = df.apply(lambda row: bs_vega(underlying_price, row['strike'], T,
                                                  risk_free_rate, row['impliedVolatility'], dividend_yield), axis=1)
        df['delta'] = df.apply(lambda row: bs_delta(underlying_price, row['strike'], T,
                                                    risk_free_rate, row['impliedVolatility'], opt_type, dividend_yield),
                               axis=1)
        df['theta'] = df.apply(lambda row: bs_theta(underlying_price, row['strike'], T,
                                                    risk_free_rate, row['impliedVolatility'], opt_type, dividend_yield),
                               axis=1)

    combinations = []

    # Strategy 1: PURE BEARISH - ATM/ITM Puts + ATM/OTM Calls
    print("   🐻 Strategy 1: Pure Bearish Directional")
    bearish_puts = puts[puts['moneyness'] >= 0.90]  # ATM to ITM puts for max bearish exposure
    hedge_calls = calls[(calls['moneyness'] >= 1.00) & (calls['moneyness'] <= 1.15)]  # ATM to moderate OTM calls

    for _, put in bearish_puts.iterrows():
        for _, call in hedge_calls.iterrows():
            combo = create_strategy_combination(put, call, 'Pure Bearish Directional', underlying_price, T, days_to_exp)
            if combo and is_valid_bearish_combo(combo):
                combinations.append(combo)

    # Strategy 2: IV CRUSH HEDGE - High IV calls vs reasonable puts
    print("   ⚡ Strategy 2: IV Crush Hedge")
    if len(calls) > 0 and len(puts) > 0:
        # Target high IV calls to short (benefit from IV crush)
        high_iv_calls = calls[calls['impliedVolatility'] >= calls['impliedVolatility'].quantile(0.6)]
        reasonable_puts = puts[puts['impliedVolatility'] <= puts['impliedVolatility'].quantile(0.7)]

        for _, call in high_iv_calls.head(15).iterrows():
            for _, put in reasonable_puts.iterrows():
                combo = create_strategy_combination(put, call, 'IV Crush Hedge', underlying_price, T, days_to_exp)
                if combo and is_valid_bearish_combo(combo):
                    combinations.append(combo)

    # Strategy 3: SYNTHETIC SHORT - Maximum bearish exposure
    print("   📉 Strategy 3: Synthetic Short Position")
    # Deep ITM puts + ATM calls for maximum bearish leverage
    deep_puts = puts[puts['moneyness'] <= 0.95]
    atm_calls = calls[(calls['moneyness'] >= 0.98) & (calls['moneyness'] <= 1.02)]

    for _, put in deep_puts.head(10).iterrows():
        for _, call in atm_calls.iterrows():
            combo = create_strategy_combination(put, call, 'Synthetic Short', underlying_price, T, days_to_exp)
            if combo and combo['net_delta'] < -0.4:  # Strong bearish bias required
                combinations.append(combo)

    return combinations


def create_strategy_combination(put_row, call_row, strategy_type, underlying_price, T, days_to_exp):
    """Create combination with proper bearish strategy context"""
    try:
        # LONG PUT (bullish on volatility, bearish on direction)
        # SHORT CALL (bearish on volatility, bearish on direction)

        # Cost calculation: Pay put ask, receive call bid
        net_debit = put_row['ask'] - call_row['bid']  # Usually a debit for bearish strategies

        # Greeks: Long put + Short call
        net_vega = put_row['vega'] - call_row['vega']  # Usually negative (short vega)
        net_delta = put_row['delta'] - call_row['delta']  # Negative (bearish)
        net_theta = put_row['theta'] - call_row['theta']  # Usually positive (time decay benefit)

        # Risk analysis
        max_profit_down = put_row['strike'] + net_debit if net_debit < 0 else put_row['strike'] - abs(net_debit)
        max_loss_up = abs(net_debit) if net_debit < 0 else 0  # Limited loss if credit received
        breakeven = put_row['strike'] + net_debit if net_debit < 0 else put_row['strike'] - abs(net_debit)

        # Upside risk (calls can be assigned)
        upside_risk = "UNLIMITED" if call_row['strike'] < underlying_price * 1.1 else "HIGH"

        # IV advantage calculation
        iv_advantage = call_row['impliedVolatility'] - put_row['impliedVolatility']

        # Bearish efficiency score
        bearish_score = abs(net_delta) * 0.4 + (1 / (abs(net_debit) + 0.1)) * 0.3 + max(iv_advantage, 0) * 0.3

        return {
            'strategy_type': strategy_type,
            'expiration': put_row.get('expiration', call_row.get('expiration', 'Unknown')),
            'days_to_exp': days_to_exp,

            # Long Put Details
            'long_put_symbol': put_row['contractSymbol'],
            'long_put_strike': put_row['strike'],
            'long_put_bid': put_row['bid'],
            'long_put_ask': put_row['ask'],
            'long_put_iv': put_row['impliedVolatility'],
            'long_put_delta': put_row['delta'],
            'long_put_vega': put_row['vega'],
            'long_put_volume': put_row.get('volume', 0),
            'long_put_oi': put_row.get('openInterest', 0),

            # Short Call Details
            'short_call_symbol': call_row['contractSymbol'],
            'short_call_strike': call_row['strike'],
            'short_call_bid': call_row['bid'],
            'short_call_ask': call_row['ask'],
            'short_call_iv': call_row['impliedVolatility'],
            'short_call_delta': call_row['delta'],
            'short_call_vega': call_row['vega'],
            'short_call_volume': call_row.get('volume', 0),
            'short_call_oi': call_row.get('openInterest', 0),

            # Net Position
            'net_debit': net_debit,
            'net_vega': net_vega,
            'net_delta': net_delta,
            'net_theta': net_theta,

            # Risk Metrics
            'max_profit_down': max_profit_down,
            'max_loss_up': max_loss_up,
            'breakeven': breakeven,
            'upside_risk': upside_risk,
            'iv_advantage': iv_advantage,
            'bearish_score': bearish_score,
            'underlying_price': underlying_price,
        }
    except Exception as e:
        print(f"Error creating combination: {e}")
        return None


def is_valid_bearish_combo(combo):
    """Validate combinations for bearish directional strategy"""
    if not combo:
        return False

    # Must be bearish (negative delta)
    if combo['net_delta'] >= -0.1:
        return False

    # Reasonable cost (don't overpay)
    if combo['net_debit'] > 3.0:  # Don't pay more than $3.00
        return False

    # IV metrics should be reasonable
    if combo['long_put_iv'] <= 0 or combo['short_call_iv'] <= 0:
        return False

    # Some liquidity required
    total_volume = combo['long_put_volume'] + combo['short_call_volume']
    total_oi = combo['long_put_oi'] + combo['short_call_oi']
    if total_volume == 0 and total_oi < 10:
        return False

    return True


def rank_combinations(combinations):
    """Rank combinations by bearish strategy effectiveness"""
    if not combinations:
        return []

    df = pd.DataFrame(combinations)

    # Normalize scoring components
    def safe_normalize(series, reverse=False):
        if series.std() == 0 or len(series) == 0:
            return pd.Series([0.5] * len(series))
        normalized = (series - series.min()) / (series.max() - series.min())
        return (1 - normalized) if reverse else normalized

    # Scoring components for bearish strategy
    df['delta_score'] = safe_normalize(-df['net_delta'])  # More negative delta = better
    df['cost_score'] = safe_normalize(-df['net_debit'])  # Lower cost = better
    df['iv_score'] = safe_normalize(df['iv_advantage'])  # IV advantage = better
    df['vega_score'] = safe_normalize(-df['net_vega'])  # Negative vega preferred for IV hedge
    df['bearish_score_norm'] = safe_normalize(df['bearish_score'])

    # Liquidity score
    df['liquidity_score'] = safe_normalize(
        np.log1p(df['long_put_volume'] + df['short_call_volume']) +
        np.log1p(df['long_put_oi'] + df['short_call_oi'])
    )

    # Total score weighted for bearish directional strategy
    df['total_score'] = (
            df['delta_score'] * 0.30 +  # Delta exposure most important
            df['cost_score'] * 0.20 +  # Cost efficiency
            df['iv_score'] * 0.20 +  # IV advantage
            df['vega_score'] * 0.15 +  # Vega hedge
            df['liquidity_score'] * 0.15  # Liquidity
    )

    return df.sort_values('total_score', ascending=False)


# -------------------------------
# Main Analysis Function
# -------------------------------

def analyze_ticker(ticker, expiration_date, underlying_price):
    """Analyze ticker for bearish directional strategy"""
    print(f"🐻 Analyzing {ticker} for BEARISH DIRECTIONAL strategy")
    print(f"   Expiration: {expiration_date}")

    calls, puts = get_options_data(ticker, expiration_date, underlying_price)

    if calls is None or puts is None or calls.empty or puts.empty:
        print(f"❌ No valid options data for {expiration_date}")
        return None

    print(f"   📊 Found {len(calls)} calls and {len(puts)} puts")

    combinations = analyze_bearish_iv_hedge_strategy(calls, puts, underlying_price, expiration_date)

    if not combinations:
        print(f"   ❌ No valid bearish combinations found")
        return None

    print(f"   ✅ Generated {len(combinations)} valid combinations")

    ranked_df = rank_combinations(combinations)
    return ranked_df


def create_corrected_dashboard(results, ticker, underlying_price):
    """Create corrected HTML dashboard"""
    if results is None or results.empty:
        print("❌ No results to display")
        return

    best = results.iloc[0]
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Determine credit/debit display
    cost_display = f"${abs(best['net_debit']):.2f} {'CREDIT' if best['net_debit'] < 0 else 'DEBIT'}"
    cost_color = 'positive' if best['net_debit'] < 0 else 'negative'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{ticker} Bearish Directional Strategy Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #c74545; text-align: center; margin-bottom: 10px; }}
            .subtitle {{ text-align: center; color: #666; font-size: 18px; margin-bottom: 30px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #c74545; padding-bottom: 10px; }}
            .strategy-box {{ background: #ffebee; border: 2px solid #c74545; padding: 20px; margin: 20px 0; border-radius: 8px; }}
            .trade-details {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .option-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px 15px; background: #e9ecef; border-radius: 5px; }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; font-weight: bold; }}
            .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            .danger {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; margin: 15px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
            th {{ background: #c74545; color: white; }}
            tr:nth-child(even) {{ background: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐻 {ticker} BEARISH DIRECTIONAL STRATEGY</h1>
            <div class="subtitle">
                Long Put + Short Call | Current Price: ${underlying_price:.2f} | Generated: {current_time}
            </div>

            <div class="strategy-box">
                <h2>🎯 RECOMMENDED BEARISH TRADE</h2>
                <p><strong>Strategy Purpose:</strong> Leveraged bearish directional bet with IV risk hedging</p>
                <p><strong>Market Outlook:</strong> Expecting significant downward movement in {ticker}</p>

                <div class="trade-details">
                    <div class="option-box">
                        <h3>📉 LONG PUT (Bearish Direction)</h3>
                        <p><strong>Contract:</strong> {best['long_put_symbol']}</p>
                        <p><strong>Strike:</strong> ${best['long_put_strike']:.0f}</p>
                        <p><strong>Ask Price:</strong> ${best['long_put_ask']:.2f}</p>
                        <p><strong>IV:</strong> {best['long_put_iv']:.1%}</p>
                        <p><strong>Delta:</strong> {best['long_put_delta']:.3f}</p>
                        <p><strong>Volume/OI:</strong> {best['long_put_volume']:.0f}/{best['long_put_oi']:.0f}</p>
                    </div>

                    <div class="option-box">
                        <h3>📈 SHORT CALL (IV Hedge)</h3>
                        <p><strong>Contract:</strong> {best['short_call_symbol']}</p>
                        <p><strong>Strike:</strong> ${best['short_call_strike']:.0f}</p>
                        <p><strong>Bid Price:</strong> ${best['short_call_bid']:.2f}</p>
                        <p><strong>IV:</strong> {best['short_call_iv']:.1%}</p>
                        <p><strong>Delta:</strong> {best['short_call_delta']:.3f}</p>
                        <p><strong>Volume/OI:</strong> {best['short_call_volume']:.0f}/{best['short_call_oi']:.0f}</p>
                    </div>
                </div>

                <div style="margin: 20px 0;">
                    <div class="metric">
                        <strong>Net Cost:</strong> 
                        <span class="{cost_color}">{cost_display}</span>
                    </div>
                    <div class="metric">
                        <strong>Net Delta:</strong> 
                        <span class="negative">{best['net_delta']:.3f}</span>
                    </div>
                    <div class="metric">
                        <strong>Net Vega:</strong> 
                        <span class="{'negative' if best['net_vega'] < 0 else 'positive'}">{best['net_vega']:.3f}</span>
                    </div>
                    <div class="metric">
                        <strong>Days to Exp:</strong> {best['days_to_exp']}
                    </div>
                    <div class="metric">
                        <strong>Breakeven:</strong> ${best['breakeven']:.2f}
                    </div>
                    <div class="metric">
                        <strong>IV Advantage:</strong> {best['iv_advantage']:.1%}
                    </div>
                </div>
            </div>

            <div class="warning">
                <h3>📋 TRADE EXECUTION</h3>
                <ol>
                    <li><strong>BUY TO OPEN:</strong> {best['long_put_symbol']} (Put @ ${best['long_put_strike']:.0f})</li>
                    <li><strong>SELL TO OPEN:</strong> {best['short_call_symbol']} (Call @ ${best['short_call_strike']:.0f})</li>
                    <li><strong>Monitor:</strong> Delta exposure and IV changes</li>
                    <li><strong>Profit Target:</strong> Stock moves below ${best['breakeven']:.2f}</li>
                    <li><strong>Risk Management:</strong> Close if stock rallies above call strike</li>
                </ol>
            </div>

            <div class="danger">
                <h3>⚠️ CRITICAL RISK WARNINGS</h3>
                <ul>
                    <li><strong>UNLIMITED UPSIDE RISK:</strong> Losses can be unlimited if stock rallies significantly</li>
                    <li><strong>Assignment Risk:</strong> Short call can be assigned at any time</li>
                    <li><strong>Directional Risk:</strong> This is NOT a neutral strategy - you need the stock to fall</li>
                    <li><strong>Time Decay:</strong> Position loses value if stock stays flat</li>
                    <li><strong>IV Risk:</strong> Changes in implied volatility affect both legs differently</li>
                </ul>
                <p><strong>This analysis is for educational purposes only. Options trading involves significant risk.</strong></p>
            </div>

            <h2>📊 Top 5 Combinations</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Strategy</th>
                    <th>Put Strike</th>
                    <th>Call Strike</th>
                    <th>Expiration</th>
                    <th>Net Cost</th>
                    <th>Net Delta</th>
                    <th>Net Vega</th>
                    <th>Breakeven</th>
                    <th>Score</th>
                </tr>
    """

    # Add top 5 combinations
    for i, (_, row) in enumerate(results.head(5).iterrows()):
        cost_text = f"${abs(row['net_debit']):.2f} {'CR' if row['net_debit'] < 0 else 'DB'}"
        html += f"""
                <tr>
                    <td>{i + 1}</td>
                    <td>{row['strategy_type']}</td>
                    <td>${row['long_put_strike']:.0f}</td>
                    <td>${row['short_call_strike']:.0f}</td>
                    <td>{row['expiration']}</td>
                    <td>{cost_text}</td>
                    <td>{row['net_delta']:.3f}</td>
                    <td>{row['net_vega']:.3f}</td>
                    <td>${row['breakeven']:.2f}</td>
                    <td>{row['total_score']:.3f}</td>
                </tr>
        """

    html += """
            </table>
        </div>
    </body>
    </html>
    """

    return html


# -------------------------------
# Main Execution
# -------------------------------

def run_corrected_analysis(ticker):
    """Run the corrected bearish directional analysis"""
    print(f"🚀 Starting CORRECTED Bearish Directional Analysis for {ticker}")

    try:
        # Get current stock price
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if hist.empty:
            print(f"❌ Could not fetch price data for {ticker}")
            return

        underlying_price = hist['Close'].iloc[-1]
        all_expirations = stock.options

        print(f"📊 Current {ticker} price: ${underlying_price:.2f}")

        # Filter for reasonable expirations (30-90 days)
        today = datetime.now()
        valid_expirations = []

        for exp in all_expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            days_to_exp = (exp_date - today).days
            if 30 <= days_to_exp <= 90:
                valid_expirations.append(exp)

        if not valid_expirations:
            print("❌ No suitable expirations found (need 30-90 days)")
            return

        print(f"🎯 Found {len(valid_expirations)} valid expirations")

        # Analyze first available expiration
        expiration = valid_expirations[0]
        print(f"\n⚡ Analyzing expiration: {expiration}")

        results = analyze_ticker(ticker, expiration, underlying_price)

        if results is None or results.empty:
            print("❌ No valid strategies found")
            return

        # Create corrected dashboard
        html_content = create_corrected_dashboard(results, ticker, underlying_price)

        # Save dashboard
        filename = f"{ticker.lower()}_corrected_bearish_strategy.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n✅ CORRECTED dashboard saved: {filename}")

        # Print summary
        best = results.iloc[0]
        print(f"\n🎯 BEST BEARISH STRATEGY:")
        print(f"   Strategy: {best['strategy_type']}")
        print(f"   Long Put: ${best['long_put_strike']:.0f} @ ${best['long_put_ask']:.2f}")
        print(f"   Short Call: ${best['short_call_strike']:.0f} @ ${best['short_call_bid']:.2f}")
        print(f"   Net Cost: ${abs(best['net_debit']):.2f} {'CREDIT' if best['net_debit'] < 0 else 'DEBIT'}")
        print(f"   Net Delta: {best['net_delta']:.3f} (Bearish)")
        print(f"   Breakeven: ${best['breakeven']:.2f}")
        print(f"   Score: {best['total_score']:.3f}")

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()


# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    print("🐻 CORRECTED Long Put + Short Call Strategy Analyzer")
    print("This is a BEARISH DIRECTIONAL strategy with IV risk hedging")
    print("=" * 60)

    ticker = input("Enter ticker symbol (e.g., AAPL, TSLA, SPY): ").upper().strip()
    if ticker:
        run_corrected_analysis(ticker)