from flask import Flask, render_template, request, jsonify
from analysis_engine import generate_bullish_report_html, generate_bearish_report_html

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/bullish_analyzer')
def bullish_analyzer():
    return render_template('bullish_analyzer.html')

@app.route('/bearish_analyzer')
def bearish_analyzer():
    return render_template('bearish_analyzer.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    ticker = data.get('ticker')
    min_dte = int(data.get('min_dte'))
    max_dte = int(data.get('max_dte'))
    analysis_type = data.get('analysis_type', 'bullish')
    
    # Call the appropriate function based on analysis type
    if analysis_type == 'bullish':
        report_html = generate_bullish_report_html(ticker, min_dte, max_dte)
    elif analysis_type == 'bearish':
        report_html = generate_bearish_report_html(ticker, min_dte, max_dte)
    else:
        report_html = "<p>Invalid analysis type specified.</p>"
    
    return jsonify({'report_html': report_html})

if __name__ == '__main__':
    app.run(debug=True) 