# Module B - Strategic Portfolio Stress Testing

    pip install -r requirements.txt
    # put transactions_sample.csv in data/ (prices.csv, mcc_codes.json, signals.jsonl are already there)
    uvicorn api:app --reload        # in the risk_engine folder (optional; the app falls back to data/signals.jsonl)
    streamlit run app.py

Files: shocks.py (event -> shocks), portfolio.py (synthetic book), stress.py (revaluation),
signals_client.py (API subscription + trigger rule), app.py (dashboard).
