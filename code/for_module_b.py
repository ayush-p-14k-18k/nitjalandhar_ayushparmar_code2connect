# import pandas as pd
# df = pd.read_csv("data/transactions_data.csv", nrows=100000)
# df.to_csv("transactions_sample.csv", index=False)

import yfinance as yf

tickers = [
    "JPM", "GS", "XOM", "AAPL", "MSFT",
    "TLT", "LQD", "HYG", "ATNX", "CL=F",
    "GC=F", "^VIX"
]

data = yf.download(
    tickers,
    period="1y",
    threads=False
)["Close"]

data.to_csv("prices.csv")

print("Download completed!")
print(data.head())
print(data.shape)