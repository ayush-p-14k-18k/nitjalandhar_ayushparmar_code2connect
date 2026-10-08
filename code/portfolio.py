"""
portfolio.py - builds the synthetic wholesale-banking book.

Loans        : top merchants in the transaction sample become corporate borrowers.
               Exposure is proportional to total spend; sector comes from the MCC code.
Bonds        : TLT / LQD / HYG units (priced from prices.csv) with duration + spread duration.
Equities     : JPM, GS, XOM, AAPL, MSFT held directly.
Derivatives  : a pay-fixed interest-rate swap and a long oil future (value = mark-to-market).
The transaction data only supplies realistic exposure sizes and sectors; ratings and the
asset mix are assumptions (the datasets contain no loans or bonds).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

LOAN_BOOK = 2_000_000_000
BOND_BOOK = 1_500_000_000
EQUITY_BOOK = 1_000_000_000

BONDS = {  # ticker: (weight, duration, spread duration, convexity)
    "TLT": (0.30, 16.5, 0.0, 300.0),
    "LQD": (0.45, 8.5, 8.0, 90.0),
    "HYG": (0.25, 3.5, 3.5, 20.0),
}
EQUITIES = {  # ticker: (weight, beta to market shock, sector, oil beta)
    "JPM": (0.25, 1.15, "Financials", 0.0),
    "GS": (0.20, 1.35, "Financials", 0.0),
    "XOM": (0.20, 0.80, "Energy", 0.40),
    "AAPL": (0.20, 1.10, "Technology", 0.0),
    "MSFT": (0.15, 1.05, "Technology", 0.0),
}
RATINGS = {  # rating: (1y PD, spread bp)
    "AA": (0.0002, 60), "A": (0.0006, 90), "BBB": (0.0018, 150),
    "BB": (0.009, 300), "B": (0.035, 500),
}
LGD = 0.45
LOAN_DURATION = 3.0
IRS = dict(notional=500_000_000, duration=4.4, base_mtm=2_500_000)   # pay-fixed: gains when rates rise
OIL_FUT = dict(barrels=1_000_000)                                      # long WTI future (CL=F)


def mcc_to_sector(mcc: int) -> str:
    m = int(mcc)
    if m in (5541, 4900): return "Energy"
    if m in (3780, 3684, 5045, 5732, 4814, 4899, 5815, 5816): return "Technology"
    if m in (4111, 4121, 4214, 4511, 4411, 3722, 3775, 3771, 4112, 4131, 4784, 3730): return "Transport"
    if 3000 <= m < 4000 or m in (3596, 3509): return "Industrials"
    if m in (5411, 5499, 5812, 5814, 5813, 5921, 5310, 5300): return "Staples"
    if m in (8099, 8021, 8011, 8041, 8043, 8049, 8062, 5912): return "Healthcare"
    if m in (5211, 5251, 1711, 5712, 5719, 5722, 5261, 3144, 3174): return "Construction"
    if m in (6300, 4829, 8931, 7276, 8111): return "Financials"
    if m in (7011, 7996, 7832, 7995, 7922, 7801, 7802, 4722): return "Hospitality"
    if 5000 <= m < 6000: return "Consumer"
    return "Services"


def load_prices(path) -> pd.DataFrame:
    p = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return p.dropna(axis=1, how="all").ffill()    # last row is partial (pre-market) -> forward fill


def build_loans(tx_path, mcc_path, n_loans=25, seed=7) -> pd.DataFrame:
    tx = pd.read_csv(tx_path)
    tx["amt"] = tx["amount"].astype(str).str.replace("$", "", regex=False).astype(float)
    tx = tx[tx["amt"] > 0]                                      # drop refunds
    mccs = json.load(open(mcc_path))
    g = (tx.groupby("merchant_id").agg(spend=("amt", "sum"), mcc=("mcc", lambda s: s.mode().iat[0]),
                                       n_tx=("amt", "size"), clients=("client_id", "nunique"))
           .sort_values("spend", ascending=False).head(n_loans).reset_index())
    w = (g["spend"] / g["spend"].sum()).clip(upper=0.12)         # concentration cap
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    g["rating"] = rng.choice(list(RATINGS), size=len(g), p=[0.05, 0.25, 0.40, 0.22, 0.08])
    g["sector"] = g["mcc"].map(mcc_to_sector)
    g["description"] = g["mcc"].astype(str).map(mccs).fillna("Other")
    g["notional"] = (w * LOAN_BOOK).round(0)
    g["pd"] = g["rating"].map(lambda r: RATINGS[r][0])
    g["name"] = "Loan: " + g["description"].str.slice(0, 28) + " #" + g["merchant_id"].astype(str)
    g["asset_class"] = "Loans"
    g["value"] = g["notional"]                                   # carried at par
    return g[["asset_class", "name", "sector", "rating", "pd", "notional", "value", "mcc", "n_tx"]]


def build_portfolio(prices_path, tx_path, mcc_path) -> dict:
    prices = load_prices(prices_path)
    last = prices.iloc[-1]
    loans = build_loans(tx_path, mcc_path)
    bonds = pd.DataFrame([dict(asset_class="Bonds", name=f"{t} (ETF proxy)", ticker=t,
                               units=round(w * BOND_BOOK / last[t]), price=last[t],
                               duration=d, spread_dur=sd, convexity=c)
                          for t, (w, d, sd, c) in BONDS.items()])
    bonds["value"] = bonds["units"] * bonds["price"]
    eq = pd.DataFrame([dict(asset_class="Equities", name=t, ticker=t, sector=s, beta=b, oil_beta=ob,
                            units=round(w * EQUITY_BOOK / last[t]), price=last[t])
                       for t, (w, b, s, ob) in EQUITIES.items()])
    eq["value"] = eq["units"] * eq["price"]
    oil = float(last["CL=F"]) if "CL=F" in last and pd.notna(last.get("CL=F")) else 80.0
    deriv = pd.DataFrame([
        dict(asset_class="Derivatives", name="Pay-fixed IRS (5y, $500m)", kind="irs",
             notional=IRS["notional"], duration=IRS["duration"], value=IRS["base_mtm"]),
        dict(asset_class="Derivatives", name="Long WTI future (1m bbl)", kind="oil",
             notional=OIL_FUT["barrels"] * oil, barrels=OIL_FUT["barrels"], price=oil, value=0.0)])
    ctx = {k: float(last[k]) for k in ("CL=F", "GC=F", "^VIX") if k in last and pd.notna(last[k])}
    return dict(loans=loans, bonds=bonds, equities=eq, derivatives=deriv, market=ctx,
                price_date=str(prices.index[-1].date()))
