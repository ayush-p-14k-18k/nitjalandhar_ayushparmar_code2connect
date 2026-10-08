"""
stress.py - revalues the portfolio under a Shock.

Bonds        dP/P = -D*dy - SD*ds + 0.5*C*dy^2
Loans        dV = -notional * (LGD * dPD)  - notional * D_loan * ds * 0.5
             (expected-loss increase + half-weight fair-value spread effect; PD uplift is sector-specific)
Equities     move = eq*beta + sector add-on + oil_beta*oil
Derivatives  IRS: dV = notional * duration * dy (pay-fixed gains if rates rise)
             Oil future: dV = barrels * price * oil move
"""
import pandas as pd
from portfolio import LGD, LOAN_DURATION
from shocks import Shock


def run_stress(pf: dict, shock: Shock) -> pd.DataFrame:
    dy, ds = shock.rate_bp / 1e4, shock.spread_bp / 1e4
    rows = []

    for r in pf["loans"].itertuples():
        d_pd = min(r.pd * shock.pd_uplift.get(r.sector, 0.0), 1 - r.pd)
        pnl = -r.notional * (LGD * d_pd) - r.notional * LOAN_DURATION * ds * 0.5
        rows.append((r.asset_class, r.name, r.sector, r.value, pnl))

    for r in pf["bonds"].itertuples():
        pct = -r.duration * dy - r.spread_dur * ds + 0.5 * r.convexity * dy ** 2
        rows.append((r.asset_class, r.name, "Fixed income", r.value, r.value * pct))

    for r in pf["equities"].itertuples():
        move = shock.eq * r.beta + shock.eq_sector_add.get(r.sector, 0.0) + r.oil_beta * shock.oil
        rows.append((r.asset_class, r.name, r.sector, r.value, r.value * max(move, -1.0)))

    for r in pf["derivatives"].itertuples():
        pnl = r.notional * r.duration * dy if r.kind == "irs" else r.barrels * r.price * shock.oil
        rows.append((r.asset_class, r.name, "Rates" if r.kind == "irs" else "Commodity", r.value, pnl))

    df = pd.DataFrame(rows, columns=["asset_class", "position", "sector", "before", "pnl"])
    df["after"] = df["before"] + df["pnl"]
    return df


def summarize(df: pd.DataFrame) -> dict:
    b, a = df["before"].sum(), df["after"].sum()
    by = df.groupby("asset_class")[["before", "after", "pnl"]].sum()
    return dict(before=b, after=a, pnl=a - b, pct=(a - b) / b, by_class=by)
