"""
app.py - Module B dashboard: Strategic Portfolio Stress Testing.
Run:  streamlit run app.py
"""
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio import build_portfolio
from shocks import BASE_SHOCKS, DEFAULT_TRIGGER_TYPES, build_shock
from signals_client import fetch_signals, to_scenarios, triggers
from stress import run_stress, summarize

DATA = Path(os.environ.get("DATA_DIR", "data"))
st.set_page_config(page_title="Portfolio Stress Test", layout="wide")
CLASS_COLORS = {"Loans": "#4C78A8", "Bonds": "#72B7B2", "Equities": "#F58518", "Derivatives": "#B279A2"}
money = lambda v: f"${v/1e6:,.1f}m"


@st.cache_data(show_spinner=False)
def get_portfolio():
    return build_portfolio(DATA / "prices.csv", DATA / "transactions_sample.csv", DATA / "mcc_codes.json")


@st.cache_data(ttl=30, show_spinner=False)
def get_signals(api_url, local_path):
    return fetch_signals(api_url, local_path)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("Risk engine feed")
api_url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
sig_path = st.sidebar.text_input("Fallback file", str(DATA / "signals.jsonl"))
threshold = st.sidebar.slider("Trigger when impact score ≥", 1.0, 10.0, 6.5, 0.1,
                              help="Spec example is >7. Live signals currently peak near 7.6, so 6.5 shows more activity.")
types = st.sidebar.multiselect("Event types that trigger", list(BASE_SHOCKS), DEFAULT_TRIGGER_TYPES)
live_only = st.sidebar.checkbox("Live news only (exclude Kaggle CSV)", True)
combine = st.sidebar.checkbox("Combine all triggered event types", False,
                              help="Off: stress the single strongest event. On: add all event shocks together (harsher).")
mode = st.sidebar.radio("Scenario source", ["Auto (from engine signals)", "Manual what-if"])
if st.sidebar.button("Refresh signals"):
    st.cache_data.clear()

pf = get_portfolio()
sigs, origin = get_signals(api_url, sig_path)
trig = triggers(sigs, threshold, types, live_only)

if mode.startswith("Auto"):
    scenarios = to_scenarios(trig)
    if not combine:
        scenarios = sorted(scenarios, key=lambda x: -x[1])[:1]
else:
    st.sidebar.subheader("Manual scenario")
    ev = st.sidebar.multiselect("Events", list(BASE_SHOCKS), ["Geopolitical"])
    imp = st.sidebar.slider("Impact score", 1.0, 10.0, 8.0, 0.1)
    scenarios = [(e, imp) for e in ev]

shock = build_shock(scenarios)

# ------------------------------------------------------------------ header + KPIs
st.title("Strategic Portfolio Stress Testing")
st.caption(f"Signals from {origin} · {len(sigs):,} signals loaded · prices as of {pf['price_date']}")

base = run_stress(pf, build_shock([]))
res = run_stress(pf, shock)
s, s0 = summarize(res), summarize(base)

if scenarios:
    st.error(f"Stress test TRIGGERED by: {shock.label}")
else:
    st.success("No signal above the trigger threshold. Portfolio shown unstressed.")

c = st.columns(4)
c[0].metric("Value before", money(s["before"]))
c[1].metric("Value after", money(s["after"]), f"{s['pct']:.2%}")
c[2].metric("Stress P&L", money(s["pnl"]))
worst = s["by_class"]["pnl"].idxmin() if scenarios else "-"
c[3].metric("Hardest-hit asset class", worst)

# ------------------------------------------------------------------ before vs after + waterfall
left, right = st.columns(2)
bc = s["by_class"]
fig = go.Figure()
fig.add_bar(name="Before", x=bc.index, y=bc["before"] / 1e6, marker_color="#9AA5B1")
fig.add_bar(name="After stress", x=bc.index, y=bc["after"] / 1e6, marker_color="#D6454B")
fig.update_layout(barmode="group", title="Value by asset class ($m)", yaxis_title="$m", height=380)
left.plotly_chart(fig, use_container_width=True)

order = list(bc.index)
wf = go.Figure(go.Waterfall(
    x=["Before"] + order + ["After"], measure=["absolute"] + ["relative"] * len(order) + ["total"],
    y=[s["before"] / 1e6] + list(bc["pnl"] / 1e6) + [0],
    increasing_marker_color="#2E9E6B", decreasing_marker_color="#D6454B", totals_marker_color="#4C78A8"))
lo = min(s["before"], s["after"]) / 1e6
wf.update_layout(title="Portfolio value bridge ($m)", height=380, showlegend=False,
                 yaxis=dict(range=[lo * 0.985, s["before"] / 1e6 * 1.005]))
right.plotly_chart(wf, use_container_width=True)

# ------------------------------------------------------------------ shock applied + per-event contribution
a, b = st.columns([1, 1])
with a:
    st.subheader("Shocks applied")
    st.table(pd.DataFrame({
        "Factor": ["Equities", "Rates (bp)", "Credit spreads (bp)", "Oil"],
        "Shock": [f"{shock.eq:+.1%}", f"{shock.rate_bp:+.0f}", f"{shock.spread_bp:+.0f}", f"{shock.oil:+.1%}"]}).set_index("Factor"))
with b:
    st.subheader("Loss by triggering event")
    if scenarios:
        contrib = pd.DataFrame({f"{e} ({i:.1f})": [run_stress(pf, build_shock([(e, i)]))["pnl"].sum() / 1e6]
                                for e, i in scenarios}).T.rename(columns={0: "P&L ($m)"})
        st.bar_chart(contrib)
        st.caption("Individual events do not sum exactly to the combined P&L (convexity and shared factors).")
    else:
        st.info("Nothing triggered.")

# ------------------------------------------------------------------ signals + positions
st.subheader("Triggering signals")
if trig.empty:
    st.write("None. Lower the threshold or add event types in the sidebar.")
else:
    st.dataframe(trig[["timestamp", "source", "event_type", "impact_score", "direction", "headline"]].head(50),
                 use_container_width=True, hide_index=True)

with st.expander("Positions (before / after)"):
    pos = res.assign(before=lambda d: d.before / 1e6, after=lambda d: d.after / 1e6, pnl=lambda d: d.pnl / 1e6)
    st.dataframe(pos.round(3).rename(columns={"before": "before $m", "after": "after $m", "pnl": "P&L $m"}),
                 use_container_width=True, hide_index=True)
with st.expander("Loan book by sector"):
    sec = res[res.asset_class == "Loans"].groupby("sector")[["before", "pnl"]].sum() / 1e6
    st.bar_chart(sec["pnl"])
with st.expander("Method and assumptions"):
    st.markdown("""
- **Severity** scales shocks: `clip(impact/7, 0.5, 1.5)`. Base shocks are in `shocks.py` and are assumptions.
- **Bonds**: duration + spread duration + convexity. **Equities**: beta, sector add-ons, oil beta for energy.
- **Loans**: expected-loss increase (`LGD × ΔPD`, sector-specific) plus half-weight fair-value spread effect.
- **Derivatives**: pay-fixed swap gains when rates rise; long oil future follows oil.
- Loans come from transaction-sample merchants (sector from MCC); ratings are randomly assigned (seeded).
- Multiple triggered event types are combined by adding their scaled shocks.
- Simplified conceptual model, not a regulatory stress test.""")
