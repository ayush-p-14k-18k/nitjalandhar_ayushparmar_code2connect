"""
shocks.py - maps an event type (+ impact score) to market / credit shocks.

All base shocks are calibrated for an event with impact score 7. They are scaled by
severity = clip(impact / 7, 0.5, 1.5), so a 9.0 event hits ~1.29x as hard and a 5.0 event ~0.71x.
These numbers are ASSUMPTIONS for a conceptual tool - edit the table to run your own scenarios.
"""
from dataclasses import dataclass, field

SECTORS = ["Energy", "Transport", "Industrials", "Technology", "Consumer", "Staples",
           "Healthcare", "Construction", "Financials", "Hospitality", "Services"]

# eq: equity move | rate_bp: yield change | spread_bp: credit-spread widening | oil: oil move
# pd_uplift: extra loan default probability as a fraction of base PD (1.0 = PD doubles)
# eq_sector_add: extra equity move for a sector | pd_sector_mult: scales pd_uplift for loan sectors
BASE_SHOCKS = {
    "Geopolitical": dict(
        eq=-0.10, rate_bp=-25, spread_bp=100, oil=0.15, pd_uplift=0.50,
        eq_sector_add={}, pd_sector_mult={"Energy": -0.5, "Transport": 2.0, "Hospitality": 1.5, "Industrials": 1.2}),
    "Macroeconomic": dict(
        eq=-0.08, rate_bp=150, spread_bp=60, oil=-0.05, pd_uplift=0.80,
        eq_sector_add={"Technology": -0.03}, pd_sector_mult={"Consumer": 1.5, "Hospitality": 1.5, "Construction": 1.8, "Staples": 0.5, "Healthcare": 0.5}),
    "Credit Event": dict(
        eq=-0.12, rate_bp=-25, spread_bp=200, oil=0.0, pd_uplift=1.50,
        eq_sector_add={"Financials": -0.08}, pd_sector_mult={"Financials": 1.8, "Construction": 1.3}),
    "M&A": dict(
        eq=-0.01, rate_bp=0, spread_bp=10, oil=0.0, pd_uplift=0.0, eq_sector_add={}, pd_sector_mult={}),
    "Product Launch": dict(
        eq=0.0, rate_bp=0, spread_bp=0, oil=0.0, pd_uplift=0.0, eq_sector_add={}, pd_sector_mult={}),
}
DEFAULT_TRIGGER_TYPES = ["Geopolitical", "Macroeconomic", "Credit Event"]


@dataclass
class Shock:
    """Combined shock applied to the portfolio (sums of the scaled event shocks)."""
    eq: float = 0.0
    rate_bp: float = 0.0
    spread_bp: float = 0.0
    oil: float = 0.0
    eq_sector_add: dict = field(default_factory=dict)
    pd_uplift: dict = field(default_factory=dict)      # sector -> fractional PD increase
    label: str = "No stress"


def severity(impact: float) -> float:
    return min(max(impact / 7.0, 0.5), 1.5)


def build_shock(scenarios, table=None) -> Shock:
    """scenarios: list of (event_type, impact_score). Several events are added together."""
    table = table or BASE_SHOCKS
    s = Shock()
    names = []
    for ev, impact in scenarios:
        b = table.get(ev)
        if not b:
            continue
        k = severity(impact)
        s.eq += b["eq"] * k
        s.rate_bp += b["rate_bp"] * k
        s.spread_bp += b["spread_bp"] * k
        s.oil += b["oil"] * k
        for sec, v in b["eq_sector_add"].items():
            s.eq_sector_add[sec] = s.eq_sector_add.get(sec, 0.0) + v * k
        for sec in SECTORS:
            m = b["pd_sector_mult"].get(sec, 1.0)
            s.pd_uplift[sec] = s.pd_uplift.get(sec, 0.0) + b["pd_uplift"] * m * k
        names.append(f"{ev} ({impact:.1f})")
    s.label = " + ".join(names) if names else "No stress"
    return s
