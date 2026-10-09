"""The Piotroski F-Score.

Joseph Piotroski's score (2000) awards one point for each of nine pass/fail
accounting signals, grouped into three themes. A score of 8-9 marks a
fundamentally strong firm; 0-2 marks a weak one. The score is computed from two
consecutive fiscal years.

Profitability (4 points)
    1. Positive net income (ROA > 0)
    2. Positive operating cash flow
    3. ROA improved year over year
    4. Cash flow quality: operating cash flow exceeds net income (low accruals)

Leverage, liquidity & funding (3 points)
    5. Long-term debt ratio fell
    6. Current ratio rose
    7. No new shares issued

Operating efficiency (2 points)
    8. Gross margin improved
    9. Asset turnover improved
"""
from __future__ import annotations

from dataclasses import dataclass

SIGNALS = [
    "positive_roa",
    "positive_cfo",
    "roa_improved",
    "accruals_quality",
    "lower_leverage",
    "higher_current_ratio",
    "no_dilution",
    "higher_gross_margin",
    "higher_asset_turnover",
]


@dataclass
class YearData:
    """One fiscal year's figures."""

    net_income: float
    cash_from_operations: float
    total_assets: float
    long_term_debt: float
    current_assets: float
    current_liabilities: float
    shares_outstanding: float
    sales: float
    cogs: float

    def roa(self) -> float:
        return self.net_income / self.total_assets

    def leverage(self) -> float:
        return self.long_term_debt / self.total_assets

    def current_ratio(self) -> float:
        if self.current_liabilities == 0:
            raise ValueError("current_liabilities must be non-zero")
        return self.current_assets / self.current_liabilities

    def gross_margin(self) -> float:
        return (self.sales - self.cogs) / self.sales

    def asset_turnover(self) -> float:
        return self.sales / self.total_assets


@dataclass
class FScoreResult:
    f_score: int
    signals: dict[str, bool]

    @property
    def strength(self) -> str:
        if self.f_score >= 8:
            return "strong"
        if self.f_score <= 2:
            return "weak"
        return "moderate"


def _validate(y: YearData, label: str) -> None:
    if y.total_assets == 0:
        raise ValueError(f"{label}: total_assets must be non-zero")
    if y.sales == 0:
        raise ValueError(f"{label}: sales must be non-zero")


def compute_signals(prior: YearData, current: YearData) -> dict[str, bool]:
    _validate(prior, "prior")
    _validate(current, "current")
    p, c = prior, current

    return {
        "positive_roa": c.net_income > 0,
        "positive_cfo": c.cash_from_operations > 0,
        "roa_improved": c.roa() > p.roa(),
        # Accruals: operating cash flow per asset beats earnings per asset.
        "accruals_quality": (c.cash_from_operations / c.total_assets) > c.roa(),
        "lower_leverage": c.leverage() < p.leverage(),
        "higher_current_ratio": c.current_ratio() > p.current_ratio(),
        "no_dilution": c.shares_outstanding <= p.shares_outstanding,
        "higher_gross_margin": c.gross_margin() > p.gross_margin(),
        "higher_asset_turnover": c.asset_turnover() > p.asset_turnover(),
    }


def compute_f_score(prior: YearData, current: YearData) -> FScoreResult:
    """Compute the 0-9 Piotroski F-Score from two consecutive years."""
    signals = compute_signals(prior, current)
    return FScoreResult(f_score=sum(signals.values()), signals=signals)

