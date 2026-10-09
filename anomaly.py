"""Optional, interpretable PCA candidates for manual financial-data review.

This module does not change fundamental scores or research status. An unusual
cross-feature relationship is not proof of error, fraud, or an investment signal.
The fit and its empirical threshold use the same cohort: no predictive performance
or out-of-sample reliability is claimed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


FEATURES = ("roe", "cash_conversion", "core_earnings_yield", "revenue_cagr_2y")
MIN_COMPLETE_COHORT = 12
METHOD = (
    "Within market/sector, complete-case median/IQR scaling, mean-centered PCA via "
    "SVD, at most two components and strictly fewer than active features. Squared "
    "reconstruction error above the empirical in-sample 95th percentile prompts "
    "manual data review only; it does not establish error or fraud."
)


def _features(company: dict[str, Any]) -> tuple[list[float] | None, list[str]]:
    metrics = company.get("metrics")
    if not isinstance(metrics, dict):
        return None, list(FEATURES)
    values: list[float] = []
    missing: list[str] = []
    for feature in FEATURES:
        value = metrics.get(feature)
        try:
            if isinstance(value, bool) or value is None:
                raise ValueError
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            missing.append(feature)
        else:
            values.append(parsed)
    return (None if missing else values), missing


def _review(cohort: tuple[str, str], complete_count: int,
            skip_reason: str | None = None) -> dict[str, Any]:
    return {
        "method": METHOD,
        "market": cohort[0], "sector": cohort[1],
        "status": "skipped" if skip_reason else "evaluated",
        "cohort_size": complete_count,
        "reconstruction_error": None, "threshold": None,
        "review_candidate": False, "skip_reason": skip_reason,
        "threshold_scope": "empirical_in_sample_95th_percentile",
        "active_features": [], "dropped_features": {}, "retained_components": None,
        "squared_residual_by_feature": {},
    }


def annotate_anomalies(companies: list[dict[str, Any]]) -> dict[str, Any]:
    """Return ``annotations`` keyed by ticker and a ``summary``; never mutate input.

    The caller may attach each annotation as a company's ``ml_data_review``.
    Missing/nonfinite observations are not imputed. Entire constant columns and
    columns with zero IQR are excluded and explicitly recorded. At least twelve
    complete cases and two remaining features are required. NumPy is optional.
    """
    cohorts: dict[tuple[str, str], list[tuple[str, list[float] | None]]] = defaultdict(list)
    annotations: dict[str, dict[str, Any]] = {}
    missing_by_ticker: dict[str, list[str]] = {}
    for company in companies:
        if not isinstance(company, dict):
            raise TypeError("each company must be a dictionary")
        ticker = company.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("each company requires a nonblank ticker")
        ticker = ticker.strip()
        if ticker in missing_by_ticker:
            raise ValueError(f"duplicate ticker: {ticker}")
        cohort = (str(company.get("market") or "").strip(),
                  str(company.get("sector") or "").strip())
        values, missing = _features(company)
        missing_by_ticker[ticker] = missing
        cohorts[cohort].append((ticker, values))

    try:
        import numpy as np
    except ImportError:
        np = None

    cohort_summary: list[dict[str, Any]] = []
    for cohort, entries in sorted(cohorts.items()):
        complete = [(ticker, values) for ticker, values in entries if values is not None]
        count = len(complete)
        group_reason = (
            "missing_market_or_sector" if not all(cohort) else
            "numpy_not_available" if np is None else
            "cohort_has_fewer_than_12_complete_companies" if count < MIN_COMPLETE_COHORT else
            None
        )
        group = {"market": cohort[0], "sector": cohort[1], "company_count": len(entries),
                 "complete_count": count, "status": "skipped", "skip_reason": group_reason,
                 "active_features": [], "dropped_features": {}}
        for ticker, values in entries:
            reason = "missing_or_nonfinite_features" if values is None else group_reason
            annotations[ticker] = _review(cohort, count, reason)
            if missing_by_ticker[ticker]:
                annotations[ticker]["missing_features"] = missing_by_ticker[ticker]
        if group_reason:
            cohort_summary.append(group)
            continue

        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                matrix = np.array([values for _, values in complete], dtype=float)
                medians = np.median(matrix, axis=0)
                q25, q75 = np.percentile(matrix, [25, 75], axis=0)
                iqr = q75 - q25
                active: list[int] = []
                dropped: dict[str, str] = {}
                for column, feature in enumerate(FEATURES):
                    if np.all(matrix[:, column] == matrix[0, column]):
                        dropped[feature] = "constant"
                    elif iqr[column] <= 0:
                        # A zero-IQR column can contain distinct values. Do not
                        # silently describe it as constant or divide by zero.
                        dropped[feature] = "zero_iqr"
                    else:
                        active.append(column)
                active_names = [FEATURES[column] for column in active]
                group.update(active_features=active_names, dropped_features=dropped)
                for ticker, _ in complete:
                    annotations[ticker].update(active_features=active_names, dropped_features=dropped)
                if len(active) < 2:
                    group_reason = "fewer_than_two_features_with_positive_iqr"
                else:
                    scaled = (matrix[:, active] - medians[active]) / iqr[active]
                    centered = scaled - np.mean(scaled, axis=0)
                    if not np.all(np.isfinite(centered)):
                        raise FloatingPointError("nonfinite standardized values")
                    _, singular_values, directions = np.linalg.svd(centered, full_matrices=False)
                    tolerance = np.finfo(float).eps * max(centered.shape) * singular_values[0]
                    numerical_rank = int(np.sum(singular_values > tolerance))
                    components = min(2, len(active) - 1, numerical_rank)
                    if components < 1:
                        group_reason = "no_nonzero_principal_components"
                    else:
                        basis = directions[:components]
                        reconstruction = (centered @ basis.T) @ basis
                        residual_squared = (centered - reconstruction) ** 2
                        errors = np.sum(residual_squared, axis=1)
                        threshold = float(np.percentile(errors, 95))
                        # Numerical noise in a perfectly reconstructed low-rank
                        # cohort must not create an arbitrary review candidate.
                        noise_floor = float(100 * np.finfo(float).eps *
                                            max(1.0, float(np.max(np.sum(centered ** 2, axis=1)))))
                        if not math.isfinite(threshold):
                            raise FloatingPointError("nonfinite reconstruction error")
                        group.update(status="evaluated", skip_reason=None,
                                     retained_components=components, threshold=threshold)
                        for row, (ticker, _) in enumerate(complete):
                            annotations[ticker].update(
                                reconstruction_error=float(errors[row]), threshold=threshold,
                                review_candidate=bool(errors[row] > threshold + noise_floor),
                                retained_components=components,
                                squared_residual_by_feature={
                                    feature: float(residual_squared[row, column])
                                    for column, feature in enumerate(active_names)
                                },
                            )
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            group_reason = "numerical_fit_failure"
        if group_reason:
            group.update(status="skipped", skip_reason=group_reason)
            for ticker, _ in complete:
                annotations[ticker].update(status="skipped", skip_reason=group_reason,
                                           review_candidate=False)
        cohort_summary.append(group)

    summary = {
        "method": METHOD, "features": list(FEATURES), "minimum_complete_cohort": MIN_COMPLETE_COHORT,
        "numpy_available": np is not None, "company_count": len(annotations),
        "evaluated_count": sum(item["status"] == "evaluated" for item in annotations.values()),
        "skipped_count": sum(item["status"] == "skipped" for item in annotations.values()),
        "review_candidate_count": sum(item["review_candidate"] for item in annotations.values()),
        "cohorts": cohort_summary,
        "limitations": (
            "Exploratory, in-sample data-quality triage. PCA can absorb unusual observations "
            "and miss anomalies. A 95th-percentile threshold is not a calibrated probability, "
            "out-of-sample accuracy, or evidence of fraud. Inspect source filings and metric "
            "definitions. Scores, status, and investment decisions are unaffected."
        ),
    }
    return {"annotations": annotations, "summary": summary}
