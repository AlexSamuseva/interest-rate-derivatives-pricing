"""Shared period-unit normalization helpers.

These utilities keep DTCC-style period-unit aliases consistent between
payment schedule generation and DTCC row parsing.
"""

from __future__ import annotations

PERIOD_UNIT_ALIASES = {
    "YEAR": "YEAR",
    "YEARS": "YEAR",
    "YR": "YEAR",
    "YRS": "YEAR",
    "MNTH": "MNTH",
    "MONTH": "MNTH",
    "MONTHS": "MNTH",
    "MTH": "MNTH",
    "MTHS": "MNTH",
    "WEEK": "WEEK",
    "WEEKS": "WEEK",
    "WK": "WEEK",
    "WKS": "WEEK",
    "DAIL": "DAIL",
    "DAILY": "DAIL",
    "DAY": "DAIL",
    "DAYS": "DAIL",
}


def normalize_period_unit(raw: object) -> str | None:
    """Normalize a period unit alias to the project standard code."""

    if raw is None:
        return None

    raw_text = str(raw).strip().upper()
    return PERIOD_UNIT_ALIASES.get(raw_text)
