"""
Standalone DOB Variations Generator (no project dependencies)

Generates DOB variations to match validator-style categories:
- ±1 day, ±3 days, ±30 days, ±90 days, ±365 days
- year+month (YYYY-MM)
"""

import random
from datetime import datetime, timedelta
from typing import List


def predict_dob_category_score(
    seed_dob: str, dob_variations: List[str], dob_weight: float = 0.1
) -> float:
    """Score = fraction of required categories covered * dob_weight.

    Required: ±1, ±3, ±30, ±90, ±365, year+month.
    """
    seed = _parse_date(seed_dob)
    found = set()
    for v in dob_variations:
        # try full date
        try:
            d = _parse_date(v)
            diff = abs((d - seed).days)
            if diff <= 1:
                found.add("±1")
            elif diff <= 3:
                found.add("±3")
            elif diff <= 30:
                found.add("±30")
            elif diff <= 90:
                found.add("±90")
            elif diff <= 365:
                found.add("±365")
        except Exception:
            pass
        # try year-month
        try:
            ym = datetime.strptime(v, "%Y-%m")
            if ym.year == seed.year and ym.month == seed.month:
                found.add("ym")
        except Exception:
            pass
    total = 6  # five ranges + year-month
    return dob_weight * (len(found) / total)


RANGES = [1, 3, 30, 90, 365]  # day offsets
EXTRA_RANGES = [
    5,
    10,
    15,
    20,
    25,
    35,
    40,
    45,
    50,
    55,
    65,
    75,
    85,
    95,
    100,
    105,
]  # day extra offsets to prevent duplicate dob


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _year_month_only(date_str: str) -> str:
    dt = _parse_date(date_str)
    return dt.strftime("%Y-%m")


def generate_dob_variations(
    seed_dob: str,
    expected_count: int = 10,
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> List[str]:
    """Generate DOB variations covering required categories.

    Args:
        seed_dob: YYYY-MM-DD
        expected_count: number of items to return (cap)
    """
    if not expected_count:
        return []

    base_date = _parse_date(seed_dob)
    variations: List[str] = []

    # 1) Guarantee coverage: one representative per required category (use +days), then year+month
    guaranteed: List[str] = []
    for days in RANGES:
        v = _fmt_date(base_date + timedelta(days=days))
        if v not in guaranteed:
            guaranteed.append(v)

    ym = _year_month_only(seed_dob)
    if ym not in guaranteed:
        guaranteed.append(ym)

    # 2) Add additional variants (mirror -days) as fillers if more are requested
    fillers: List[str] = []
    for days in RANGES:
        v = _fmt_date(base_date - timedelta(days=days))
        if v not in guaranteed and v not in fillers:
            fillers.append(v)

    # 3) Add extra variants
    extras: List[str] = []
    for days in EXTRA_RANGES:
        v = _fmt_date(base_date - timedelta(days=days))
        if v not in guaranteed and v not in fillers and v not in extras:
            fillers.append(v)

        v = _fmt_date(base_date + timedelta(days=days))
        if v not in guaranteed and v not in fillers and v not in extras:
            fillers.append(v)

    # Stable randomized ordering with salt
    rng_seed = (hash(seed_dob) & 0xFFFFFFFF) ^ (miner_salt * 131) ^ (batch_salt * 911)

    rng = random.Random(rng_seed)
    rng.shuffle(guaranteed)
    rng.shuffle(fillers)
    variations = guaranteed + fillers + extras

    if not variations:
        return []

    for _ in range(expected_count // len(variations) + 1):
        variations.extend(variations)
    return variations[:expected_count]


if __name__ == "__main__":
    dobs = generate_dob_variations("1990-01-15", 20)
    print(len(dobs), len(set(dobs)))
