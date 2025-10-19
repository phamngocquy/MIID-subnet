#!/usr/bin/env python3
"""
Standalone demo: generate name/DOB/address variations and predict total score
using validator-like weights.

Weights (aligned with validator):
- name:    0.3 (phonetic + orthographic)
- DOB:     0.1 (coverage of required categories)
- address: 0.6 (fraction of valid same-region addresses)
"""

from typing import Dict, List, Tuple

from standalone_variations.name_variations import (
    generate_name_variations,
    predict_name_score,
)
from standalone_variations.dob_variations import (
    generate_dob_variations,
    predict_dob_category_score,
)
from standalone_variations.address_variations import (
    generate_address_variations,
    predict_address_score,
    validate_address_variation,
)


def generate_and_predict(
    original_name: str,
    seed_dob: str,
    seed_address: str,
    expected_count: int = 10,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Generate variations and predict an overall validator-like score.

    Returns:
        (variations_dict, scores_dict)
    """
    # Generate variations
    name_vars = generate_name_variations(original_name, expected_count)
    dob_vars = generate_dob_variations(seed_dob, expected_count)
    addr_vars = generate_address_variations(seed_address, expected_count)

    # Predict sub-scores
    # Name: average of per-variation scores (can also use top-k)
    if name_vars:
        name_scores = [predict_name_score(original_name, v, name_weight=0.3) for v in name_vars]
        name_score = sum(name_scores) / len(name_scores)
    else:
        name_score = 0.0

    dob_score = predict_dob_category_score(seed_dob, dob_vars, dob_weight=0.1)
    address_score = predict_address_score(seed_address, addr_vars, address_weight=0.6)

    total_score = min(1.0, name_score + dob_score + address_score)

    variations = {
        "name": name_vars,
        "dob": dob_vars,
        "address": addr_vars,
    }
    scores = {
        "name": float(name_score),
        "dob": float(dob_score),
        "address": float(address_score),
        "total": float(total_score),
    }
    return variations, scores


if __name__ == "__main__":
    samples = [
        # Latin / Vietnamese (Latin script)
        {
            "label": "Vietnamese (Latin)",
            "name": "Nguyen Van A",
            "dob": "1992-06-20",
            "address": "45 Nguyen Hue, Ho Chi Minh City, Vietnam",
        },
        # Arabic
        {
            "label": "Arabic",
            "name": "محمد أحمد",
            "dob": "1990-01-15",
            "address": "10 Al Wasl Road, Dubai, UAE",
        },
        # CJK - Chinese
        {
            "label": "Chinese (CJK)",
            "name": "王小明",
            "dob": "1988-12-01",
            "address": "123 Maple Ave, New York, USA",
        },
        # Latin - English
        {
            "label": "English (Latin)",
            "name": "John Smith",
            "dob": "1990-01-15",
            "address": "123 Maple Ave, New York, USA",
        },
    ]

    for s in samples:
        print("=" * 72)
        print(f"Sample: {s['label']}")
        print(f"Name: {s['name']} | DOB: {s['dob']} | Address: {s['address']}")

        vars_dict, scores = generate_and_predict(s["name"], s["dob"], s["address"], expected_count=10)

        print("\n=== Variations ===")
        for k, v in vars_dict.items():
            preview = v[:5]
            print(f"{k} (showing {len(preview)}/{len(v)}): {preview}")

        print("\n=== Predicted Scores ===")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
        print()


