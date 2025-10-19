#!/usr/bin/env python3
"""
Example runner for standalone variations modules.

Usage:
  PYTHONPATH=. uv run standalone_variations/run_example.py \
    --name "Nguyen Van A" --dob 1990-01-15 \
    --address "123 Maple Ave, New York, USA" \
    --count 8 --miner-salt 1234 --batch-salt 56
"""

import argparse
from pprint import pprint

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
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone variations example")
    parser.add_argument("--name", required=True, help="Original name")
    parser.add_argument("--dob", required=True, help="Seed DOB YYYY-MM-DD")
    parser.add_argument("--address", required=True, help="Seed address (street, city, country)")
    parser.add_argument("--count", type=int, default=10, help="Expected number of variations")
    parser.add_argument("--country", type=str, default=None, help="Country name to force script (from sanctioned_countries.json)")
    parser.add_argument("--miner-salt", type=int, default=0, help="Miner salt to diversify output")
    parser.add_argument("--batch-salt", type=int, default=0, help="Batch salt to diversify output")
    args = parser.parse_args()

    # Generate
    name_vars = generate_name_variations(
        args.name, args.count, miner_salt=args.miner_salt, batch_salt=args.batch_salt, country=args.country
    )
    dob_vars = generate_dob_variations(
        args.dob, args.count, miner_salt=args.miner_salt, batch_salt=args.batch_salt
    )
    addr_vars = generate_address_variations(
        args.address, args.count, miner_salt=args.miner_salt, batch_salt=args.batch_salt
    )

    print("=== Variations ===")
    if args.country:
        print(f"(country mapping applied: {args.country})")
    print("name:")
    pprint(name_vars)
    print("dob:")
    pprint(dob_vars)
    print("address:")
    pprint(addr_vars)

    # Predict scores (validator-like weights)
    name_scores = [predict_name_score(args.name, v, name_weight=0.3) for v in name_vars] if name_vars else [0.0]
    name_score = sum(name_scores) / len(name_scores) if name_scores else 0.0
    dob_score = predict_dob_category_score(args.dob, dob_vars, dob_weight=0.1)
    addr_score = predict_address_score(args.address, addr_vars, address_weight=0.6)
    total = min(1.0, name_score + dob_score + addr_score)

    print("\n=== Predicted Scores ===")
    print(f"name:    {name_score:.4f}")
    print(f"dob:     {dob_score:.4f}")
    print(f"address: {addr_score:.4f}")
    print(f"total:   {total:.4f}")


if __name__ == "__main__":
    main()


