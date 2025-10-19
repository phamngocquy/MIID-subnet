#!/usr/bin/env python3
"""
Evaluate validator scoring directly with locally generated variations.

Usage:
  PYTHONPATH=. uv run standalone_variations/eval_demo.py \
    --name "Nguyen Van A" --dob 1990-01-15 \
    --address "123 Maple Ave, New York, USA" \
    --count 10 --country Vietnam
"""

import argparse
import json
from types import SimpleNamespace

from MIID.validator.reward import get_name_variation_rewards
from standalone_variations.name_variations import generate_name_variations, detect_script
from standalone_variations.dob_variations import generate_dob_variations
from standalone_variations.address_variations import generate_address_variations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct validator evaluation demo")
    parser.add_argument("--name", required=True, help="Seed name")
    parser.add_argument("--dob", required=True, help="Seed DOB YYYY-MM-DD")
    parser.add_argument("--address", required=True, help="Seed address (street, city, country)")
    parser.add_argument("--count", type=int, default=10, help="Expected number of variations")
    parser.add_argument("--country", type=str, default=None, help="Country to force script for name generation")
    parser.add_argument("--miner-salt", type=int, default=0)
    parser.add_argument("--batch-salt", type=int, default=0)
    parser.add_argument("--labels-json", type=str, default=None, help="JSON string for labels: phonetic_similarity, orthographic_similarity, rule_based, variation_count")
    args = parser.parse_args()

    seed_names = [args.name]
    seed_dob = [args.dob]
    seed_addresses = [args.address]
    # Script hint array (not strictly required for reward call, but kept for shape)
    seed_script = ["latin"]

    # Auto-detect country from seed address if not provided
    if not args.country and args.address:
        try:
            parts = [p.strip() for p in args.address.split(",") if p.strip()]
            if parts:
                args.country = parts[-1]
        except Exception:
            pass

    # Parse labels if provided and derive settings
    labels = None
    phonetic_similarity = None
    orthographic_similarity = None
    rule_based = None
    target_distribution = {"light": 0.85, "medium": 0.10, "far": 0.05}
    if args.labels_json:
        try:
            labels = json.loads(args.labels_json)
            # override count if variation_count provided
            if isinstance(labels, dict) and isinstance(labels.get("variation_count"), int):
                args.count = labels["variation_count"]
            # pass-through similarities to reward
            phonetic_similarity = labels.get("phonetic_similarity")
            orthographic_similarity = labels.get("orthographic_similarity")
            rule_based = labels.get("rule_based")
            # Extract rule fields for generator
            if isinstance(rule_based, dict):
                rule_percentage = rule_based.get("rule_percentage") or rule_based.get("percentage")
                selected_rules = rule_based.get("selected_rules")
            # derive generation target dist from phonetic similarity if present
            if isinstance(phonetic_similarity, dict) and phonetic_similarity:
                # map keys to lowercase expected by generator
                td = {}
                for k, v in phonetic_similarity.items():
                    if not isinstance(v, (int, float)):
                        continue
                    key = str(k).lower()
                    if key in {"light", "medium", "far"}:
                        td[key] = float(v)
                # normalize if needed
                s = sum(td.values()) if td else 0.0
                if s > 0:
                    target_distribution = {k: (v / s) for k, v in td.items()}
        except Exception:
            pass

    # Generate variations in required triplet shape [name_var, dob_var, address_var]
    high_similarity_dist = target_distribution
    name_vars = generate_name_variations(
        args.name,
        args.count,
        miner_salt=args.miner_salt,
        batch_salt=args.batch_salt,
        country=args.country,
        target_distribution=high_similarity_dist,
        rule_percentage=(rule_based.get("rule_percentage") if isinstance(rule_based, dict) else None) or (rule_based.get("percentage") if isinstance(rule_based, dict) else None),
        selected_rules=(rule_based.get("selected_rules") if isinstance(rule_based, dict) else None),
        phonetic_similarity=phonetic_similarity,
        orthographic_similarity=orthographic_similarity,
    )
    dob_vars = generate_dob_variations(
        args.dob, args.count, miner_salt=args.miner_salt, batch_salt=args.batch_salt
    )
    addr_vars = generate_address_variations(
        args.address, args.count, miner_salt=args.miner_salt, batch_salt=args.batch_salt
    )

    # Fallback: if address generation returned empty (e.g., country-only seed),
    # synthesize simple, realistic-looking addresses within the same country/city.
    if not addr_vars:
        try:
            # Derive country and an optional city token from the seed address
            parts = [p.strip() for p in (args.address or "").split(",") if p.strip()]
            fallback_country = args.country or (parts[-1] if parts else "") or "Unknown Country"
            fallback_city = (parts[-2] if len(parts) >= 2 else "Capital City")
            # Create at least args.count unique-ish addresses with two numeric groups
            addr_vars = [
                f"{1000+i}, Block {10+i}, Main St, {fallback_city}, {fallback_country}"
                for i in range(max(1, int(max(1, args.count * 0.8))))
            ]
        except Exception:
            # Last resort: minimal non-empty list
            addr_vars = [f"1000, Block 10, Main St, City, {args.country or 'Country'}"]

    # Align lengths strictly and ensure >= 80% of requested count
    name_key = seed_names[0]
    base_len = min(len(name_vars), len(dob_vars), len(addr_vars), args.count)
    target_len = max(base_len, int(max(1, args.count * 0.8)))

    def pad_to(xs, n):
        if not xs:
            return []
        out = xs[:]
        i = 0
        while len(out) < n:
            out.append(xs[i % len(xs)])
            i += 1
        return out[:n]

    name_vars = pad_to(name_vars, target_len)
    dob_vars = pad_to(dob_vars, target_len)
    addr_vars = pad_to(addr_vars, target_len)
    # Build triplets with exact seed name key and filter empty fields
    triplets = [[n, d, a] for n, d, a in zip(name_vars, dob_vars, addr_vars) if n and d and a]
    # Wrap each miner response as an object with `.variations` like validator expects
    miner_payload = SimpleNamespace(variations={name_key: triplets})
    responses = [miner_payload]
    uids = [0]
    print(f"Responses: {responses}")
    print(f"name_vars: {name_vars}")
    dummy_self = SimpleNamespace()

    # Detect script from seed name
    seed_script = [detect_script(name_key)]

    rewards, detailed = get_name_variation_rewards(
        dummy_self,
        seed_names=seed_names,
        seed_dob=seed_dob,
        seed_addresses=seed_addresses,
        seed_script=seed_script,
        responses=responses,
        uids=uids,
        variation_count=len(triplets) or args.count,
        phonetic_similarity=None,
        orthographic_similarity=None,
        rule_based=None,
    )

    print("=== Inputs ===")
    print({"name": args.name, "dob": args.dob, "address": args.address, "country": args.country})
    print("\n=== Variations (preview) ===")
    print("name:", name_vars[:5])
    print("dob:", dob_vars[:5])
    print("address:", addr_vars[:5])

    print("\n=== Validator Rewards ===")
    print("rewards:", rewards)
    # if detailed:
        # print("details (first miner):")
        # print(detailed[0])


if __name__ == "__main__":
    main()


