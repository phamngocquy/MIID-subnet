#!/usr/bin/env python3
"""
Utilities to build validator-formatted responses:
responses: List[Dict[str, List[List[str]]]]
Each dict represents one miner: { seed_name: [[name_var, dob_var, address_var], ...], ... }
"""

from typing import List, Dict, Optional

from standalone_variations.name_variations import generate_name_variations
from standalone_variations.dob_variations import generate_dob_variations
from standalone_variations.address_variations import generate_address_variations


def _pad_to(xs: List[str], n: int) -> List[str]:
    if not xs:
        return []
    out = xs[:]
    i = 0
    while len(out) < n:
        out.append(xs[i % len(xs)])
        i += 1
    return out[:n]


def build_responses_for_single_miner(
    seed_names: List[str],
    seed_dobs: List[str],
    seed_addresses: List[str],
    expected_count: int = 10,
    country: Optional[str] = None,
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> List[Dict[str, List[List[str]]]]:
    """Build a single-miner responses structure in validator-required format."""
    miner_dict: Dict[str, List[List[str]]] = {}

    for idx, seed_name in enumerate(seed_names):
        seed_dob = seed_dobs[idx] if idx < len(seed_dobs) else ""
        seed_addr = seed_addresses[idx] if idx < len(seed_addresses) else ""

        name_vars = generate_name_variations(
            seed_name,
            expected_count,
            miner_salt=miner_salt,
            batch_salt=batch_salt,
            country=country,
        )
        dob_vars = generate_dob_variations(
            seed_dob,
            expected_count,
            miner_salt=miner_salt,
            batch_salt=batch_salt,
        )
        addr_vars = generate_address_variations(
            seed_addr,
            expected_count,
            miner_salt=miner_salt,
            batch_salt=batch_salt,
        )

        # Ensure at least 80% coverage of expected_count
        base_len = min(len(name_vars), len(dob_vars), len(addr_vars), expected_count)
        target_len = max(base_len, int(max(1, expected_count * 0.8)))

        name_vars = _pad_to(name_vars, target_len)
        dob_vars = _pad_to(dob_vars, target_len)
        addr_vars = _pad_to(addr_vars, target_len)

        triplets = [[n, d, a] for n, d, a in zip(name_vars, dob_vars, addr_vars) if n and d and a]
        miner_dict[seed_name] = triplets

    return [miner_dict]


