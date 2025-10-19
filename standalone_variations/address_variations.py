#!/usr/bin/env python3
"""
Standalone Address Variations Generator (no project dependencies)

Rules:
- Keep the same city/country region as seed
- Produce realistic-looking addresses (street + number + city + country)
- Unique variations
- Static data only (no external API)
"""

from typing import List, Tuple
import random
import re


STATIC_DATA = {
    "USA": {
        "New York": {
            "streets": ["Broadway", "5th Ave", "Madison Ave", "Park Ave", "Lexington Ave", "Maple Ave", "Elm St"],
            "areas": ["Manhattan", "Brooklyn", "Queens", "Bronx"],
        },
        "Los Angeles": {
            "streets": ["Sunset Blvd", "Hollywood Blvd", "Wilshire Blvd", "Santa Monica Blvd", "Main St"],
            "areas": ["Downtown", "Hollywood", "Santa Monica", "Westwood"],
        },
    },
    "UAE": {
        "Dubai": {
            "streets": [
                "Sheikh Zayed Road",
                "Al Wasl Road",
                "Jumeirah Beach Road",
                "Al Khail Road",
                "Emirates Road",
                "Al Rigga Street",
                "Shaikh Mohammed Bin Zayed Rd"
            ],
            "areas": [
                "Downtown",
                "Dubai Marina",
                "Jumeirah",
                "Deira",
                "Bur Dubai",
                "Business Bay"
            ],
        },
        "Abu Dhabi": {
            "streets": [
                "Corniche Road",
                "Al Salam Street",
                "Airport Road",
                "Al Falah Street",
                "Muroor Road"
            ],
            "areas": [
                "Corniche",
                "Al Reem Island",
                "Yas Island",
                "Khalifa City"
            ],
        },
    },
    "Vietnam": {
        "Ho Chi Minh City": {
            "streets": ["Lê Lợi", "Nguyễn Huệ", "Pasteur", "Nam Kỳ Khởi Nghĩa", "Điện Biên Phủ"],
            "areas": ["Quận 1", "Quận 3", "Phú Nhuận", "Bình Thạnh"],
        },
        "Hanoi": {
            "streets": ["Tràng Tiền", "Phố Huế", "Kim Mã", "Láng Hạ", "Giảng Võ"],
            "areas": ["Hoàn Kiếm", "Ba Đình", "Đống Đa", "Cầu Giấy"],
        },
    },
}


def _extract_city_country(address: str) -> Tuple[str, str]:
    # Heuristic: last two comma-separated tokens
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        city, country = parts[-2], parts[-1]
        # Normalize common aliases
        if country.lower() in {"uae", "u.a.e", "united arab emirates"}:
            country = "UAE"
        return city, country
    # Fallback: try to detect known cities/countries
    for country, cities in STATIC_DATA.items():
        for city in cities.keys():
            if city.lower() in address.lower() and country.lower() in address.lower():
                return city, country
    return "", ""


def _random_house_number() -> str:
    return str(random.randint(10, 9999))


def _looks_like_address(text: str) -> bool:
    # Must contain a number and a street-like word
    has_number = any(ch.isdigit() for ch in text)
    has_word = bool(re.search(r"[A-Za-zÀ-ỹ]+", text))
    return has_number and has_word and len(text) >= 8


def generate_address_variations(
    seed_address: str,
    expected_count: int = 10,
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> List[str]:
    city, country = _extract_city_country(seed_address)
    variations: List[str] = []

    if not city or not country:
        return variations

    country_data = STATIC_DATA.get(country)
    if not country_data:
        return variations

    city_data = country_data.get(city)
    if not city_data:
        # Try first available city in that country if seed city not in static
        city, city_data = next(iter(country_data.items()))

    streets = city_data.get("streets", [])
    areas = city_data.get("areas", [])

    # Build variations
    rng_seed = (hash(seed_address) & 0xFFFFFFFF) ^ (miner_salt * 7919) ^ (batch_salt * 271)
    rng = random.Random(rng_seed)
    while len(variations) < expected_count and streets:
        street = rng.choice(streets)
        area = rng.choice(areas) if areas else ""
        number = _random_house_number()
        parts = [number, street]
        if area:
            parts.append(area)
        parts.extend([city, country])
        candidate = ", ".join(parts)
        if _looks_like_address(candidate) and candidate not in variations:
            variations.append(candidate)

    # Dedup and shuffle to avoid identical signatures
    unique = list(dict.fromkeys(variations))
    rng.shuffle(unique)
    return unique[:expected_count]


def validate_address_variation(candidate: str, seed_address: str) -> bool:
    """Heuristic validator similar to project logic: looks-like + same region."""
    c_city, c_country = _extract_city_country(candidate)
    s_city, s_country = _extract_city_country(seed_address)
    if not (c_city and c_country and s_city and s_country):
        return False
    if c_city.lower() != s_city.lower() or c_country.lower() != s_country.lower():
        return False
    return _looks_like_address(candidate)


def predict_address_score(seed_address: str, variations: List[str], address_weight: float = 0.6) -> float:
    """Score portion proportional to valid variations fraction, scaled by weight."""
    if not variations:
        return 0.0
    valid = sum(1 for v in variations if validate_address_variation(v, seed_address))
    frac = valid / len(variations)
    return address_weight * frac


if __name__ == "__main__":
    print(generate_address_variations("123 Maple Ave, New York, USA", 5))

