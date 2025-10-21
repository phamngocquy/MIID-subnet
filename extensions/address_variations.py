# coding=utf-8
import random
import unicodedata

import pandas as pd
from numpy import var

GEO_DATA_PATH = (
    "/home/qpham/IProjects/bittensor/miid/MIID-subnet/geo_locations_norm.csv"
)


def strip_accents(s):
    return "".join(
        c.lower()
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).replace(" ", "")


def normalize_address(addr_str):
    """Normalize address string by removing extra spaces and standardizing format"""
    if not addr_str:
        return ""
    # Remove extra spaces, convert to lowercase, and standardize common separators
    normalized = " ".join(addr_str.split()).lower()
    # Replace common separators with spaces
    normalized = normalized.replace(",", " ").replace(";", " ").replace("-", " ")
    # Remove multiple spaces
    normalized = " ".join(normalized.split())
    return normalized


def generate_address_variations(country_name: str, expected_count: int):
    if not expected_count:
        return []

    variations = []
    normalize_country_name = strip_accents(country_name)
    dframe = pd.read_csv(GEO_DATA_PATH)
    cnt_dframe = dframe[dframe["country_no_accents"] == normalize_country_name]
    for row in cnt_dframe.to_dict("records"):
        for pair in eval(row["tested_pairs"]):
            variations.append(row["address"].format(n1=pair[0], n2=pair[1]))
    if not variations:
        return []
    for _ in range((expected_count // len(variations)) + 1):
        variations.extend(variations)
    return variations[:expected_count]


if __name__ == "__main__":
    print(generate_address_variations("South Korea", 20))
