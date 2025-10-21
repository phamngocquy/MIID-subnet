"""
Enhanced Name Variations Generator with Weighted Similarity Support

Features:
- Variable orthographic and phonetic weights from noted.md configuration
- Support for all countries from JSON files (sanctioned_countries.json, Sanctioned_list.json, Sanctioned_Transliteration.json)
- Additional Latin language countries support
- Script detection (latin, arabic, cjk, cyrillic, other_scripts)
- Weighted similarity distribution selection
- Rule-based variation generation
- Country-specific name generation patterns
"""

import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import jellyfish
import Levenshtein


class ScriptType(Enum):
    LATIN = "latin"
    ARABIC = "arabic"
    CJK = "cjk"
    CYRILLIC = "cyrillic"
    OTHER_SCRIPTS = "other_scripts"


@dataclass
class CountryInfo:
    """Information about a country and its script"""

    name: str
    script: ScriptType
    faker_locale: Optional[str] = None
    is_sanctioned: bool = False


class CountryManager:
    """Manages country information and script detection"""

    def __init__(self):
        self.countries: Dict[str, CountryInfo] = {}
        self._load_countries()

    def _load_countries(self):
        """Load countries from JSON files and add additional Latin countries"""
        # Load from sanctioned_countries.json
        self._load_sanctioned_countries()

        # Load from Sanctioned_list.json for additional countries
        self._load_sanctioned_list()

        # Add additional Latin language countries
        self._add_latin_countries()

    def _load_sanctioned_countries(self):
        """Load countries from sanctioned_countries.json"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(
                current_dir, "..", "MIID", "validator", "sanctioned_countries.json"
            )
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for script_key, countries in data.items():
                    script_type = ScriptType(script_key)
                    for country_data in countries:
                        country_name = country_data.get("country")
                        faker_locale = country_data.get("faker_locale")
                        if country_name:
                            self.countries[country_name.lower()] = CountryInfo(
                                name=country_name,
                                script=script_type,
                                faker_locale=faker_locale,
                                is_sanctioned=True,
                            )
        except Exception as e:
            print(f"Error loading sanctioned countries: {e}")

    def _load_sanctioned_list(self):
        """Load countries from Sanctioned_list.json"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(
                current_dir, "..", "MIID", "validator", "Sanctioned_list.json"
            )
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for entry in data:
                    country = entry.get("Country_Residence", "")
                    if country and country.lower() not in self.countries:
                        # Determine script based on country name patterns
                        script = self._detect_script_from_country(country)
                        self.countries[country.lower()] = CountryInfo(
                            name=country, script=script, is_sanctioned=True
                        )
        except Exception as e:
            print(f"Error loading sanctioned list: {e}")

    def _detect_script_from_country(self, country: str) -> ScriptType:
        """Detect script type from country name"""
        country_lower = country.lower()

        # Arabic countries
        arabic_countries = [
            "syria",
            "iraq",
            "iran",
            "afghanistan",
            "sudan",
            "lebanon",
            "libya",
            "somalia",
            "yemen",
            "algeria",
        ]
        if any(arabic in country_lower for arabic in arabic_countries):
            return ScriptType.ARABIC

        # CJK countries
        cjk_countries = ["china", "taiwan", "japan", "north korea", "south korea"]
        if any(cjk in country_lower for cjk in cjk_countries):
            return ScriptType.CJK

        # Cyrillic countries
        cyrillic_countries = [
            "russia",
            "ukraine",
            "belarus",
            "bulgaria",
            "armenia",
            "crimea",
            "donetsk",
            "luhansk",
        ]
        if any(cyrillic in country_lower for cyrillic in cyrillic_countries):
            return ScriptType.CYRILLIC

        # Other scripts
        other_scripts_countries = [
            "myanmar",
            "laos",
            "nepal",
            "vietnam",
            "india",
            "thailand",
            "israel",
            "greece",
            "georgia",
        ]
        if any(other in country_lower for other in other_scripts_countries):
            return ScriptType.OTHER_SCRIPTS

        # Default to Latin
        return ScriptType.LATIN

    def _add_latin_countries(self):
        """Add additional Latin language countries"""
        latin_countries = [
            # European Latin countries
            "Spain",
            "France",
            "Italy",
            "Portugal",
            "Romania",
            "Moldova",
            # Latin American countries
            "Mexico",
            "Brazil",
            "Argentina",
            "Colombia",
            "Peru",
            "Chile",
            "Ecuador",
            "Guatemala",
            "Cuba",
            "Honduras",
            "Paraguay",
            "Uruguay",
            "Costa Rica",
            "Panama",
            "El Salvador",
            "Nicaragua",
            "Dominican Republic",
            "Haiti",
            "Jamaica",
            "Trinidad and Tobago",
            "Barbados",
            "Guyana",
            "Suriname",
            "Belize",
            "Grenada",
            "Saint Lucia",
            "Saint Vincent and the Grenadines",
            "Antigua and Barbuda",
            "Dominica",
            "Saint Kitts and Nevis",
            # African Latin countries
            "Angola",
            "Mozambique",
            "Cape Verde",
            "Guinea-Bissau",
            "São Tomé and Príncipe",
            "Equatorial Guinea",
            "Chad",
            "Central African Republic",
            "Cameroon",
            "Ivory Coast",
            "Burkina Faso",
            "Mali",
            "Niger",
            "Senegal",
            "Guinea",
            "Sierra Leone",
            "Liberia",
            "Ghana",
            "Togo",
            "Benin",
            "Nigeria",
            "Gabon",
            "Republic of the Congo",
            "Democratic Republic of the Congo",
            "Rwanda",
            "Burundi",
            "Madagascar",
            "Mauritius",
            "Seychelles",
            "Comoros",
            "Djibouti",
            "Eritrea",
            "Ethiopia",
            "Somalia",
            "Kenya",
            "Tanzania",
            "Uganda",
            "South Sudan",
            "Sudan",
            "Egypt",
            "Libya",
            "Tunisia",
            "Algeria",
            "Morocco",
            "Western Sahara",
            "Mauritania",
            "Mali",
            "Burkina Faso",
            "Niger",
            "Chad",
            "Central African Republic",
            "Cameroon",
            "Nigeria",
            "Benin",
            "Togo",
            "Ghana",
            "Burkina Faso",
            "Ivory Coast",
            "Liberia",
            "Sierra Leone",
            "Guinea",
            "Guinea-Bissau",
            "Senegal",
            "Gambia",
            "Mauritania",
            "Mali",
            "Burkina Faso",
            "Niger",
            "Chad",
            "Sudan",
            "South Sudan",
            "Ethiopia",
            "Eritrea",
            "Djibouti",
            "Somalia",
            "Kenya",
            "Uganda",
            "Tanzania",
            "Rwanda",
            "Burundi",
            "Democratic Republic of the Congo",
            "Republic of the Congo",
            "Central African Republic",
            "Cameroon",
            "Equatorial Guinea",
            "Gabon",
            "São Tomé and Príncipe",
            "Angola",
            "Zambia",
            "Malawi",
            "Mozambique",
            "Madagascar",
            "Mauritius",
            "Seychelles",
            "Comoros",
            "Mayotte",
            "Réunion",
            "Saint Helena",
            "Ascension Island",
            "Tristan da Cunha",
            "Bouvet Island",
            "South Georgia",
            "South Sandwich Islands",
            "Falkland Islands",
            "South Shetland Islands",
            "South Orkney Islands",
            "South Georgia and the South Sandwich Islands",
            # Other Latin countries
            "Philippines",
            "East Timor",
            "Vatican City",
            "San Marino",
            "Andorra",
            "Liechtenstein",
            "Monaco",
            "Luxembourg",
            "Belgium",
            "Netherlands",
            "Switzerland",
            "Austria",
            "Germany",
            "Denmark",
            "Sweden",
            "Norway",
            "Finland",
            "Iceland",
            "Ireland",
            "United Kingdom",
            "Malta",
            "Cyprus",
        ]

        for country in latin_countries:
            if country.lower() not in self.countries:
                self.countries[country.lower()] = CountryInfo(
                    name=country, script=ScriptType.LATIN, is_sanctioned=False
                )

    def get_country_info(self, country: Optional[str]) -> Optional[CountryInfo]:
        """Get country information by name"""
        if not country:
            return None
        return self.countries.get(country.lower())

    def get_script_for_country(self, country: Optional[str]) -> ScriptType:
        """Get script type for a country"""
        country_info = self.get_country_info(country)
        if country_info:
            return country_info.script
        return ScriptType.LATIN  # Default fallback


class WeightedSimilaritySelector:
    """Handles weighted selection of similarity configurations"""

    def __init__(self):
        self.phonetic_configs = self._load_phonetic_configs()
        self.orthographic_configs = self._load_orthographic_configs()

    def get_balanced_configs(self):
        """Get balanced phonetic and orthographic configurations"""
        return (
            {"Light": 0.3, "Medium": 0.4, "Far": 0.3},  # Phonetic config
            {"Light": 0.3, "Medium": 0.4, "Far": 0.3},  # Orthographic config
        )

    def _load_phonetic_configs(self) -> List[Tuple[Dict[str, float], float]]:
        """Load phonetic similarity configurations from noted.md"""
        return [
            # Balanced distribution - high weight for balanced testing
            ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
            # Focus on Medium similarity - most common real-world scenario
            ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
            # Focus on Far similarity - important for edge cases
            ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
            # Light-Medium mix - moderate weight
            ({"Light": 0.5, "Medium": 0.5}, 0.12),
            # Medium-Far mix - moderate weight
            ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
            # Only Medium similarity - common case
            ({"Medium": 1.0}, 0.08),
            # High Light but not 100% - reduced frequency
            ({"Light": 0.7, "Medium": 0.3}, 0.05),
            # Only Far similarity - edge case
            ({"Far": 1.0}, 0.03),
            # Only Light similarity - reduced frequency
            ({"Light": 1.0}, 0.02),
        ]

    def _load_orthographic_configs(self) -> List[Tuple[Dict[str, float], float]]:
        """Load orthographic similarity configurations from noted.md"""
        return [
            # Balanced distribution - high weight for balanced testing
            ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
            # Focus on Medium similarity - most common real-world scenario
            ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
            # Focus on Far similarity - important for edge cases
            ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
            # Light-Medium mix - moderate weight
            ({"Light": 0.5, "Medium": 0.5}, 0.12),
            # Medium-Far mix - moderate weight
            ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
            # Only Medium similarity - common case
            ({"Medium": 1.0}, 0.08),
            # High Light but not 100% - reduced frequency
            ({"Light": 0.7, "Medium": 0.3}, 0.05),
            # Only Far similarity - edge case
            ({"Far": 1.0}, 0.03),
            # Only Light similarity - reduced frequency
            ({"Light": 1.0}, 0.02),
        ]

    def select_phonetic_config(self) -> Dict[str, float]:
        """Select phonetic configuration using weighted random selection"""
        configs, weights = zip(*self.phonetic_configs)
        return random.choices(configs, weights=weights, k=1)[0]

    def select_orthographic_config(self) -> Dict[str, float]:
        """Select orthographic configuration using weighted random selection"""
        configs, weights = zip(*self.orthographic_configs)
        return random.choices(configs, weights=weights, k=1)[0]


def detect_script(text: str) -> ScriptType:
    """Detect script type from text content"""
    if all(ord(ch) < 128 for ch in text):
        return ScriptType.LATIN

    # Specific script ranges
    ranges = [
        (
            ScriptType.ARABIC,
            [
                (0x0600, 0x06FF),
                (0x0750, 0x077F),
                (0x08A0, 0x08FF),
                (0xFB50, 0xFDFF),
                (0xFE70, 0xFEFF),
            ],
        ),
        (
            ScriptType.CYRILLIC,
            [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
        ),
        (ScriptType.CJK, [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
    ]

    for script, blocks in ranges:
        for start, end in blocks:
            if any(start <= ord(ch) <= end for ch in text):
                return script

    return ScriptType.LATIN


def _generate_latin_variations(
    name: str, count: int, country_info: Optional[CountryInfo] = None
) -> List[str]:
    """Generate Latin script variations with country-specific patterns"""
    variations: List[str] = []

    # Country-specific patterns
    if country_info and country_info.faker_locale:
        locale = country_info.faker_locale
        if "es" in locale:  # Spanish-speaking countries
            variations.extend(_generate_spanish_variations(name, count))
        elif "fr" in locale:  # French-speaking countries
            variations.extend(_generate_french_variations(name, count))
        elif "pt" in locale:  # Portuguese-speaking countries
            variations.extend(_generate_portuguese_variations(name, count))
        elif "de" in locale:  # German-speaking countries
            variations.extend(_generate_german_variations(name, count))
        elif "it" in locale:  # Italian-speaking countries
            variations.extend(_generate_italian_variations(name, count))

    # General Latin variations
    variations.extend(_generate_general_latin_variations(name, count))

    return variations[:count]


def _generate_spanish_variations(name: str, count: int) -> List[str]:
    """Generate Spanish-specific variations"""
    variations = []

    # Spanish accent variations
    accent_swaps = {
        "a": ["á", "à", "â", "ä", "ã"],
        "e": ["é", "è", "ê", "ë"],
        "i": ["í", "ì", "î", "ï"],
        "o": ["ó", "ò", "ô", "ö", "õ"],
        "u": ["ú", "ù", "û", "ü"],
        "n": ["ñ"],
        "c": ["ç"],
    }

    for i, char in enumerate(name):
        if char.lower() in accent_swaps:
            for accent in accent_swaps[char.lower()]:
                variation = name[:i] + accent + name[i + 1 :]
                if variation != name:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    return variations


def _generate_french_variations(name: str, count: int) -> List[str]:
    """Generate French-specific variations"""
    variations = []

    # French accent variations
    accent_swaps = {
        "a": ["à", "â", "ä"],
        "e": ["é", "è", "ê", "ë"],
        "i": ["î", "ï"],
        "o": ["ô", "ö"],
        "u": ["ù", "û", "ü"],
        "c": ["ç"],
    }

    for i, char in enumerate(name):
        if char.lower() in accent_swaps:
            for accent in accent_swaps[char.lower()]:
                variation = name[:i] + accent + name[i + 1 :]
                if variation != name:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    return variations


def _generate_portuguese_variations(name: str, count: int) -> List[str]:
    """Generate Portuguese-specific variations"""
    variations = []

    # Portuguese accent variations
    accent_swaps = {
        "a": ["á", "à", "â", "ã"],
        "e": ["é", "ê"],
        "i": ["í"],
        "o": ["ó", "ô", "õ"],
        "u": ["ú"],
        "c": ["ç"],
    }

    for i, char in enumerate(name):
        if char.lower() in accent_swaps:
            for accent in accent_swaps[char.lower()]:
                variation = name[:i] + accent + name[i + 1 :]
                if variation != name:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    return variations


def _generate_german_variations(name: str, count: int) -> List[str]:
    """Generate German-specific variations"""
    variations = []

    # German umlaut variations
    umlaut_swaps = {"a": ["ä"], "o": ["ö"], "u": ["ü"], "s": ["ß"]}

    for i, char in enumerate(name):
        if char.lower() in umlaut_swaps:
            for umlaut in umlaut_swaps[char.lower()]:
                variation = name[:i] + umlaut + name[i + 1 :]
                if variation != name:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    return variations


def _generate_italian_variations(name: str, count: int) -> List[str]:
    """Generate Italian-specific variations"""
    variations = []

    # Italian accent variations
    accent_swaps = {
        "a": ["à"],
        "e": ["é", "è"],
        "i": ["í", "ì"],
        "o": ["ó", "ò"],
        "u": ["ú", "ù"],
    }

    for i, char in enumerate(name):
        if char.lower() in accent_swaps:
            for accent in accent_swaps[char.lower()]:
                variation = name[:i] + accent + name[i + 1 :]
                if variation != name:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    return variations


def _generate_general_latin_variations(name: str, count: int) -> List[str]:
    """Generate general Latin variations optimized for reward system scoring"""
    variations = []
    seen = set()  # Track seen variations to avoid duplicates

    # PRIORITY 1: Light variations (high similarity) - Target ~60% when aiming for high scores
    # Vowel swaps (very high similarity)
    vowel_swaps = {"a": ["e"], "e": ["a"], "i": ["y"], "o": ["u"], "u": ["o"]}
    light_count = 0
    max_light = max(1, int(count * 0.6))  # Boost light share

    for i, ch in enumerate(name):
        if light_count >= max_light:
            break
        low = ch.lower()
        if low in vowel_swaps:
            for repl in vowel_swaps[low]:
                new_ch = repl.upper() if ch.isupper() else repl
                variation = name[:i] + new_ch + name[i + 1 :]
                if variation != name and variation not in seen:
                    variations.append(variation)
                    seen.add(variation)
                    light_count += 1
                    if len(variations) >= count:
                        return variations

    # Single character substitutions (high similarity) - only one minimal change
    for i, ch in enumerate(name):
        if light_count >= max_light:
            break
        if ch.isalpha():
            # Try common high-similarity substitutions
            subs = {
                "c": "k",
                "k": "c",
                "s": "z",
                "z": "s",
                "ph": "f",
                "f": "ph",
                "v": "b",
                "b": "v",
            }
            for old, new in subs.items():
                if name[i : i + len(old)].lower() == old:
                    variation = name[:i] + new + name[i + len(old) :]
                    if variation != name and variation not in seen:
                        variations.append(variation)
                        seen.add(variation)
                        light_count += 1
                        if len(variations) >= count:
                            return variations

    # PRIORITY 2: Medium variations - Target ~30%
    tokens = name.split()
    medium_count = 0
    max_medium = max(2, int(count * 0.3))

    # Transposition within tokens (medium similarity)
    for ti, tok in enumerate(tokens):
        if medium_count >= max_medium:
            break
        if len(tok) >= 4:
            # Swap adjacent characters
            for i in range(len(tok) - 1):
                variation_tokens = tokens[:]
                new_tok = tok[:i] + tok[i + 1] + tok[i] + tok[i + 2 :]
                variation_tokens[ti] = new_tok
                variation = " ".join(variation_tokens)
                if variation != name and variation not in seen:
                    variations.append(variation)
                    seen.add(variation)
                    medium_count += 1
                    if len(variations) >= count:
                        return variations

    # Abbreviations (medium similarity)
    if len(tokens) >= 2 and medium_count < max_medium:
        first = tokens[0]
        middle = tokens[1:-1]
        last = tokens[-1]
        prefix_len = 4 if len(first) >= 6 else 3
        first_initial = (first[:prefix_len] + ".") if first else ""
        candidates = [
            " ".join([first_initial] + middle + [last]),
        ]
        for c in candidates:
            c = c.strip()
            if c and c != name and c not in seen:
                variations.append(c)
                seen.add(c)
                medium_count += 1
                if len(variations) >= count:
                    return variations

    # Additional medium variations - character changes
    for i, ch in enumerate(name):
        if medium_count >= max_medium:
            break
        if ch.isalpha():
            # Try different character substitutions for medium similarity
            if ch.lower() == "c":
                variation = name[:i] + "k" + name[i + 1 :]
                if variation != name and variation not in seen:
                    variations.append(variation)
                    seen.add(variation)
                    medium_count += 1
                    if len(variations) >= count:
                        return variations
            elif ch.lower() == "k":
                variation = name[:i] + "c" + name[i + 1 :]
                if variation != name and variation not in seen:
                    variations.append(variation)
                    seen.add(variation)
                    medium_count += 1
                    if len(variations) >= count:
                        return variations

    # PRIORITY 3: Far variations (lower similarity) - Target ~10%
    far_count = 0
    max_far = max(1, int(count * 0.1))

    # Word order changes
    if len(tokens) >= 2 and far_count < max_far:
        # Swap first and last name
        if len(tokens) == 2:
            variation = f"{tokens[1]} {tokens[0]}"
            if variation != name and variation not in seen:
                variations.append(variation)
                seen.add(variation)
                far_count += 1
                if len(variations) >= count:
                    return variations

    # Remove middle names/words for lower similarity
    if len(tokens) > 2 and far_count < max_far:
        # Remove middle word
        variation = f"{tokens[0]} {tokens[-1]}"
        if variation != name and variation not in seen:
            variations.append(variation)
            seen.add(variation)
            far_count += 1
            if len(variations) >= count:
                return variations

    # Additional character changes for Far similarity
    for i, ch in enumerate(name):
        if far_count >= max_far:
            break
        if ch.isalpha():
            # Try different character substitutions for far similarity
            subs = {
                "c": "k",
                "k": "c",
                "s": "z",
                "z": "s",
                "ph": "f",
                "f": "ph",
                "v": "b",
                "b": "v",
                "j": "g",
                "g": "j",
            }
            for old, new in subs.items():
                if name[i : i + len(old)].lower() == old:
                    variation = name[:i] + new + name[i + len(old) :]
                    if variation != name and variation not in seen:
                        variations.append(variation)
                        seen.add(variation)
                        far_count += 1
                        if len(variations) >= count:
                            return variations
                        break  # Only one substitution per character

    # Additional far variations - more aggressive changes
    if far_count < max_far and len(tokens) >= 2:
        # Try different combinations
        if len(tokens) == 3:  # First Middle Last
            # Try: Last First Middle
            variation = f"{tokens[2]} {tokens[0]} {tokens[1]}"
            if variation != name and variation not in seen:
                variations.append(variation)
                seen.add(variation)
                far_count += 1
                if len(variations) >= count:
                    return variations

    return variations


def _generate_arabic_variations(name: str, count: int) -> List[str]:
    """Generate Arabic script variations"""
    variations = []

    # Character repetition
    for i, ch in enumerate(name):
        variation = name[:i] + ch + name[i:]
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Word order changes
    parts = name.split()
    if len(parts) >= 2:
        swapped = " ".join([parts[-1], parts[0]] + parts[1:-1])
        if swapped != name and swapped not in variations:
            variations.append(swapped)
            if len(variations) >= count:
                return variations

    # Add ibn/abu style
    if len(parts) >= 2:
        ibn = f"{parts[0]} بن {parts[1]}"
        if ibn != name and ibn not in variations:
            variations.append(ibn)

    return variations


def _generate_cjk_variations(name: str, count: int) -> List[str]:
    """Generate CJK script variations"""
    variations = []

    # Character repetition
    for i, ch in enumerate(name):
        variation = name[:i] + ch + name[i:]
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Character removal
    if len(name) > 1:
        for i in range(len(name)):
            variation = name[:i] + name[i + 1 :]
            if variation and variation != name and variation not in variations:
                variations.append(variation)
                if len(variations) >= count:
                    return variations

    return variations


def _generate_cyrillic_variations(name: str, count: int) -> List[str]:
    """Generate Cyrillic script variations"""
    variations = []

    # Vowel-like substitutions
    subs = {"а": ["о"], "е": ["ё"], "и": ["й"], "у": ["ю"], "о": ["ё"]}
    for i, ch in enumerate(name):
        low = ch.lower()
        if low in subs:
            for repl in subs[low]:
                new_ch = repl.upper() if ch.isupper() else repl
                variation = name[:i] + new_ch + name[i + 1 :]
                if variation != name and variation not in variations:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    return variations


def _generate_other_scripts_variations(name: str, count: int) -> List[str]:
    """Generate variations for other scripts"""
    variations = []

    # Character repetition
    for i, ch in enumerate(name):
        variation = name[:i] + ch + name[i:]
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Character removal
    if len(name) > 1:
        for i in range(len(name)):
            variation = name[:i] + name[i + 1 :]
            if variation and variation != name and variation not in variations:
                variations.append(variation)
                if len(variations) >= count:
                    return variations

    return variations


def _generate_quality_variations(name: str, count: int) -> List[str]:
    """Generate high-quality variations without number suffixes"""
    variations = []

    # Split name into parts
    parts = name.split()
    if len(parts) < 2:
        return variations

    # Generate variations by manipulating parts
    for i in range(len(parts)):
        for j in range(len(parts)):
            if i != j:
                # Swap parts
                new_parts = parts[:]
                new_parts[i], new_parts[j] = new_parts[j], new_parts[i]
                variation = " ".join(new_parts)
                if variation != name and variation not in variations:
                    variations.append(variation)
                    if len(variations) >= count:
                        return variations

    # Generate variations by removing parts
    for i in range(len(parts)):
        new_parts = [p for j, p in enumerate(parts) if j != i]
        if new_parts:
            variation = " ".join(new_parts)
            if variation != name and variation not in variations:
                variations.append(variation)
                if len(variations) >= count:
                    return variations

    # Generate variations by duplicating parts
    for i in range(len(parts)):
        new_parts = parts[:]
        new_parts.insert(i, parts[i])
        variation = " ".join(new_parts)
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Generate variations by abbreviating parts
    for i in range(len(parts)):
        if len(parts[i]) > 2:
            new_parts = parts[:]
            new_parts[i] = parts[i][0] + "."
            variation = " ".join(new_parts)
            if variation != name and variation not in variations:
                variations.append(variation)
                if len(variations) >= count:
                    return variations

    # Generate variations by changing case
    for i in range(len(parts)):
        new_parts = parts[:]
        new_parts[i] = new_parts[i].upper()
        variation = " ".join(new_parts)
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Generate variations by changing case (lower)
    for i in range(len(parts)):
        new_parts = parts[:]
        new_parts[i] = new_parts[i].lower()
        variation = " ".join(new_parts)
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Generate variations by adding common prefixes/suffixes
    prefixes = ["Mr.", "Ms.", "Dr.", "Prof.", "Mrs.", "Miss"]
    suffixes = ["Jr.", "Sr.", "II", "III", "IV", "V"]

    for prefix in prefixes:
        variation = f"{prefix} {name}"
        if variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    for suffix in suffixes:
        variation = f"{name} {suffix}"
        if variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Generate variations by character manipulation (no numbers)
    for i, part in enumerate(parts):
        if len(part) > 1:
            # Remove last character
            new_parts = parts[:]
            new_parts[i] = part[:-1]
            variation = " ".join(new_parts)
            if variation != name and variation not in variations:
                variations.append(variation)
                if len(variations) >= count:
                    return variations

            # Remove first character
            new_parts = parts[:]
            new_parts[i] = part[1:]
            variation = " ".join(new_parts)
            if variation != name and variation not in variations:
                variations.append(variation)
                if len(variations) >= count:
                    return variations

    # Generate variations by adding common middle names
    middle_names = [
        "James",
        "Michael",
        "David",
        "Robert",
        "William",
        "Richard",
        "Thomas",
        "Christopher",
    ]
    for middle in middle_names:
        new_parts = parts[:]
        new_parts.insert(1, middle)
        variation = " ".join(new_parts)
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    # Generate variations by hyphenating
    if len(parts) == 2:
        variation = f"{parts[0]}-{parts[1]}"
        if variation != name and variation not in variations:
            variations.append(variation)
            if len(variations) >= count:
                return variations

    return variations


def _phonetic_similarity(a: str, b: str) -> float:
    """Calculate phonetic similarity with better scoring for reward system"""
    if jellyfish is None:
        return _orthographic_similarity(a, b)

    try:
        # Use multiple algorithms with weighted combination for better distribution
        jaro_winkler = jellyfish.jaro_winkler(a, b)
        jaro_sim = jellyfish.jaro_similarity(a, b)

        # Binary phonetic matches (0 or 1) - these are too harsh
        soundex_match = 1.0 if jellyfish.soundex(a) == jellyfish.soundex(b) else 0.0
        metaphone_match = (
            1.0 if jellyfish.metaphone(a) == jellyfish.metaphone(b) else 0.0
        )
        nysiis_match = 1.0 if jellyfish.nysiis(a) == jellyfish.nysiis(b) else 0.0

        # Use weighted combination that gives better distribution
        # 70% Jaro-Winkler, 20% Jaro, 10% phonetic matches
        phonetic_score = (
            0.7 * jaro_winkler
            + 0.2 * jaro_sim
            + 0.1 * (soundex_match + metaphone_match + nysiis_match) / 3.0
        )

        # Apply scaling to better match reward system expectations
        # Scale up scores to get better distribution across Light/Medium/Far
        if phonetic_score >= 0.9:
            return min(1.0, phonetic_score * 1.1)  # Boost high scores slightly
        elif phonetic_score >= 0.7:
            return min(1.0, phonetic_score * 1.05)  # Small boost for medium scores
        elif phonetic_score >= 0.5:
            return min(1.0, phonetic_score * 1.1)  # Boost medium-low scores more
        else:
            return min(1.0, phonetic_score * 1.2)  # Boost low scores significantly

    except Exception:
        return _orthographic_similarity(a, b)


def _orthographic_similarity(a: str, b: str) -> float:
    """Calculate orthographic similarity using Levenshtein distance (matching reward.py)"""
    if Levenshtein is None:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()

    try:
        # Use Levenshtein distance to compare (matching reward.py approach)
        distance = Levenshtein.distance(a, b)
        max_len = max(len(a), len(b))

        # Calculate orthographic similarity score (0-1)
        return 1.0 - (distance / max_len)
    except Exception:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio()


def _classify_similarity(a: str, b: str) -> Tuple[float, float, str]:
    """Classify similarity into buckets matching reward.py boundaries"""
    phon = _phonetic_similarity(a, b)
    ortho = _orthographic_similarity(a, b)

    # Use phonetic similarity for classification (matching reward.py logic)
    # Reward system uses phonetic boundaries: Light(0.80-1.0), Medium(0.60-0.79), Far(0.30-0.59)
    if phon >= 0.80:
        bucket = "Light"
    elif phon >= 0.60:
        bucket = "Medium"
    elif phon >= 0.30:
        bucket = "Far"
    else:
        bucket = "Reject"

    return phon, ortho, bucket


def _generate_variations_by_similarity_target(
    name: str,
    similarity_target: str,
    count: int,
    script: ScriptType,
    country_info: Optional[CountryInfo] = None,
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> List[str]:
    """Generate variations targeting a specific similarity level (Light, Medium, Far)"""
    variations = []
    max_attempts = count * 15  # Try more times to get enough variations
    attempts = 0

    # Set similarity thresholds based on target - use phonetic similarity for classification
    if similarity_target == "Light":
        min_similarity = 0.80
        max_similarity = 1.0
    elif similarity_target == "Medium":
        min_similarity = 0.60
        max_similarity = 0.79
    elif similarity_target == "Far":
        min_similarity = 0.30
        max_similarity = 0.59
    else:
        return variations

    # Generate base variations based on script
    if script == ScriptType.LATIN:
        base_variations = _generate_latin_variations(name, max_attempts, country_info)
    elif script == ScriptType.ARABIC:
        base_variations = _generate_arabic_variations(name, max_attempts)
    elif script == ScriptType.CJK:
        base_variations = _generate_cjk_variations(name, max_attempts)
    elif script == ScriptType.CYRILLIC:
        base_variations = _generate_cyrillic_variations(name, max_attempts)
    else:
        base_variations = _generate_other_scripts_variations(name, max_attempts)

    # Filter variations by similarity target using phonetic similarity
    for var in base_variations:
        if var == name or var in variations:
            continue

        phon, ortho, bucket = _classify_similarity(name, var)

        # Use phonetic similarity for classification (matching reward system)
        if min_similarity <= phon < max_similarity:
            variations.append(var)
            if len(variations) >= count:
                break

        attempts += 1
        if attempts >= max_attempts:
            break

    # If we don't have enough variations, try generating more with quality methods
    if len(variations) < count:
        additional_variations = _generate_quality_variations(
            name, (count - len(variations)) * 5
        )
        for var in additional_variations:
            if var == name or var in variations:
                continue

            phon, ortho, bucket = _classify_similarity(name, var)

            if min_similarity <= phon < max_similarity:
                variations.append(var)
                if len(variations) >= count:
                    break

    # If still not enough, be more lenient for Light and Medium targets
    if len(variations) < count and similarity_target in ["Light", "Medium"]:
        for var in base_variations:
            if var == name or var in variations:
                continue

            phon, ortho, bucket = _classify_similarity(name, var)

            # Be more lenient: allow slightly lower scores
            if similarity_target == "Light" and phon >= 0.75:
                variations.append(var)
                if len(variations) >= count:
                    break
            elif similarity_target == "Medium" and 0.50 <= phon < 0.85:
                variations.append(var)
                if len(variations) >= count:
                    break

    return variations[:count]


def _select_variations_by_weights(
    variations: List[str],
    original: str,
    phonetic_config: Dict[str, float],
    orthographic_config: Dict[str, float],
    expected_count: int,
) -> List[str]:
    """Select variations based on weighted similarity configurations"""
    if not variations:
        return []

    # Score all variations
    scored_variations = []
    for var in variations:
        if var == original:
            continue

        phon, ortho, bucket = _classify_similarity(original, var)

        # Calculate weighted score based on configurations
        phon_weight = phonetic_config.get(bucket, 0.0)
        ortho_weight = orthographic_config.get(bucket, 0.0)
        combined_weight = (phon_weight + ortho_weight) / 2.0

        # Secondary sort signals: prefer higher combined similarity, then smaller edit distance
        combined_similarity = (phon + ortho) / 2.0
        try:
            distance = Levenshtein.distance(original, var)
        except Exception:
            from difflib import SequenceMatcher

            # Approximate: invert ratio to behave like distance (smaller is better)
            distance = 1.0 - SequenceMatcher(None, original, var).ratio()

        scored_variations.append(
            (var, combined_weight, combined_similarity, -distance, bucket)
        )

    # Sort by weight
    # Primary: combined_weight desc; Secondary: similarity desc; Tertiary: distance asc
    scored_variations.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

    # Select based on expected count
    selected = []
    for var, weight, _, _, bucket in scored_variations:
        if len(selected) >= expected_count:
            break
        if weight > 0:  # Only include variations with positive weight
            selected.append(var)

    return selected


def _enforce_first_last(original: str, variations: List[str]) -> List[str]:
    """Ensure every variation has exactly two tokens (first and last).

    - Removes leading honorifics.
    - Repairs single-token variations by pairing with the missing part from the original.
    - Trims extra tokens beyond two to strictly keep first and last.
    - Deduplicates (case-insensitive) while preserving order.
    """
    orig_parts = [p for p in original.split() if p.strip()]
    if not orig_parts:
        return []

    if len(orig_parts) >= 2:
        orig_first, orig_last = orig_parts[0], orig_parts[-1]
    else:
        # If the seed is a single token, we cannot reliably repair; require two-token variations
        orig_first, orig_last = orig_parts[0], None

    honorifics = {"mr.", "ms.", "mrs.", "miss", "dr.", "prof."}

    fixed: List[str] = []
    for v in variations:
        if not v or not isinstance(v, str):
            continue
        parts = [p for p in v.split() if p.strip()]
        if not parts:
            continue

        # Drop honorific prefixes
        while parts and parts[0].lower().rstrip(".") in {
            h.rstrip(".") for h in honorifics
        }:
            parts = parts[1:]
        if not parts:
            continue

        if len(parts) >= 2:
            fixed.append(" ".join(parts[:2]))
            continue

        # Single-token variation: attempt repair using original
        token = parts[0]
        if orig_last:
            if token.lower() == orig_last.lower():
                fixed.append(f"{orig_first} {token}")
            else:
                fixed.append(f"{token} {orig_last}")
        else:
            # Seed lacks last name; skip singletons
            continue

    # Deduplicate case-insensitively while preserving order and remove near-duplicates
    seen = set()
    out: List[str] = []
    for v in fixed:
        key = v.lower()
        if key in seen:
            continue
        # drop degenerates where first == last
        parts = v.split()
        if len(parts) == 2 and parts[0].lower() == parts[1].lower():
            continue
        # remove near-duplicates (Levenshtein distance <= 1)
        is_near_dup = False
        for kept in out:
            try:
                if Levenshtein.distance(kept, v) <= 1:
                    is_near_dup = True
                    break
            except Exception:
                from difflib import SequenceMatcher

                ratio = SequenceMatcher(None, kept, v).ratio()
                if ratio >= 0.95:
                    is_near_dup = True
                    break
        if is_near_dup:
            continue
        seen.add(key)
        out.append(v)
    return out


def generate_name_variations(
    name: str,
    expected_count: int = 10,
    miner_salt: int = 0,
    batch_salt: int = 0,
    country: Optional[str] = None,
    phonetic_config: Optional[Dict[str, float]] = None,
    orthographic_config: Optional[Dict[str, float]] = None,
    rule_percentage: Optional[int] = None,
    selected_rules: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate name variations with weighted similarity support and rule-based control.

    phonetic_config available options = [
        # Balanced distribution - high weight for balanced testing
        ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
        # Focus on Medium similarity - most common real-world scenario
        ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
        # Focus on Far similarity - important for edge cases
        ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
        # Light-Medium mix - moderate weight
        ({"Light": 0.5, "Medium": 0.5}, 0.12),
        # Medium-Far mix - moderate weight
        ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
        # Only Medium similarity - common case
        ({"Medium": 1.0}, 0.08),
        # High Light but not 100% - reduced frequency
        ({"Light": 0.7, "Medium": 0.3}, 0.05),
        # Only Far similarity - edge case
        ({"Far": 1.0}, 0.03),
        # Only Light similarity - reduced frequency
        ({"Light": 1.0}, 0.02),
        ]

    orthographic_config available options = [
        # Balanced distribution - high weight for balanced testing
        ({"Light": 0.3, "Medium": 0.4, "Far": 0.3}, 0.25),
        # Focus on Medium similarity - most common real-world scenario
        ({"Light": 0.2, "Medium": 0.6, "Far": 0.2}, 0.20),
        # Focus on Far similarity - important for edge cases
        ({"Light": 0.1, "Medium": 0.3, "Far": 0.6}, 0.15),
        # Light-Medium mix - moderate weight
        ({"Light": 0.5, "Medium": 0.5}, 0.12),
        # Medium-Far mix - moderate weight
        ({"Light": 0.1, "Medium": 0.5, "Far": 0.4}, 0.10),
        # Only Medium similarity - common case
        ({"Medium": 1.0}, 0.08),
        # High Light but not 100% - reduced frequency
        ({"Light": 0.7, "Medium": 0.3}, 0.05),
        # Only Far similarity - edge case
        ({"Far": 1.0}, 0.03),
        # Only Light similarity - reduced frequency
        ({"Light": 1.0}, 0.02),
    ]

    Args:
        name: The original name to generate variations for
        expected_count: Total number of variations to generate
        miner_salt: Salt for miner-specific randomization
        batch_salt: Salt for batch-specific randomization
        country: Country name for script detection
        phonetic_config: Phonetic similarity weight configuration -
            should be one of the config options from the available list above
        orthographic_config: Orthographic similarity weight configuration -
            should be one of the config options from the available list above
        rule_percentage: Percentage of rule-based variations (0-100)
        selected_rules: List of selected rules for variation generation

    Returns:
        List of name variations

    Note:
        - rule_percentage: 0-100, where 0 = all similarity-based, 100 = all rule-based
        - If rule_percentage is not specified, uses 0% (all similarity-based)
    """
    # Initialize managers
    country_manager = CountryManager()

    # Get country information
    country_info = country_manager.get_country_info(country)
    script = country_manager.get_script_for_country(country)

    # Determine rule-based and similarity-based counts based on rule_percentage
    rule_count = 0
    similarity_count = expected_count

    if rule_percentage and rule_percentage > 0:
        # Calculate counts based on percentage
        rule_count = max(1, int(expected_count * (rule_percentage / 100.0)))
        similarity_count = max(0, expected_count - rule_count)

    # Generate rule-based variations if needed
    rule_variations = []
    if rule_count > 0 and selected_rules:
        try:
            # Generate more rule-based variations to ensure we have enough
            rule_variations = generate_for_rules(
                name,
                selected_rules,
                per_rule_count=3,  # Increased from 2 to 3
                total_limit=rule_count * 2,  # Generate more than needed
                miner_salt=miner_salt,
                batch_salt=batch_salt,
            )
        except ImportError:
            pass  # Rule-based functionality not available

    # Initialize unique_variations for fallback use
    unique_variations = []

    # Generate similarity-based variations using config-driven approach
    similarity_variations = []
    if similarity_count > 0 and phonetic_config and orthographic_config:
        # Calculate how many variations to generate for each similarity level
        total_weight = sum(phonetic_config.values()) + sum(orthographic_config.values())
        if total_weight > 0:
            # Normalize weights
            normalized_phonetic = {
                k: v / sum(phonetic_config.values()) for k, v in phonetic_config.items()
            }
            normalized_orthographic = {
                k: v / sum(orthographic_config.values())
                for k, v in orthographic_config.items()
            }

            # Calculate counts for each similarity level
            for similarity_level in ["Light", "Medium", "Far"]:
                phon_weight = normalized_phonetic.get(similarity_level, 0.0)
                ortho_weight = normalized_orthographic.get(similarity_level, 0.0)
                combined_weight = (phon_weight + ortho_weight) / 2.0

                if combined_weight > 0:
                    level_count = max(1, int(similarity_count * combined_weight))
                    level_variations = _generate_variations_by_similarity_target(
                        name,
                        similarity_level,
                        level_count,
                        script,
                        country_info,
                        miner_salt,
                        batch_salt,
                    )
                    similarity_variations.extend(level_variations)
    else:
        # Fallback to old method if configs are not provided
        base_variations = []
        if script == ScriptType.LATIN:
            base_variations = _generate_latin_variations(
                name, expected_count * 10, country_info
            )
        elif script == ScriptType.ARABIC:
            base_variations = _generate_arabic_variations(name, expected_count * 10)
        elif script == ScriptType.CJK:
            base_variations = _generate_cjk_variations(name, expected_count * 10)
        elif script == ScriptType.CYRILLIC:
            base_variations = _generate_cyrillic_variations(name, expected_count * 10)
        else:
            base_variations = _generate_other_scripts_variations(
                name, expected_count * 10
            )

        # Remove duplicates and original
        unique_variations = list(
            dict.fromkeys([v for v in base_variations if v != name])
        )
        similarity_variations = unique_variations[:similarity_count]

    # Combine rule-based and similarity-based variations
    all_variations = rule_variations + similarity_variations

    # If we don't have enough variations, add more from the base variations
    if len(all_variations) < expected_count:
        # Add more variations from the base pool
        additional_needed = expected_count - len(all_variations)
        additional_variations = [
            v for v in unique_variations if v not in all_variations
        ][:additional_needed]
        all_variations.extend(additional_variations)

    # If we still don't have enough, generate more using quality methods
    if len(all_variations) < expected_count:
        still_needed = expected_count - len(all_variations)
        # Generate more quality variations using additional methods
        more_variations = _generate_quality_variations(name, still_needed * 2)
        for var in more_variations:
            if var not in all_variations and var != name:
                all_variations.append(var)
                if len(all_variations) >= expected_count:
                    break

    # If we still don't have enough, create better quality variations
    if len(all_variations) < expected_count:
        still_needed = expected_count - len(all_variations)
        # Create better quality variations instead of simple number suffixes
        better_variations = _generate_quality_variations(name, still_needed)
        for var in better_variations:
            if var not in all_variations and var != name:
                all_variations.append(var)
                if len(all_variations) >= expected_count:
                    break

    # If configs are provided, select by weighted similarity to better match validator targets
    if phonetic_config and orthographic_config:
        selected = _select_variations_by_weights(
            variations=all_variations,
            original=name,
            phonetic_config=phonetic_config,
            orthographic_config=orthographic_config,
            expected_count=expected_count,
        )
        # If not enough selected (due to weights), fill from the remainder preserving order
        if len(selected) < expected_count:
            selected_keys = set(v.lower() for v in selected)
            for v in all_variations:
                if len(selected) >= expected_count:
                    break
                if v.lower() not in selected_keys and v != name:
                    selected.append(v)
                    selected_keys.add(v.lower())
        all_variations = selected

    # Enforce first+last structure before final ordering
    all_variations = _enforce_first_last(name, all_variations)

    # Apply salted shuffle for consistent randomization
    rng_seed = (hash(name) & 0xFFFFFFFF) ^ (miner_salt * 10007) ^ (batch_salt * 97)
    rng = random.Random(rng_seed)
    rng.shuffle(all_variations)

    # Ensure we return exactly the expected count
    return all_variations[:expected_count]


def predict_name_score(
    original: str, variation: str, name_weight: float = 0.3
) -> float:
    """Predict validator-like score contribution for name variations"""
    phon = _phonetic_similarity(original, variation)
    ortho = _orthographic_similarity(original, variation)
    combined = (phon + ortho) / 2.0
    return name_weight * combined


if __name__ == "__main__":

    a = generate_name_variations(
        name="Carlos Javier Gómez",
        expected_count=15,
        miner_salt=1,
        batch_salt=1,
        country="Spain",
        phonetic_config={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
        orthographic_config={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
        rule_percentage=0,
        selected_rules=["remove_random_vowel", "swap_random_letter"],
    )
    print(a)
