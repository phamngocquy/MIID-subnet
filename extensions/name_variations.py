"""
Enhanced Name Variations Generator with Weighted Similarity Support

Features:
- Variable orthographic and phonetic weights
- Script detection (latin, arabic, cjk, cyrillic, other_scripts)
- Weighted similarity distribution selection
- High-quality variation generation
"""

import random
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


def _generate_latin_variations(name: str, count: int) -> List[str]:
    """Generate Latin script variations"""
    variations: List[str] = []

    # General Latin variations
    variations.extend(_generate_general_latin_variations(name, count))

    return variations[:count]


def _generate_general_latin_variations(name: str, count: int) -> List[str]:
    """Generate general Latin variations optimized for reward system scoring"""
    variations = []
    seen = set()  # Track seen variations to avoid duplicates

    # PRIORITY 1: Light variations (high similarity) - Target ~60% when aiming for high scores
    # Vowel swaps (very high similarity)
    vowel_swaps = {
        "a": ["e", "i", "o"],
        "e": ["a", "i"],
        "i": ["y", "a", "e"],
        "o": ["u", "a"],
        "u": ["o", "a"],
    }
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
                "c": ["k", "s"],
                "k": ["c", "q"],
                "s": ["z", "c"],
                "z": ["s"],
                "ph": ["f"],
                "f": ["ph"],
                "v": ["b", "w"],
                "b": ["v", "p"],
                "j": ["g", "y"],
                "g": ["j"],
                "w": ["v"],
                "p": ["b"],
                "t": ["d"],
                "d": ["t"],
                "m": ["n"],
                "n": ["m"],
            }
            low_ch = ch.lower()
            if low_ch in subs:
                for new in subs[low_ch]:
                    new_ch = new.upper() if ch.isupper() else new
                    variation = name[:i] + new_ch + name[i + 1 :]
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

    # If we still don't have enough variations, generate more using additional methods
    if len(variations) < count:
        # Generate more variations by character manipulation
        for i, ch in enumerate(name):
            if len(variations) >= count:
                break
            if ch.isalpha():
                # Try removing character
                variation = name[:i] + name[i + 1 :]
                if variation and variation != name and variation not in seen:
                    variations.append(variation)
                    seen.add(variation)
                    if len(variations) >= count:
                        return variations

                # Try duplicating character
                variation = name[:i] + ch + name[i:]
                if variation != name and variation not in seen:
                    variations.append(variation)
                    seen.add(variation)
                    if len(variations) >= count:
                        return variations

    # If still not enough, try case variations
    if len(variations) < count:
        tokens = name.split()
        for i in range(len(tokens)):
            if len(variations) >= count:
                break
            new_tokens = tokens[:]
            new_tokens[i] = new_tokens[i].upper()
            variation = " ".join(new_tokens)
            if variation != name and variation not in seen:
                variations.append(variation)
                seen.add(variation)
                if len(variations) >= count:
                    return variations

    # If still not enough, try adding common suffixes
    if len(variations) < count:
        suffixes = ["Jr", "Sr", "II", "III"]
        for suffix in suffixes:
            if len(variations) >= count:
                break
            variation = f"{name} {suffix}"
            if variation not in seen:
                variations.append(variation)
                seen.add(variation)
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
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> List[str]:
    """Generate variations targeting a specific similarity level (Light, Medium, Far)"""
    variations = []
    max_attempts = count * 20  # Try more times to get enough variations
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

    # Generate base variations based on script - use Latin variations as default
    base_variations = _generate_latin_variations(name, max_attempts)

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
            if similarity_target == "Light" and phon >= 0.70:
                variations.append(var)
                if len(variations) >= count:
                    break
            elif similarity_target == "Medium" and 0.40 <= phon < 0.90:
                variations.append(var)
                if len(variations) >= count:
                    break
            elif similarity_target == "Far" and phon >= 0.20:
                variations.append(var)
                if len(variations) >= count:
                    break

    # If still not enough, accept any variation that's not identical
    if len(variations) < count:
        for var in base_variations:
            if var != name and var not in variations:
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


def _select_variations_by_bucket_targets(
    variations: List[str],
    original: str,
    expected_count: int,
    targets: Dict[str, float],
) -> List[str]:
    """Select variations to match target bucket distribution (by phonetic bucket).

    targets example: {"Light": 0.3, "Medium": 0.4, "Far": 0.3}
    """
    if not variations:
        return []

    # Compute target counts per bucket
    light_target = int(expected_count * targets.get("Light", 0.0))
    medium_target = int(expected_count * targets.get("Medium", 0.0))
    # Ensure totals add up by putting remainder to Far
    far_target = max(0, expected_count - light_target - medium_target)

    # Classify and score variations
    bucket_to_scored: Dict[str, List[Tuple[str, float]]] = {
        "Light": [],
        "Medium": [],
        "Far": [],
    }
    for var in variations:
        if var == original:
            continue
        phon, ortho, bucket = _classify_similarity(original, var)
        if bucket not in bucket_to_scored:
            continue
        # Prefer higher combined similarity within the bucket
        combined_similarity = (phon + ortho) / 2.0
        bucket_to_scored[bucket].append((var, combined_similarity))

    # Sort each bucket by combined similarity desc
    for bucket in bucket_to_scored:
        bucket_to_scored[bucket].sort(key=lambda x: x[1], reverse=True)

    selected: List[str] = []

    def take_from(bucket: str, n: int):
        nonlocal selected
        pool = bucket_to_scored.get(bucket, [])
        for var, _ in pool:
            if len(selected) >= expected_count:
                break
            if var not in selected:
                selected.append(var)
            if (
                len(
                    [
                        v
                        for v in selected
                        if _classify_similarity(original, v)[2] == bucket
                    ]
                )
                >= n
            ):
                break

    # Take per target
    take_from("Light", light_target)
    take_from("Medium", medium_target)
    take_from("Far", far_target)

    # If still short, fill from remaining buckets by best similarity overall
    if len(selected) < expected_count:
        remaining: List[Tuple[str, float]] = []
        seen = set(selected)
        for bucket in ["Light", "Medium", "Far"]:
            for var, score in bucket_to_scored.get(bucket, []):
                if var not in seen:
                    remaining.append((var, score))
        remaining.sort(key=lambda x: x[1], reverse=True)
        for var, _ in remaining:
            if len(selected) >= expected_count:
                break
            selected.append(var)

    return selected[:expected_count]


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
        # Be less strict about near-duplicates - only remove exact duplicates
        seen.add(key)
        out.append(v)
    return out


def _generate_part_candidates(original: str) -> List[str]:
    # Generate simple candidates for a single token (part) without changing length too much
    candidates: List[str] = []
    seen = set()

    # LIGHT variations (high similarity) - minimal changes
    # Single vowel swaps (very high similarity)
    vowel_swaps = {"a": ["e"], "e": ["a"], "i": ["y"], "o": ["u"], "u": ["o"]}
    for i, ch in enumerate(original):
        low = ch.lower()
        if low in vowel_swaps:
            for repl in vowel_swaps[low]:
                new_ch = repl.upper() if ch.isupper() else repl
                v = original[:i] + new_ch + original[i + 1 :]
                if v != original and v not in seen:
                    candidates.append(v)
                    seen.add(v)

    # Single consonant swaps (high similarity)
    light_subs = {
        "c": ["k"],
        "k": ["c"],
        "s": ["z"],
        "z": ["s"],
        "v": ["b"],
        "b": ["v"],
    }
    for i, ch in enumerate(original):
        low = ch.lower()
        if low in light_subs:
            for repl in light_subs[low]:
                new_ch = repl.upper() if ch.isupper() else repl
                v = original[:i] + new_ch + original[i + 1 :]
                if v != original and v not in seen:
                    candidates.append(v)
                    seen.add(v)

    # MEDIUM variations (moderate similarity)
    # Adjacent transpositions
    if len(original) >= 4:
        for i in range(len(original) - 1):
            v = original[:i] + original[i + 1] + original[i] + original[i + 2 :]
            if v != original and v not in seen:
                candidates.append(v)
                seen.add(v)

    # More aggressive consonant swaps
    medium_subs = {
        "j": ["g", "y"],
        "g": ["j"],
        "t": ["d"],
        "d": ["t"],
        "m": ["n"],
        "n": ["m"],
    }
    for i, ch in enumerate(original):
        low = ch.lower()
        if low in medium_subs:
            for repl in medium_subs[low]:
                new_ch = repl.upper() if ch.isupper() else repl
                v = original[:i] + new_ch + original[i + 1 :]
                if v != original and v not in seen:
                    candidates.append(v)
                    seen.add(v)

    # FAR variations (lower similarity)
    # Character removal (only for longer names)
    if len(original) >= 4:
        for i in range(len(original)):
            v = original[:i] + original[i + 1 :]
            if v and v != original and v not in seen:
                candidates.append(v)
                seen.add(v)

    # Character duplication
    for i, ch in enumerate(original):
        v = original[:i] + ch + original[i:]
        if v != original and v not in seen:
            candidates.append(v)
            seen.add(v)

    # Abbreviation (prefix + dot) when long enough
    if len(original) >= 5:
        abbr = original[:3] + "."
        if abbr != original and abbr not in seen:
            candidates.append(abbr)
            seen.add(abbr)

    return candidates


def _filter_by_bucket(
    original: str, candidates: List[str], bucket: str, limit: int
) -> List[str]:
    out: List[str] = []
    for v in candidates:
        phon, _, _ = _classify_similarity(original, v)
        if bucket == "Light" and 0.80 <= phon <= 1.0:
            out.append(v)
        elif bucket == "Medium" and 0.60 <= phon < 0.80:
            out.append(v)
        elif bucket == "Far" and 0.30 <= phon < 0.60:
            out.append(v)
        if len(out) >= limit:
            break
    return out


def _generate_bucketed_part_variations(
    original_part: str, bucket: str, count: int
) -> List[str]:
    if count <= 0:
        return []
    base = _generate_part_candidates(original_part)
    vars_bucket = _filter_by_bucket(original_part, base, bucket, count * 3)
    # If insufficient, accept a bit looser boundaries
    if len(vars_bucket) < count:
        relaxed: List[str] = []
        for v in base:
            phon, _, _ = _classify_similarity(original_part, v)
            if bucket == "Light" and phon >= 0.75:
                relaxed.append(v)
            elif bucket == "Medium" and 0.50 <= phon < 0.85:
                relaxed.append(v)
            elif bucket == "Far" and 0.25 <= phon < 0.65:
                relaxed.append(v)
            if len(relaxed) >= count:
                break
        vars_bucket.extend([v for v in relaxed if v not in vars_bucket])
    # Trim
    dedup = []
    seen = set()
    for v in vars_bucket:
        k = v.lower()
        if k not in seen:
            dedup.append(v)
            seen.add(k)
        if len(dedup) >= count:
            break
    return dedup


def _compose_fullname_variations(
    first: str, last: str, plan: List[Tuple[str, str, int]]
) -> List[str]:
    """Compose full-name variations using a plan of (first_bucket, last_bucket, count).
    Use empty bucket string to mean keep original for that side.
    """
    out: List[str] = []
    seen = set()
    for f_bucket, l_bucket, cnt in plan:
        f_vars = [first]
        l_vars = [last]
        if f_bucket:
            f_vars = _generate_bucketed_part_variations(first, f_bucket, cnt)
            if not f_vars:
                f_vars = [first]
        if l_bucket:
            l_vars = _generate_bucketed_part_variations(last, l_bucket, cnt)
            if not l_vars:
                l_vars = [last]
        # Compose pairs, prefer changing one side at a time
        i = 0
        while len(out) < len(out) + cnt and i < max(len(f_vars), len(l_vars)):
            f = f_vars[min(i, len(f_vars) - 1)]
            l = l_vars[min(i, len(l_vars) - 1)]
            v = f"{f} {l}"
            key = v.lower()
            if key not in seen and (f != first or l != last):
                out.append(v)
                seen.add(key)
            i += 1
        # If still short, cross product up to cnt
        if len(out) < len(out) + cnt:
            for fv in f_vars:
                for lv in l_vars:
                    if len(out) >= len(out) + cnt:
                        break
                    v = f"{fv} {lv}"
                    key = v.lower()
                    if key not in seen and (fv != first or lv != last):
                        out.append(v)
                        seen.add(key)
                if len(out) >= len(out) + cnt:
                    break
    return out


# --- Updated generation pipeline focusing on part-based composition ---


def generate_name_variations(
    name: str,
    expected_count: int = 10,
    miner_salt: int = 0,
    batch_salt: int = 0,
    phonetic_config: Optional[Dict[str, float]] = None,
    orthographic_config: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Generate name variations with weighted similarity support.

    Args:
        name: The original name to generate variations for
        expected_count: Total number of variations to generate
        miner_salt: Salt for miner-specific randomization
        batch_salt: Salt for batch-specific randomization
        phonetic_config: Phonetic similarity weight configuration
        orthographic_config: Orthographic similarity weight configuration

    Returns:
        List of name variations
    """

    # Determine rule-based and similarity-based counts based on rule_percentage
    similarity_count = expected_count

    # Initialize unique_variations for fallback use
    unique_variations = []

    # Generate similarity-based variations using config-driven approach
    similarity_variations = []

    # Generate variations that preserve the original name structure
    parts = [p for p in name.split() if p.strip()]

    if len(parts) == 1:
        # Single-part names: generate character-level variations
        single_name = parts[0]
        part_variations = _generate_part_candidates(single_name)

        # Take up to expected_count variations
        for var in part_variations[:expected_count]:
            if var != single_name and len(var) > 0:
                similarity_variations.append(var)
                if len(similarity_variations) >= expected_count:
                    break

        # If we still need more variations, generate additional ones
        if len(similarity_variations) < expected_count:
            # Try adding common surnames to create two-part names
            common_surnames = [
                "Smith",
                "Johnson",
                "Williams",
                "Brown",
                "Jones",
                "Garcia",
                "Miller",
                "Davis",
            ]
            for surname in common_surnames:
                if len(similarity_variations) >= expected_count:
                    break
                # Create variations like "Condrát'jilapin Smith", "Condrát'jilapin Johnson", etc.
                for var in part_variations[
                    :3
                ]:  # Use first 3 variations of the single name
                    if var != single_name:
                        two_part_variation = f"{var} {surname}"
                        if two_part_variation not in similarity_variations:
                            similarity_variations.append(two_part_variation)
                            if len(similarity_variations) >= expected_count:
                                break

                # Also try original name with surname
                two_part_original = f"{single_name} {surname}"
                if two_part_original not in similarity_variations:
                    similarity_variations.append(two_part_original)
                    if len(similarity_variations) >= expected_count:
                        break

    elif len(parts) >= 2:
        # Always preserve the full structure: First [Middle] Last
        orig_first, orig_last = parts[0], parts[-1]

        if len(parts) == 3:
            # 3-part names: First Middle Last
            orig_middle = parts[1]
            first_variations = _generate_part_candidates(orig_first)
            middle_variations = _generate_part_candidates(orig_middle)
            last_variations = _generate_part_candidates(orig_last)

            # Generate variations maintaining "First Middle Last" structure
            # First name variations (keep middle and last same)
            for f_var in first_variations[: expected_count // 3]:
                if f_var != orig_first and len(f_var) > 0:
                    similarity_variations.append(f"{f_var} {orig_middle} {orig_last}")

            # Middle name variations (keep first and last same)
            for m_var in middle_variations[: expected_count // 3]:
                if m_var != orig_middle and len(m_var) > 0:
                    similarity_variations.append(f"{orig_first} {m_var} {orig_last}")

            # Last name variations (keep first and middle same)
            for l_var in last_variations[: expected_count // 3]:
                if l_var != orig_last and len(l_var) > 0:
                    similarity_variations.append(f"{orig_first} {orig_middle} {l_var}")

            # Some combinations (First + Last, keep middle)
            for f_var in first_variations[:2]:
                for l_var in last_variations[:2]:
                    if (
                        (f_var != orig_first or l_var != orig_last)
                        and len(f_var) > 0
                        and len(l_var) > 0
                    ):
                        similarity_variations.append(f"{f_var} {orig_middle} {l_var}")
                        if len(similarity_variations) >= expected_count:
                            break
                if len(similarity_variations) >= expected_count:
                    break
        else:
            # 2-part names: First Last
            first_variations = _generate_part_candidates(orig_first)
            last_variations = _generate_part_candidates(orig_last)

            # First name variations (keep last name same)
            for f_var in first_variations[: expected_count // 2]:
                if f_var != orig_first:
                    similarity_variations.append(f"{f_var} {orig_last}")

            # Last name variations (keep first name same)
            for l_var in last_variations[: expected_count // 2]:
                if l_var != orig_last:
                    similarity_variations.append(f"{orig_first} {l_var}")

            # Some both-changed variations
            for f_var in first_variations[:3]:
                for l_var in last_variations[:3]:
                    if f_var != orig_first or l_var != orig_last:
                        similarity_variations.append(f"{f_var} {l_var}")
                        if len(similarity_variations) >= expected_count:
                            break
                if len(similarity_variations) >= expected_count:
                    break

        # Remove duplicates and cap
        seen = set()
        unique_variations = []
        for var in similarity_variations:
            if var not in seen and var != name:
                unique_variations.append(var)
                seen.add(var)
                if len(unique_variations) >= expected_count:
                    break
        similarity_variations = unique_variations

    # Generate additional variations if needed using proper structure-preserving method
    if len(similarity_variations) < expected_count:
        needed = expected_count - len(similarity_variations)
        # Generate more variations using the same structure-preserving approach
        parts = [p for p in name.split() if p.strip()]
        if len(parts) >= 2:
            orig_first, orig_last = parts[0], parts[-1]

            if len(parts) == 3:
                orig_middle = parts[1]
                # Generate more first name variations
                first_variations = _generate_part_candidates(orig_first)
                for f_var in first_variations[expected_count // 3 :]:
                    if (
                        f_var != orig_first
                        and len(f_var) > 0
                        and len(similarity_variations) < expected_count
                    ):
                        similarity_variations.append(
                            f"{f_var} {orig_middle} {orig_last}"
                        )

                # Generate more last name variations
                last_variations = _generate_part_candidates(orig_last)
                for l_var in last_variations[expected_count // 3 :]:
                    if (
                        l_var != orig_last
                        and len(l_var) > 0
                        and len(similarity_variations) < expected_count
                    ):
                        similarity_variations.append(
                            f"{orig_first} {orig_middle} {l_var}"
                        )
            else:
                # 2-part names
                first_variations = _generate_part_candidates(orig_first)
                last_variations = _generate_part_candidates(orig_last)

                for f_var in first_variations[expected_count // 2 :]:
                    if (
                        f_var != orig_first
                        and len(f_var) > 0
                        and len(similarity_variations) < expected_count
                    ):
                        similarity_variations.append(f"{f_var} {orig_last}")

                for l_var in last_variations[expected_count // 2 :]:
                    if (
                        l_var != orig_last
                        and len(l_var) > 0
                        and len(similarity_variations) < expected_count
                    ):
                        similarity_variations.append(f"{orig_first} {l_var}")

    # Combine rule-based and similarity-based variations
    all_variations = similarity_variations

    # Debug: Show what we have so far
    print(f"Generated {len(similarity_variations)} similarity variations")
    print(f"First 5: {similarity_variations[:5]}")

    # Use only our structure-preserving variations (no fallback to avoid malformed variations)

    # If configs are provided, select by bucket targets (30/40/30) to match evaluator
    if phonetic_config and orthographic_config:
        print(f"Before bucket-target selection: {all_variations[:5]}")
        bucket_targets = {
            "Light": float(phonetic_config.get("Light", 0)),
            "Medium": float(phonetic_config.get("Medium", 0)),
            "Far": float(phonetic_config.get("Far", 0)),
        }
        selected = _select_variations_by_bucket_targets(
            variations=all_variations,
            original=name,
            expected_count=expected_count,
            targets=bucket_targets,
        )
        # Fill if short
        if len(selected) < expected_count:
            selected_keys = set(v.lower() for v in selected)
            for v in all_variations:
                if len(selected) >= expected_count:
                    break
                if v.lower() not in selected_keys and v != name:
                    selected.append(v)
                    selected_keys.add(v.lower())
        all_variations = selected
        print(f"After bucket-target selection: {all_variations[:5]}")

    # Skip _enforce_first_last for 3-part names to preserve structure

    # If we still don't have enough after enforcing first+last, generate more
    if len(all_variations) < expected_count:
        still_needed = expected_count - len(all_variations)
        # Generate more variations using quality methods
        more_variations = _generate_quality_variations(name, still_needed * 3)
        for var in more_variations:
            if var not in all_variations and var != name:
                all_variations.append(var)
                if len(all_variations) >= expected_count:
                    break

    # If still not enough, create simple variations
    if len(all_variations) < expected_count:
        tokens = name.split()
        if len(tokens) >= 2:
            # Create simple word order variations
            simple_variations = [
                f"{tokens[1]} {tokens[0]}",  # Last First
            ]
            if len(tokens) >= 3:
                simple_variations.extend(
                    [
                        f"{tokens[2]} {tokens[0]}",  # Last First (skip middle)
                        f"{tokens[0]} {tokens[2]}",  # First Last (skip middle)
                    ]
                )

            for var in simple_variations:
                if var not in all_variations and var != name:
                    all_variations.append(var)
                    if len(all_variations) >= expected_count:
                        break

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
    original_name = "Vitalenksin"
    # a = generate_name_variations(
    #     name=original_name,
    #     expected_count=15,
    #     miner_salt=1,
    #     batch_salt=1,
    #     phonetic_config={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
    #     orthographic_config={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
    # )
    name_variation = generate_name_variations(
        name=original_name,
        expected_count=15,
        miner_salt=1,
        batch_salt=1,
        phonetic_config={"Light": 1},
        orthographic_config={"Light": 1},
    )
    print(name_variation)

    # Evaluate quality using validator reward logic (both phonetic-only and combined)
    try:
        from MIID.validator.reward import (
            calculate_variation_quality,
            calculate_variation_quality_phonetic_only,
        )

        # Diagnostic: Check similarity scores of generated variations
        print("=== DIAGNOSTIC ANALYSIS ===")
        first_name = (
            original_name.split()[0] if original_name.split() else original_name
        )
        last_name = (
            original_name.split()[-1] if len(original_name.split()) > 1 else None
        )

        print(f"Original: {original_name}")
        print(f"Parts: {original_name.split()}")
        print(f"First: {first_name}, Last: {last_name}")
        print(f"Generated {len(name_variation)} variations")

        # Analyze similarity distribution
        phonetic_scores = []
        orthographic_scores = []
        for var in name_variation:
            parts = var.split()
            if len(parts) >= 2:
                first_var = parts[0]
                last_var = parts[-1]

                # Calculate similarities for first name
                phon_first = _phonetic_similarity(first_name, first_var)
                ortho_first = _orthographic_similarity(first_name, first_var)

                print(phon_first, ortho_first)

                # Calculate similarities for last name
                if last_name:
                    phon_last = _phonetic_similarity(last_name, last_var)
                    ortho_last = _orthographic_similarity(last_name, last_var)

                    # Average for full name
                    phon_avg = (phon_first + phon_last) / 2
                    ortho_avg = (ortho_first + ortho_last) / 2
                else:
                    phon_avg = phon_first
                    ortho_avg = ortho_first

                phonetic_scores.append(phon_avg)
                orthographic_scores.append(ortho_avg)

        # Analyze distribution
        def analyze_distribution(scores, name):
            light = sum(1 for s in scores if 0.80 <= s <= 1.00)
            medium = sum(1 for s in scores if 0.60 <= s < 0.80)
            far = sum(1 for s in scores if 0.30 <= s < 0.60)
            reject = sum(1 for s in scores if s < 0.30)

            print(f"{name} distribution:")
            print(
                f"  Light (0.80-1.00): {light}/{len(scores)} ({light/len(scores)*100:.1f}%)"
            )
            print(
                f"  Medium (0.60-0.79): {medium}/{len(scores)} ({medium/len(scores)*100:.1f}%)"
            )
            print(
                f"  Far (0.30-0.59): {far}/{len(scores)} ({far/len(scores)*100:.1f}%)"
            )
            print(
                f"  Reject (<0.30): {reject}/{len(scores)} ({reject/len(scores)*100:.1f}%)"
            )
            print(f"  Avg score: {sum(scores)/len(scores):.3f}")
            print(f"  Min score: {min(scores):.3f}, Max score: {max(scores):.3f}")

        if phonetic_scores:
            analyze_distribution(phonetic_scores, "Phonetic")
        if orthographic_scores:
            analyze_distribution(orthographic_scores, "Orthographic")

        print("=== END DIAGNOSTIC ===\n")

        # Debug: Show individual variation analysis
        print("=== INDIVIDUAL VARIATION ANALYSIS ===")
        for i, var in enumerate(name_variation[:5]):  # Show first 5
            parts = var.split()
            if len(parts) >= 2:
                first_var, last_var = parts[0], parts[-1]
                phon_first = _phonetic_similarity(first_name, first_var)
                if last_name:
                    phon_last = _phonetic_similarity(last_name, last_var)
                    print(
                        f"  {i+1}. '{var}' -> First: '{first_var}' (phon: {phon_first:.3f}), Last: '{last_var}' (phon: {phon_last:.3f})"
                    )
                else:
                    print(f"  {i+1}. '{var}' -> Single name variation")
            else:
                # Single-part variation
                phon_single = _phonetic_similarity(first_name, var)
                print(f"  {i+1}. '{var}' -> Single name (phon: {phon_single:.3f})")

        # Debug: Show what part candidates are being generated
        print("\n=== PART CANDIDATES DEBUG ===")
        first_candidates = _generate_part_candidates(first_name)
        print(f"First name '{first_name}' candidates: {first_candidates[:5]}")
        if last_name:
            last_candidates = _generate_part_candidates(last_name)
            print(f"Last name '{last_name}' candidates: {last_candidates[:5]}")
        else:
            print("Last name: None (single-part name)")
        print("=== END PART CANDIDATES DEBUG ===\n")

        # Debug: Show what similarity_variations contains before any processing
        print("=== SIMILARITY VARIATIONS DEBUG ===")
        print(f"Generated {len(similarity_variations)} similarity variations")
        print(f"First 5: {similarity_variations[:5]}")
        print("=== END SIMILARITY VARIATIONS DEBUG ===\n")

        # Phonetic-only
        # p_final, p_base, p_details = calculate_variation_quality_phonetic_only(
        #     original_name=original_name,
        #     variations=a,
        #     phonetic_similarity={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
        #     expected_count=15,
        # )
        p_final, p_base, p_details = calculate_variation_quality_phonetic_only(
            original_name=original_name,
            variations=name_variation,
            phonetic_similarity={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
            expected_count=15,
        )
        print(
            {
                "method": "phonetic_only",
                "quality_final_score": float(p_final),
                "quality_base_score": float(p_base),
                "details_keys": (
                    list(p_details.keys())
                    if isinstance(p_details, dict)
                    else type(p_details).__name__
                ),
            }
        )

        # Show detailed metrics for debugging
        if isinstance(p_details, dict):
            print("Phonetic-only detailed metrics:")
            for key, value in p_details.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

        # Combined (phonetic + orthographic)
        c_final, c_base, c_details = calculate_variation_quality(
            original_name=original_name,
            variations=name_variation,
            phonetic_similarity={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
            orthographic_similarity={"Light": 0.3, "Medium": 0.4, "Far": 0.3},
            expected_count=15,
        )
        print(
            {
                "method": "combined",
                "quality_final_score": float(c_final),
                "quality_base_score": float(c_base),
                "details_keys": (
                    list(c_details.keys())
                    if isinstance(c_details, dict)
                    else type(c_details).__name__
                ),
            }
        )
    except Exception as e:
        print({"quality_eval_error": str(e)})
