#!/usr/bin/env python3
"""
Generate name variations targeted to phonetic_similarity and orthographic_similarity labels.

- Accepts distributions like {"Light": 0.3, "Medium": 0.5, "Far": 0.2}
- Adapts when weights change (e.g., Medium can be changed)
- Stays within the same script as the seed name
- Uses stable randomness via miner_salt and batch_salt
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import random
from difflib import SequenceMatcher

try:
    import jellyfish  # optional for better phonetic similarity
except Exception:
    jellyfish = None

try:
    import Levenshtein  # optional for better orthographic similarity
except Exception:
    Levenshtein = None


# ----------------------------- Utilities -----------------------------

def detect_script(text: str) -> str:
    for ch in text:
        print(ch, ord(ch))
    if all(ord(ch) < 128 for ch in text):
        return "latin"
    ranges = [
        ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
        ("cyrillic", [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
        ("cjk", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
        ("greek", [(0x0370, 0x03FF)]),
        ("hebrew", [(0x0590, 0x05FF)]),
        ("devanagari", [(0x0900, 0x097F)]),
        ("thai", [(0x0E00, 0x0E7F)]),
    ]
    for script, blocks in ranges:
        for start, end in blocks:
            if any(start <= ord(ch) <= end for ch in text):
                return script
    return "other_scripts"


def _rng(name: str, miner_salt: int, batch_salt: int) -> random.Random:
    seed = (hash(name) & 0xFFFFFFFF) ^ (miner_salt * 10007) ^ (batch_salt * 97)
    return random.Random(seed)


def _phonetic_similarity(a: str, b: str) -> float:
    if jellyfish is not None:
        try:
            return float(jellyfish.jaro_winkler(a, b))
        except Exception:
            pass
    return float(SequenceMatcher(None, a, b).ratio())


def _orthographic_similarity(a: str, b: str) -> float:
    if Levenshtein is not None:
        try:
            # Normalize to 0..1 similarity like reward.py does
            dist = Levenshtein.distance(a, b)
            max_len = max(len(a), len(b)) or 1
            return 1.0 - (dist / max_len)
        except Exception:
            pass
    return float(SequenceMatcher(None, a, b).ratio())


# Boundaries aligned with MIID/validator/reward.py
PHONETIC_BOUNDARIES = {
    "Light": (0.8, 1.0),
    "Medium": (0.6, 0.8),
    "Far": (0.3, 0.6),
}
ORTHO_BOUNDARIES = {
    "Light": (0.7, 1.0),
    "Medium": (0.5, 0.7),
    "Far": (0.2, 0.5),
}


def _bucket_of(x: float, boundaries: Dict[str, Tuple[float, float]]) -> str:
    for k, (lo, hi) in boundaries.items():
        if lo <= x <= hi:
            return k
    return "Far"


# ----------------------- Script-specific generators -----------------------

def _latin_candidates(name: str) -> List[str]:
    out: List[str] = []
    tokens = name.split()
    if len(tokens) >= 2:
        first, middle, last = tokens[0], tokens[1:-1], tokens[-1]
        # very light: strong prefix abbreviation
        if first:
            prefix_len = 5 if len(first) >= 8 else (4 if len(first) >= 6 else 3)
            out.append(" ".join([(first[:prefix_len] + "."), *middle, last]).strip())
        # light: one vowel swap
        vowel_swaps = {"a":"e","e":"a","i":"y","y":"i","o":"u","u":"o"}
        for i, ch in enumerate(first):
            low = ch.lower()
            if low in vowel_swaps:
                repl = vowel_swaps[low]
                new_ch = repl.upper() if ch.isupper() else repl
                tweaked = first[:i] + new_ch + first[i+1:]
                out.append(" ".join([tweaked, *middle, last]).strip())
                break
        # medium: duplicate one consonant after preserved prefix
        prefix_len = 5 if len(first) >= 8 else (4 if len(first) >= 6 else 3)
        for i, ch in enumerate(first):
            if i < prefix_len:
                continue
            if ch.isalpha() and ch.lower() not in "aeiou":
                dup = first[:i] + ch + first[i:]
                out.append(" ".join([dup, *middle, last]).strip())
                break
        # medium: near-keyboard swap after prefix
        adjacency = {
            "q":"w","w":"e","e":"r","r":"t","t":"y","y":"u","u":"i","i":"o","o":"p",
            "a":"s","s":"d","d":"f","f":"g","g":"h","h":"j","j":"k","k":"l",
            "z":"x","x":"c","c":"v","v":"b","b":"n","n":"m",
        }
        pos = max(1, prefix_len-1) if first else 0
        if first and pos < len(first):
            ch = first[pos]
            low = ch.lower()
            if low in adjacency and adjacency[low]:
                repl = adjacency[low][0]
                repl = repl.upper() if ch.isupper() else repl
                tweaked = first[:pos] + repl + first[pos+1:]
                out.append(" ".join([tweaked, *middle, last]).strip())
    else:
        # single-part latin name: apply light vowel swap and consonant dup
        s = name
        vowel_swaps = {"a":"e","e":"a","i":"y","y":"i","o":"u","u":"o"}
        for i, ch in enumerate(s):
            low = ch.lower()
            if low in vowel_swaps:
                repl = vowel_swaps[low]
                new_ch = repl.upper() if ch.isupper() else repl
                out.append(s[:i] + new_ch + s[i+1:])
                break
        for i, ch in enumerate(s):
            if ch.isalpha() and ch.lower() not in "aeiou":
                out.append(s[:i] + ch + s[i:])
                break
    return list(dict.fromkeys(v for v in out if v and v != name))


def _arabic_candidates(name: str) -> List[str]:
    out: List[str] = []
    # repetition of a char
    for i, ch in enumerate(name):
        out.append(name[:i] + ch + name[i:])
        break
    # remove one char (if length > 1)
    if len(name) > 1:
        out.append(name[1:])
    return list(dict.fromkeys(v for v in out if v and v != name))


def _cyrillic_candidates(name: str) -> List[str]:
    out: List[str] = []
    subs = {"а":"о","е":"ё","и":"й","у":"ю","о":"ё"}
    for i, ch in enumerate(name):
        low = ch.lower()
        if low in subs:
            rep = subs[low]
            rep = rep.upper() if ch.isupper() else rep
            out.append(name[:i] + rep + name[i+1:])
            break
    if len(name) > 1:
        out.append(name[:-1])
    return list(dict.fromkeys(v for v in out if v and v != name))


def _cjk_candidates(name: str) -> List[str]:
    out: List[str] = []
    # repetition
    for i, ch in enumerate(name):
        out.append(name[:i] + ch + name[i:])
        break
    # removal
    if len(name) > 1:
        out.append(name[:-1])
    return list(dict.fromkeys(v for v in out if v and v != name))


def _greek_candidates(name: str) -> List[str]:
    # conservative vowel-like substitutions + repetition/removal
    out: List[str] = []
    subs = {"ά":"α","έ":"ε","ί":"ι","ό":"ο","ύ":"υ","ή":"η","ώ":"ω"}
    for i, ch in enumerate(name):
        if ch in subs:
            rep = subs[ch]
            out.append(name[:i] + rep + name[i+1:])
            break
    if len(name) > 1:
        out.append(name[:-1])
    for i, ch in enumerate(name):
        out.append(name[:i] + ch + name[i:])
        break
    return list(dict.fromkeys(v for v in out if v and v != name))


def _hebrew_candidates(name: str) -> List[str]:
    out: List[str] = []
    # repetition then removal
    for i, ch in enumerate(name):
        out.append(name[:i] + ch + name[i:])
        break
    if len(name) > 1:
        out.append(name[:-1])
    return list(dict.fromkeys(v for v in out if v and v != name))


def _devanagari_candidates(name: str) -> List[str]:
    out: List[str] = []
    # repetition then removal
    for i, ch in enumerate(name):
        out.append(name[:i] + ch + name[i:])
        break
    if len(name) > 1:
        out.append(name[:-1])
    return list(dict.fromkeys(v for v in out if v and v != name))


def _thai_candidates(name: str) -> List[str]:
    out: List[str] = []
    # repetition then removal
    for i, ch in enumerate(name):
        out.append(name[:i] + ch + name[i:])
        break
    if len(name) > 1:
        out.append(name[:-1])
    return list(dict.fromkeys(v for v in out if v and v != name))


def _generate_candidates(name: str) -> List[str]:
    script = detect_script(name)
    print(script, name)
    if script == "latin":
        return _latin_candidates(name)
    if script == "arabic":
        return _arabic_candidates(name)
    if script == "cyrillic":
        return _cyrillic_candidates(name)
    if script == "cjk":
        return _cjk_candidates(name)
    if script == "greek":
        return _greek_candidates(name)
    if script == "hebrew":
        return _hebrew_candidates(name)
    if script == "devanagari":
        return _devanagari_candidates(name)
    if script == "thai":
        return _thai_candidates(name)
    return _latin_candidates(name)


# ----------------------------- Main API -----------------------------

def generate_by_similarity(
    name: str,
    variation_count: int,
    phonetic_similarity: Optional[Dict[str, float]] = None,
    orthographic_similarity: Optional[Dict[str, float]] = None,
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> List[str]:
    """Generate name variations to match provided phonetic/orthographic distributions.

    - Computes per-bucket targets from distributions
    - Selects candidates whose individual phonetic and orthographic buckets
      match the requested buckets as closely as possible
    """
    # defaults (Medium if not provided)
    phonetic_similarity = phonetic_similarity or {"Medium": 1.0}
    orthographic_similarity = orthographic_similarity or {"Medium": 1.0}

    # Normalize distributions
    def norm(d: Dict[str, float]) -> Dict[str, float]:
        keys = ["Light", "Medium", "Far"]
        s = sum(d.get(k, 0.0) for k in keys)
        return {k: (d.get(k, 0.0) / s) if s > 0 else (1.0 if k == "Medium" else 0.0) for k in keys}

    phon_d = norm(phonetic_similarity)
    ortho_d = norm(orthographic_similarity)

    # Compute targets per bucket (simple average between phonetic and orthographic requests)
    combined = {k: (phon_d.get(k, 0.0) + ortho_d.get(k, 0.0)) / 2.0 for k in ["Light", "Medium", "Far"]}
    n_light = max(0, int(variation_count * combined.get("Light", 0.0)))
    n_medium = max(0, int(variation_count * combined.get("Medium", 0.0)))
    n_far = max(0, int(variation_count * combined.get("Far", 0.0)))
    while n_light + n_medium + n_far < variation_count:
        # fill remainder into dominant bucket
        dom = max(combined.items(), key=lambda t: t[1])[0]
        if dom == "Light":
            n_light += 1
        elif dom == "Far":
            n_far += 1
        else:
            n_medium += 1

    # Build candidate pool
    base = _generate_candidates(name)
    print(base)
    # add some extra local tweaks by shuffling and repeating generation if needed
    rng = _rng(name, miner_salt, batch_salt)
    rng.shuffle(base)

    # Score all candidates
    scored: List[Tuple[str, float, float, str, str]] = []  # (var, p, o, pb, ob)
    seen = set()
    for v in base:
        if v in seen:
            continue
        seen.add(v)
        p = _phonetic_similarity(name, v)
        o = _orthographic_similarity(name, v)
        pb = _bucket_of(p, PHONETIC_BOUNDARIES)
        ob = _bucket_of(o, ORTHO_BOUNDARIES)
        scored.append((v, p, o, pb, ob))

    # Helper to pick best matches for a bucket: prefer items where both pb/ob equal bucket,
    # then those where at least one matches; sort by avg similarity desc to keep quality high
    def pick(bucket: str, need: int, pool: List[Tuple[str, float, float, str, str]], taken: set) -> List[str]:
        ranked = []
        for v, p, o, pb, ob in pool:
            if v in taken:
                continue
            match_score = (1 if pb == bucket else 0) + (1 if ob == bucket else 0)
            ranked.append((match_score, (p + o) / 2.0, v))
        ranked.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out: List[str] = []
        for ms, _, v in ranked:
            if len(out) >= need:
                break
            if ms == 0 and need > 0:
                # if still need, allow partial match late in the list
                pass
            out.append(v)
        return out[:need]

    selected: List[str] = []
    taken: set = set()
    for bucket, need in (("Light", n_light), ("Medium", n_medium), ("Far", n_far)):
        picks = pick(bucket, need, scored, taken)
        print(picks)
        for v in picks:
            if v not in taken:
                selected.append(v)
                taken.add(v)

    # If still short, backfill with highest average similarity overall
    print(variation_count)
    print(len(selected))
    if len(selected) < variation_count:
        remaining = sorted([( (p+o)/2.0, v) for v,p,o,_,_ in scored if v not in taken], reverse=True)
        for _, v in remaining:
            selected.append(v)
            taken.add(v)
            if len(selected) >= variation_count:
                break

    # Stable shuffle
    rng.shuffle(selected)
    return selected[:variation_count]


if __name__ == "__main__":
    name = "Phạm Ngọc Quý"
    phon = {"Medium": 0.5}
    ortho = {"Medium": 0.5}
    print(generate_by_similarity(name, 15, phon, ortho, miner_salt=1, batch_salt=13))


