#!/usr/bin/env python3
"""
Rule-based name variation generators aligned with MIID.validator.rule_evaluator RULE_EVALUATORS.

This module provides small, deterministic generators that produce variations intended
to satisfy each validator rule heuristic. Use these to guarantee rule coverage
and increase rule compliance scores without relying on LLMs.
"""

from __future__ import annotations

from typing import List, Dict, Callable, Optional
import re
import random

# Accept human-readable rule descriptions from validator module
try:
    # Maps internal rule name -> human-readable description
    from MIID.validator.rule_extractor import RULE_DESCRIPTIONS as _RULE_DESCRIPTIONS
except Exception:
    _RULE_DESCRIPTIONS = {}

# Build inverse map: description (lowercased) -> internal rule name
_DESC_TO_RULE: Dict[str, str] = {
    desc.lower(): name for name, desc in _RULE_DESCRIPTIONS.items()
}


def _split_parts(name: str) -> List[str]:
    return [p for p in name.split() if p]


def detect_script(text: str) -> str:
    # Simple Unicode block detection; default latin
    if all(ord(ch) < 128 for ch in text):
        return "latin"
    ranges = [
        ("arabic", [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
        ("cyrillic", [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
        ("cjk", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
    ]
    for script, blocks in ranges:
        for start, end in blocks:
            if any(start <= ord(ch) <= end for ch in text):
                return script
    return "latin"


def _make_rng(name: str, miner_salt: int = 0, batch_salt: int = 0) -> random.Random:
    seed = (hash(name) & 0xFFFFFFFF) ^ (miner_salt * 10007) ^ (batch_salt * 97)
    return random.Random(seed)


# ----------------------- Generators per rule -----------------------

def gen_replace_spaces_with_random_special_characters(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    specials = ["-", "_", ".", "/", "|"]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(specials)
    out: List[str] = []
    for i in range(min(n, len(specials))):
        s = specials[i]
        out.append(s.join(parts))
    return out


def gen_replace_double_letters_with_single_letter(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    out: List[str] = []
    idx = [i for i in range(len(name) - 1) if name[i].isalpha() and name[i] == name[i + 1]]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx:
        if name[i].isalpha() and name[i] == name[i + 1]:
            out.append(name[:i] + name[i] + name[i + 2 :])
            if len(out) >= n:
                break
    return out


def gen_replace_random_vowel_with_random_vowel(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    indices = [i for i, ch in enumerate(name) if ch in vowels]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    repl = {"a":"e","e":"a","i":"y","o":"u","u":"o","A":"E","E":"A","I":"Y","O":"U","U":"O"}
    for i in indices[:n]:
        ch = name[i]
        r = repl.get(ch, ch)
        if r != ch:
            out.append(name[:i] + r + name[i + 1 :])
    return out


def gen_replace_random_consonant_with_random_consonant(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    indices = [i for i, ch in enumerate(name) if ch.isalpha() and ch not in vowels]
    keyboard = {
        "q":"w","w":"e","e":"r","r":"t","t":"y","y":"u","u":"i","i":"o","o":"p",
        "a":"s","s":"d","d":"f","f":"g","g":"h","h":"j","j":"k","k":"l",
        "z":"x","x":"c","c":"v","v":"b","b":"n","n":"m"
    }
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices[:n]:
        ch = name[i]
        low = ch.lower()
        rep = keyboard.get(low)
        if rep:
            rep = rep.upper() if ch.isupper() else rep
            out.append(name[:i] + rep + name[i + 1 :])
    return out


def gen_replace_random_special_character_with_random_special_character(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    specials = "!@#$%^&*()_+-=[]{}|;:,.<>?/"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch in specials]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    alt = ["-", "_", ".", ",", ":"]
    for k, i in enumerate(idx[:n]):
        rep = alt[k % len(alt)]
        if rep != name[i]:
            out.append(name[:i] + rep + name[i + 1 :])
    return out


def gen_swap_random_letter(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    out: List[str] = []
    indices = list(range(len(name) - 1))
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices:
        if name[i] != name[i + 1]:
            out.append(name[:i] + name[i + 1] + name[i] + name[i + 2 :])
            if len(out) >= n:
                break
    return out


def gen_swap_adjacent_consonants(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    def is_cons(c: str) -> bool:
        return c.isalpha() and c.lower() not in "aeiou"
    out: List[str] = []
    indices = list(range(len(name) - 1))
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices:
        if is_cons(name[i]) and is_cons(name[i + 1]) and name[i].lower() != name[i + 1].lower():
            out.append(name[:i] + name[i + 1] + name[i] + name[i + 2 :])
            if len(out) >= n:
                break
    return out


def gen_swap_adjacent_syllables(name: str, n: int = 2, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    # Heuristic: swap between space-separated parts
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    out = [" ".join(parts[1:] + parts[:1])]
    return out[:n]


def gen_delete_random_letter(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    out: List[str] = []
    indices = list(range(len(name)))
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices:
        out.append(name[:i] + name[i + 1 :])
        if len(out) >= n:
            break
    return out


def gen_remove_random_vowel(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch in vowels]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx[:n]:
        out.append(name[:i] + name[i + 1 :])
    return out


def gen_remove_random_consonant(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch.isalpha() and ch not in vowels]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx[:n]:
        out.append(name[:i] + name[i + 1 :])
    return out


def gen_remove_random_special_character(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    specials = "!@#$%^&*()_+-=[]{}|;:,.<>?/"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch in specials]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx[:n]:
        out.append(name[:i] + name[i + 1 :])
    return out


def gen_remove_title(name: str, n: int = 2, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    titles = ["Mr.", "Mrs.", "Ms.", "Mr", "Mrs", "Ms", "Miss", "Dr.", "Dr",
              "Prof.", "Prof", "Sir", "Lady", "Lord", "Dame", "Rev.", "Hon.", "Capt.", "Col.", "Lt.", "Sgt.", "Maj."]
    for t in titles:
        t2 = t + " "
        if name.startswith(t2):
            return [name[len(t2):]]
    return []


def gen_remove_all_spaces(name: str, n: int = 1, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    if " " not in name:
        return []
    return [name.replace(" ", "")]


def gen_duplicate_random_letter_as_double_letter(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    out: List[str] = []
    indices = [i for i, ch in enumerate(name) if ch.isalpha()]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices:
        ch = name[i]
        out.append(name[:i] + ch + name[i:])
        if len(out) >= n:
            break
    return out


def gen_insert_random_letter(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    out: List[str] = []
    rng = _make_rng(name, miner_salt, batch_salt)
    for i in range(min(n, max(1, len(name)))):
        ch = letters[rng.randrange(len(letters))]
        pos = rng.randrange(len(name) + 1)
        out.append(name[:pos] + ch + name[pos:])
    return out


def gen_add_random_leading_title(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    titles = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(titles)
    out = [(t + " " + name) for t in titles[:n]]
    return out


def gen_add_random_trailing_title(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    suffixes = ["Jr.", "Sr.", "III", "PhD"]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(suffixes)
    out = [(name + " " + s) for s in suffixes[:n]]
    return out


def gen_shorten_name_to_initials(name: str, n: int = 1, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    initials = ".".join(p[0] for p in parts) + "."
    return [initials]


def gen_name_parts_permutations(name: str, n: int = 2, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    out = [" ".join(parts[::-1])]
    return out[:n]


def gen_initial_only_first_name(name: str, n: int = 1, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    return [parts[0][0] + ". " + " ".join(parts[1:])]


def gen_shorten_name_to_abbreviations(name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    parts = _split_parts(name)
    if not parts:
        return []
    out: List[str] = []
    # For first part, use strong prefix abbreviation to keep similarity high
    first = parts[0]
    prefix_len = 5 if len(first) >= 8 else (4 if len(first) >= 6 else 3)
    out.append((first[:prefix_len] + ". "+ " ".join(parts[1:])).strip())
    if len(parts) >= 2:
        # Also produce shorter prefix variant
        short_len = 3 if prefix_len > 3 else 2
        out.append((first[:short_len] + ". "+ " ".join(parts[1:])).strip())
    return list(dict.fromkeys([v for v in out if v and v != name]))[:n]


# Map rule name to generator function
RULE_GENERATORS: Dict[str, Callable[..., List[str]]] = {
    "replace_spaces_with_random_special_characters": gen_replace_spaces_with_random_special_characters,
    "replace_double_letters_with_single_letter": gen_replace_double_letters_with_single_letter,
    "replace_random_vowel_with_random_vowel": gen_replace_random_vowel_with_random_vowel,
    "replace_random_consonant_with_random_consonant": gen_replace_random_consonant_with_random_consonant,
    "replace_random_special_character_with_random_special_character": gen_replace_random_special_character_with_random_special_character,
    "swap_random_letter": gen_swap_random_letter,
    "swap_adjacent_consonants": gen_swap_adjacent_consonants,
    "swap_adjacent_syllables": gen_swap_adjacent_syllables,
    "delete_random_letter": gen_delete_random_letter,
    "remove_random_vowel": gen_remove_random_vowel,
    "remove_random_consonant": gen_remove_random_consonant,
    "remove_random_special_character": gen_remove_random_special_character,
    "remove_title": gen_remove_title,
    "remove_all_spaces": gen_remove_all_spaces,
    "duplicate_random_letter_as_double_letter": gen_duplicate_random_letter_as_double_letter,
    "insert_random_letter": gen_insert_random_letter,
    "add_random_leading_title": gen_add_random_leading_title,
    "add_random_trailing_title": gen_add_random_trailing_title,
    "shorten_name_to_initials": gen_shorten_name_to_initials,
    "name_parts_permutations": gen_name_parts_permutations,
    "initial_only_first_name": gen_initial_only_first_name,
    "shorten_name_to_abbreviations": gen_shorten_name_to_abbreviations,
}


def generate_by_rule(original: str, rule: str, per_rule_count: int = 3, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    gen = RULE_GENERATORS.get(rule)
    if not gen:
        return []
    try:
        vars = gen(original, per_rule_count, miner_salt=miner_salt, batch_salt=batch_salt)
        # Dedup and drop identical to original
        seen = set()
        out: List[str] = []
        for v in vars:
            if v and v != original and v not in seen:
                seen.add(v)
                out.append(v)
        return out
    except Exception:
        return []


def _generate_for_rule_names(original: str, rule_names: List[str], per_rule_count: int = 2, total_limit: int | None = None, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    out: List[str] = []
    for r in rule_names:
        out.extend(generate_by_rule(original, r, per_rule_count, miner_salt=miner_salt, batch_salt=batch_salt))
        if total_limit and len(out) >= total_limit:
            break
    # Dedup while preserving order
    seen = set()
    ordered: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered[: total_limit or len(ordered)]


def _resolve_rule_descriptions_to_names(rule_descriptions: List[str]) -> List[str]:
    """Resolve human-readable rule descriptions to internal rule names.

    Unknown descriptions are ignored. Matching is case-insensitive and trims whitespace.
    """
    names: List[str] = []
    for desc in rule_descriptions:
        if not desc:
            continue
        key = desc.strip().lower()
        name = _DESC_TO_RULE.get(key)
        if name:
            names.append(name)
    return names


def generate_for_rule_descriptions(original: str, rule_descriptions: List[str], per_rule_count: int = 2, total_limit: int | None = None, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    """Generate variations using a list of human-readable rule descriptions.

    Args:
        original: Seed name
        rule_descriptions: List of descriptions from RULE_DESCRIPTIONS values
        per_rule_count: Variations to attempt per rule
        total_limit: Optional cap on total variations returned
    """
    rule_names = _resolve_rule_descriptions_to_names(rule_descriptions)
    return _generate_for_rule_names(
        original,
        rule_names,
        per_rule_count,
        total_limit,
        miner_salt=miner_salt,
        batch_salt=batch_salt,
    )


def generate_for_rules(original: str, rules: List[str], per_rule_count: int = 2, total_limit: int | None = None, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    """Backward-compatible entry that now accepts rule DESCRIPTIONS only.

    If inputs are not recognized as internal rule names, they are treated
    as human-readable descriptions and resolved to internal names.
    """
    # If every entry is a known internal rule name, use directly
    if rules and all(r in RULE_GENERATORS for r in rules):
        return _generate_for_rule_names(
            original,
            rules,
            per_rule_count,
            total_limit,
            miner_salt=miner_salt,
            batch_salt=batch_salt,
        )
    # Otherwise, assume descriptions
    rule_names = _resolve_rule_descriptions_to_names(rules)
    return _generate_for_rule_names(
        original,
        rule_names,
        per_rule_count,
        total_limit,
        miner_salt=miner_salt,
        batch_salt=batch_salt,
    )


def generate_all_rules(original: str, per_rule_count: int = 2, total_limit: int | None = None, *, miner_salt: int = 0, batch_salt: int = 0) -> List[str]:
    return generate_for_rules(original, list(RULE_GENERATORS.keys()), per_rule_count, total_limit, miner_salt=miner_salt, batch_salt=batch_salt)


if __name__ == "__main__":
    name = "nicholas berg"
    rules = list(RULE_GENERATORS.keys())
    print(rules)
    print("Original:", name)
    for r in rules:
        print(f"\n{r}:")
        print(generate_by_rule(name, r, 2))


