"""
Rule-based name variation generators aligned with MIID.validator.rule_evaluator RULE_EVALUATORS.

This module provides small, deterministic generators that produce variations intended
to satisfy each validator rule heuristic. Use these to guarantee rule coverage
and increase rule compliance scores without relying on LLMs.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List

import bittensor as bt
from numpy import promote_types

from extensions import input_parse
from MIID.validator.rule_evaluator import (
    is_adjacent_consonants_swapped,
    is_all_spaces_removed,
    is_consonant_removed,
    is_consonant_replaced,
    is_double_letter_replaced,
    is_first_name_initial,
    is_initials_only,
    is_letter_duplicated,
    is_letter_removed,
    is_letters_swapped,
    is_name_abbreviated,
    is_name_parts_permutation,
    is_random_letter_inserted,
    is_random_special_removed,
    is_space_replaced_with_special_chars,
    is_special_character_replaced,
    is_suffix_added,
    is_title_added,
    is_title_removed,
    is_vowel_removed,
    is_vowel_replaced,
)


def _split_parts(name: str) -> List[str]:
    return [p for p in name.split() if p]


def detect_script(text: str) -> str:
    # Simple Unicode block detection; default latin
    if all(ord(ch) < 128 for ch in text):
        return "latin"
    ranges = [
        (
            "arabic",
            [
                (0x0600, 0x06FF),
                (0x0750, 0x077F),
                (0x08A0, 0x08FF),
                (0xFB50, 0xFDFF),
                (0xFE70, 0xFEFF),
            ],
        ),
        (
            "cyrillic",
            [(0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
        ),
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


def gen_replace_spaces_with_random_special_characters(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    specials = [
        "-",
        "_",
        ".",
        "/",
        "|",
        "\\",
        "~",
        "`",
        "'",
        '"',
        ":",
        ";",
        "=",
        "+",
        "*",
        "&",
        "%",
        "$",
        "#",
        "@",
        "!",
        "?",
        "^",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "<",
        ">",
    ]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(specials)
    out: List[str] = []
    for i in range(min(n, len(specials))):
        s = specials[i]
        out.append(s.join(parts))
    return out


def gen_replace_double_letters_with_single_letter(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    out: List[str] = []
    idx = [
        i for i in range(len(name) - 1) if name[i].isalpha() and name[i] == name[i + 1]
    ]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx:
        if name[i].isalpha() and name[i] == name[i + 1]:
            out.append(name[:i] + name[i + 1 :])
            if len(out) >= n:
                break
    return out


def gen_replace_random_vowel_with_random_vowel(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    indices = [i for i, ch in enumerate(name) if ch in vowels]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    repl = {
        "a": "e",
        "e": "a",
        "i": "o",
        "o": "i",
        "o": "u",
        "u": "o",
        "A": "E",
        "E": "A",
        "I": "O",
        "O": "I",
        "O": "U",
        "U": "O",
    }
    for i in indices[:n]:
        ch = name[i]
        r = repl.get(ch, ch)
        if r != ch:
            out.append(name[:i] + r + name[i + 1 :])
    return out


def gen_replace_random_consonant_with_random_consonant(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    indices = [i for i, ch in enumerate(name) if ch.isalpha() and ch not in vowels]
    consonant = [
        "b",
        "c",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices[:n]:
        ch = name[i]
        rep = rng.choice(consonant)
        while rep == ch.lower():
            rep = rng.choice(consonant)
        rep = rep.upper() if ch.isupper() else rep
        out.append(name[:i] + rep + name[i + 1 :])
    return out


def gen_replace_random_special_character_with_random_special_character(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    specials = [
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "_",
        "+",
        "-",
        "=",
        "[",
        "]",
        "{",
        "}",
        "|",
        ";",
        ":",
        ",",
        ".",
        "<",
        ">",
        "?",
        "/",
    ]
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


def gen_swap_random_letter(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
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


def gen_swap_adjacent_consonants(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    def is_cons(c: str) -> bool:
        return c.isalpha() and c.lower() not in "aeiou"

    out: List[str] = []
    indices = list(range(len(name) - 1))
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices:
        if (
            is_cons(name[i])
            and is_cons(name[i + 1])
            and name[i].lower() != name[i + 1].lower()
        ):
            out.append(name[:i] + name[i + 1] + name[i] + name[i + 2 :])
            if len(out) >= n:
                break
    return out


def gen_swap_adjacent_syllables(
    name: str, n: int = 2, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    # Heuristic: swap between space-separated parts
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    out = [" ".join(parts[1:] + parts[:1])]
    return out[:n]


def gen_delete_random_letter(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    out: List[str] = []
    indices = list(range(len(name)))
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(indices)
    for i in indices:
        out.append(name[:i] + name[i + 1 :])
        if len(out) >= n:
            break
    return out


def gen_remove_random_vowel(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch in vowels]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx[:n]:
        out.append(name[:i] + name[i + 1 :])
    return out


def gen_remove_random_consonant(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    vowels = "aeiouAEIOU"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch.isalpha() and ch not in vowels]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx[:n]:
        out.append(name[:i] + name[i + 1 :])
    return out


def gen_remove_random_special_character(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    specials = "!@#$%^&*()_+-=[]{}|;:,.<>?/"
    out: List[str] = []
    idx = [i for i, ch in enumerate(name) if ch in specials]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(idx)
    for i in idx[:n]:
        out.append(name[:i] + name[i + 1 :])
    return out


def gen_remove_title(
    name: str, n: int = 2, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    titles = [
        "Mr.",
        "Mrs.",
        "Ms.",
        "Mr",
        "Mrs",
        "Ms",
        "Miss",
        "Dr.",
        "Dr",
        "Prof.",
        "Prof",
        "Sir",
        "Lady",
        "Lord",
        "Dame",
        "Rev.",
        "Hon.",
        "Capt.",
        "Col.",
        "Lt.",
        "Sgt.",
        "Maj.",
    ]
    for t in titles:
        t2 = t + " "
        if name.startswith(t2):
            return [name[len(t2) :]]
    return []


def gen_remove_all_spaces(
    name: str, n: int = 1, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    if " " not in name:
        return []
    return [name.replace(" ", "")]


def gen_duplicate_random_letter_as_double_letter(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
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


def gen_insert_random_letter(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    out: List[str] = []
    rng = _make_rng(name, miner_salt, batch_salt)
    for i in range(min(n, max(1, len(name)))):
        ch = letters[rng.randrange(len(letters))]
        pos = rng.randrange(len(name) + 1)
        out.append(name[:pos] + ch + name[pos:])
    return out


def gen_add_random_leading_title(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    titles = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(titles)
    out = [(t + " " + name) for t in titles[:n]]
    return out


def gen_add_random_trailing_title(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    suffixes = ["Jr.", "Sr.", "III", "PhD"]
    rng = _make_rng(name, miner_salt, batch_salt)
    rng.shuffle(suffixes)
    out = [(name + " " + s) for s in suffixes[:n]]
    return out


def gen_shorten_name_to_initials(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []

    # Get first letter of each part
    first_letters = [p[0].upper() for p in parts]

    # Generate all three formats that is_initials_only accepts
    variations = []

    # Format 1: Dots without spaces (e.g., "J.S.")
    initials_no_space = ".".join(first_letters) + "."
    variations.append(initials_no_space)

    # Format 2: Dots with spaces (e.g., "J. S.")
    initials_with_space = ". ".join(first_letters) + "."
    variations.append(initials_with_space)

    # Format 3: No dots (e.g., "JS")
    initials_no_dots = "".join(first_letters)
    variations.append(initials_no_dots)

    return variations[:n]


def gen_name_parts_permutations(
    name: str, n: int = 2, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    out = [" ".join(parts[::-1])]
    return out[:n]


def gen_initial_only_first_name(
    name: str, n: int = 1, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    parts = _split_parts(name)
    if len(parts) < 2:
        return []
    return [
        parts[0][0] + ". " + " ".join(parts[1:]),
        parts[0][0] + "." + " ".join(parts[1:]),
    ]


def gen_shorten_name_to_abbreviations(
    name: str, n: int = 3, *, miner_salt: int = 0, batch_salt: int = 0
) -> List[str]:
    parts = _split_parts(name)
    if not parts:
        return []

    out: set[str] = set()

    rng = _make_rng(name, miner_salt, batch_salt)
    shortens = [
        [part[:idx] for idx in range(1, len(part))] if len(part) > 1 else part
        for part in parts
    ]
    for _ in range(n):
        var_name = ""
        for part in shortens:
            if not part:
                continue
            var_name += f" {rng.choice(part)}"
        out.add(var_name.strip())

    return list(out)


# Map rule name to (generator function, check function)
RULE_GENERATORS: Dict[str, tuple[Callable, Callable]] = {
    "replace_spaces_with_random_special_characters": (
        gen_replace_spaces_with_random_special_characters,
        is_space_replaced_with_special_chars,
    ),
    "replace_double_letters_with_single_letter": (
        gen_replace_double_letters_with_single_letter,
        is_double_letter_replaced,
    ),
    "replace_random_vowel_with_random_vowel": (
        gen_replace_random_vowel_with_random_vowel,
        is_vowel_replaced,
    ),
    "replace_random_consonant_with_random_consonant": (
        gen_replace_random_consonant_with_random_consonant,
        is_consonant_replaced,
    ),
    "replace_random_special_character_with_random_special_character": (
        gen_replace_random_special_character_with_random_special_character,
        is_special_character_replaced,
    ),
    "swap_random_letter": (gen_swap_random_letter, is_letters_swapped),
    "swap_adjacent_consonants": (
        gen_swap_adjacent_consonants,
        is_adjacent_consonants_swapped,
    ),
    "swap_adjacent_syllables": (gen_swap_random_letter, is_letters_swapped),
    "delete_random_letter": (gen_delete_random_letter, is_letter_removed),
    "remove_random_vowel": (gen_remove_random_vowel, is_vowel_removed),
    "remove_random_consonant": (gen_remove_random_consonant, is_consonant_removed),
    "remove_random_special_character": (
        gen_remove_random_special_character,
        is_random_special_removed,
    ),
    "remove_title": (gen_remove_title, is_title_removed),
    "remove_all_spaces": (gen_remove_all_spaces, is_all_spaces_removed),
    "duplicate_random_letter_as_double_letter": (
        gen_duplicate_random_letter_as_double_letter,
        is_letter_duplicated,
    ),
    "insert_random_letter": (gen_insert_random_letter, is_random_letter_inserted),
    "add_random_leading_title": (gen_add_random_leading_title, is_title_added),
    "add_random_trailing_title": (gen_add_random_trailing_title, is_suffix_added),
    "shorten_name_to_initials": (gen_shorten_name_to_initials, is_initials_only),
    "name_parts_permutations": (gen_name_parts_permutations, is_name_parts_permutation),
    "initial_only_first_name": (gen_initial_only_first_name, is_first_name_initial),
    "shorten_name_to_abbreviations": (
        gen_shorten_name_to_abbreviations,
        is_name_abbreviated,
    ),
}


def generate_variations(
    name: str,
    selected_rules: list[str],
    rule_expected_count: int,
    miner_salt: int = 0,
    batch_salt: int = 0,
) -> list[str]:

    out: list[str] = []
    variations: list[list[str]] = []
    variation_per_rule = max(1, math.ceil(rule_expected_count / len(selected_rules)))
    for rule in selected_rules:
        gen_func, val_func = RULE_GENERATORS[rule]
        vars = gen_func(
            name,
            n=variation_per_rule,
            miner_salt=miner_salt,
            batch_salt=batch_salt,
        )
        if not all([val_func(name, v) for v in vars]):
            raise ValueError(
                f"Generated variations do not pass validation for rule {rule}"
            )
        variations.append(vars)

    bt.logging.critical(
        f"Totol rule variation generated: {sum([len(vars) for vars in variations])}"
    )

    # assert expected_count <= sum([len(vars) for vars in variations])

    bt.logging.critical(f"rule variations for {variations}")
    while len(out) < rule_expected_count and sum([len(vars) for vars in variations]):
        for vars in variations:
            if vars:
                out.append(vars.pop())
    bt.logging.critical(f"rule variations for {name}: {out}")
    return out


def test_all_rules():
    print("=" * 80)
    print("COMPREHENSIVE RULE-BASED VARIATION TEST SUITE")
    print("=" * 80)

    # Test cases with different name types (all with at least first + last name)
    test_cases = [
        "Ashley Thompson",  # Standard two-part name
        "John Smith",  # Common two-part name
        "Mary Johnson",  # Female two-part name
        "John Paul Smith",  # Three-part name
        "Dr. Mary Watson",  # Name with title
        "James Wilson",  # Name with double letters
        "Jean Claude",  # Two-word first name style
        "Mr. Robert Smith",  # Name with leading title
    ]

    print("\n" + "=" * 80)
    print("TESTING ALL RULES WITH MULTIPLE TEST CASES")
    print("=" * 80)

    rules = list(RULE_GENERATORS.keys())

    for rule_name in rules:
        print(f"\n{'='*80}")
        print(f"RULE: {rule_name}")
        print(f"{'='*80}")

        gen_func, check_func = RULE_GENERATORS[rule_name]

        for original in test_cases:
            print(f"\n  Original: '{original}'")

            # Generate variations
            try:
                variations = gen_func(original, n=3, miner_salt=0, batch_salt=0)

                if variations:
                    print(f"  ✓ Generated {len(variations)} variation(s):")
                    for i, var in enumerate(variations, 1):
                        # Validate against the check function
                        is_valid = check_func(original, var)
                        status = "✓ PASS" if is_valid else "✗ FAIL"
                        print(f"    {i}. '{var}' [{status}]")
                else:
                    print(f"  ⚠ No variations generated (may not apply to this name)")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

    print("\n" + "=" * 80)
    print("DETAILED RULE VALIDATION TEST")
    print("=" * 80)

    # Specific test for each rule with expected behavior
    # All inputs have at least first name + last name
    detailed_tests = {
        "replace_spaces_with_random_special_characters": {
            "input": "John Smith",
            "expected_pattern": "No spaces, similar characters",
        },
        "replace_double_letters_with_single_letter": {
            "input": "Matthew Williams",
            "expected_pattern": "Double 't' becomes single",
        },
        "replace_random_vowel_with_random_vowel": {
            "input": "Ashley Thompson",
            "expected_pattern": "Vowel replaced with different vowel",
        },
        "replace_random_consonant_with_random_consonant": {
            "input": "Robert Thompson",
            "expected_pattern": "Consonant replaced with different consonant",
        },
        "swap_random_letter": {
            "input": "Ashley Johnson",
            "expected_pattern": "Adjacent letters swapped",
        },
        "swap_adjacent_consonants": {
            "input": "John Smith",
            "expected_pattern": "Adjacent consonants swapped",
        },
        "swap_adjacent_syllables": {
            "input": "John Smith",
            "expected_pattern": "Name parts reordered",
        },
        "delete_random_letter": {
            "input": "Thompson Williams",
            "expected_pattern": "One letter removed",
        },
        "remove_random_vowel": {
            "input": "Ashley Brown",
            "expected_pattern": "One vowel removed",
        },
        "remove_random_consonant": {
            "input": "Thompson Miller",
            "expected_pattern": "One consonant removed",
        },
        "remove_all_spaces": {
            "input": "John Smith",
            "expected_pattern": "All spaces removed",
        },
        "duplicate_random_letter_as_double_letter": {
            "input": "Ashley Davis",
            "expected_pattern": "Letter duplicated",
        },
        "insert_random_letter": {
            "input": "John Smith",
            "expected_pattern": "Random letter inserted",
        },
        "add_random_leading_title": {
            "input": "John Smith",
            "expected_pattern": "Title prefix added (Mr., Dr., etc.)",
        },
        "add_random_trailing_title": {
            "input": "John Smith",
            "expected_pattern": "Title suffix added (Jr., Sr., etc.)",
        },
        "shorten_name_to_initials": {
            "input": "John Smith",
            "expected_pattern": "Converted to initials",
        },
        "name_parts_permutations": {
            "input": "John Paul Smith",
            "expected_pattern": "Name parts reordered",
        },
        "initial_only_first_name": {
            "input": "John Smith",
            "expected_pattern": "First name as initial only",
        },
        "shorten_name_to_abbreviations": {
            "input": "Jonathan Williams",
            "expected_pattern": "Name abbreviated",
        },
    }

    print("\nRunning detailed validation tests...\n")

    passed = 0
    failed = 0
    skipped = 0

    for rule_name, test_info in detailed_tests.items():
        if rule_name in RULE_GENERATORS:
            gen_func, check_func = RULE_GENERATORS[rule_name]
            original = test_info["input"]

            try:
                variations = gen_func(original, n=3, miner_salt=0, batch_salt=0)

                if variations:
                    all_valid = all(check_func(original, var) for var in variations)
                    if all_valid:
                        print(f"✓ {rule_name}")
                        print(f"  Input: '{original}'")
                        print(f"  Output: {variations[:2]}")  # Show first 2
                        print(f"  Expected: {test_info['expected_pattern']}")
                        passed += 1
                    else:
                        print(f"✗ {rule_name} - VALIDATION FAILED")
                        print(f"  Input: '{original}'")
                        print(f"  Output: {variations}")
                        for var in variations:
                            is_valid = check_func(original, var)
                            print(f"    '{var}': {is_valid}")
                        failed += 1
                else:
                    print(f"⚠ {rule_name} - No variations (may not apply)")
                    skipped += 1

            except Exception as e:
                print(f"✗ {rule_name} - ERROR: {e}")
                failed += 1
        else:
            print(f"⚠ {rule_name} - Not found in RULE_GENERATORS")
            skipped += 1

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Skipped: {skipped}")
    print(f"Total: {passed + failed + skipped}")
    print("=" * 80)

    # Edge case testing - all with at least two words
    print("\n" + "=" * 80)
    print("EDGE CASE TESTING")
    print("=" * 80)

    edge_cases = {
        "All caps": "JOHN SMITH",
        "All lowercase": "john smith",
        "Mixed case": "JoHn SmItH",
        "Numbers in name": "John Smith2",
        "Unicode (Spanish)": "José García",
        "Unicode (French)": "François Martin",
        "Three words": "John Paul Jones",
        "With hyphen": "Mary-Jane Watson",
        "With apostrophe": "Patrick O'Brien",
        "Double letters": "Matthew Connelly",
    }

    print(
        "\nTesting edge cases with a sample rule (replace_random_vowel_with_random_vowel):\n"
    )

    sample_rule = "replace_random_vowel_with_random_vowel"
    if sample_rule in RULE_GENERATORS:
        gen_func, check_func = RULE_GENERATORS[sample_rule]

        for case_name, test_input in edge_cases.items():
            print(f"  {case_name}: '{test_input}'")
            try:
                variations = gen_func(test_input, n=2, miner_salt=0, batch_salt=0)
                if variations:
                    is_valid = [check_func(test_input, v) for v in variations]
                    print(f"    → {variations} [Valid: {is_valid}]")
                else:
                    print(f"    → No variations (expected for this case)")
            except Exception as e:
                print(f"    → ERROR: {e}")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


def test_rule_generator():
    name = "Vitaleksin"
    prompt = """
      Generate 15 variations of the name Vitaleksin, ensuring phonetic similarity: {'Medium': 0.5}, and orthographic similarity: {'Medium': 0.5}, and also include 30% of variations that follow: Additionally, generate variations that perform these transformations: Reorder name parts, Insert a random letter, and Duplicate a random letter. The following address is the seed country/city to generate address variations for: Lesotho. Generate unique real addresses within the specified country/city for each variation.  The following date of birth is the seed DOB to generate variations for: 1986-11-06.

[ADDITIONAL CONTEXT]:
- Address variations should be realistic addresses within the specified country/city
- DOB variations ATLEAST one in each category (±1 day, ±3 days, ±30 days, ±90 days, ±365 days, year+month only)
- Each variation must have a different, realistic address and DOB

    """
    selected_rules, _ = input_parse.find_variations_rules(prompt)

    rule_expected_count = input_parse.find_number_rule_variations(prompt)

    variations = generate_variations(
        name, selected_rules, rule_expected_count, miner_salt=1, batch_salt=1
    )
    print(variations)


if __name__ == "__main__":
    # test_all_rules()
    test_rule_generator()
    # for item in gen_shorten_name_to_abbreviations("Huaying L", n=20):
    # print(item, is_name_abbreviated("Huaying L",item))
    # for item in gen_replace_random_consonant_with_random_consonant(
    #     "Vinai PITCHAYOS", n=20
    # ):
    #     print(item, is_consonant_replaced("Vinai PITCHAYOS", item))
