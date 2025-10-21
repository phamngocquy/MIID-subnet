# coding=utf-8

import dotenv

dotenv.load_dotenv()

import json
import math
import os

from openai import OpenAI
from pydantic import BaseModel

from extensions import utils

RULE_DESCRIPTIONS = {
    "replace_spaces_with_random_special_characters": "Replace spaces with special characters",
    "replace_double_letters_with_single_letter": "Replace double letters with a single letter",
    "replace_random_vowel_with_random_vowel": "Replace random vowels with different vowels",
    "replace_random_consonant_with_random_consonant": "Replace random consonants with different consonants",
    "swap_adjacent_consonants": "Swap adjacent consonants",
    "swap_adjacent_syllables": "Swap adjacent syllables",
    "swap_random_letter": "Swap random adjacent letters",
    "delete_random_letter": "Delete a random letter",
    "remove_random_vowel": "Remove a random vowel",
    "remove_random_consonant": "Remove a random consonant",
    "remove_all_spaces": "Remove all spaces",
    "duplicate_random_letter_as_double_letter": "Duplicate a random letter",
    "insert_random_letter": "Insert a random letter",
    "add_random_leading_title": "Add a title prefix (Mr., Dr., etc.)",
    "add_random_trailing_title": "Add a title suffix (Jr., PhD, etc.)",
    "initial_only_first_name": "Use first name initial with last name",
    "shorten_name_to_initials": "Convert name to initials",
    "shorten_name_to_abbreviations": "Abbreviate name parts",
    "name_parts_permutations": "Reorder name parts",
}


VALIDATOR_PROMT_TMPL = """
Generate {variation_count} variations of {name}, ensuring phonetic similarity ({phonetic_spec}) and orthographic similarity ({orthographic_spec}),
Approximately {rule_percentage} of the total {variation_count} variations should follow these rule-based transformations: (Additionally, generate variations that | Additionally, generate variations that perform these transformations): {rules}
"""


class ValidatorRequest(BaseModel):
    variation_count: int
    phonetic_spec: dict[str, float]
    orthographic_spec: dict[str, float]
    rule_percentage: int
    rules: str


def parse_validator_request(request: str):
    prompt = f"""
    I have a string template:
    {VALIDATOR_PROMT_TMPL}
    and a string output of this template:
    {request}
    Help me extract variable value form ouput string.
    Here is output example:
        {{
            "variation_count": 20,
            "phonetic_spec": {{"Light": 0.3, "Medium": 0.4, "Far": 0.3}},
            "orthographic_spec": {{"Light": 0.3, "Medium": 0.4, "Far": 0.3}},
            "rule_percentage": 30,
            "rules": ""Replace double letters with a single letter, Replace random consonants with different consonants""

        }}
    """

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    payload = utils.prepare_payload(
        messages, schema=ValidatorRequest.model_json_schema()
    )
    response = client.chat.completions.create(**payload)
    data = json.loads(response.choices[0].message.content)
    return ValidatorRequest(**data)


def find_variation_count(request: str) -> int:
    start_idx = None
    words = request.strip().lower().split()
    for idx, word in enumerate(words):
        if word == "generate":
            start_idx = idx
            break
    if start_idx is None:
        raise Exception("Cannot find 'generate' keyword")
    for idx in range(start_idx, len(words)):
        if words[idx].isdigit():
            return int(words[idx])
    raise Exception("Cannot parse variation count")


def find_variations_rules(prompt: str) -> tuple[list[str], list[str]]:
    """find_variations_rules.
    - find rules mentioned in the prompt
    - find percentage of rule-based variations if mentioned

    Args:
        prompt (str): prompt

    Returns:
        list:
    """
    rules_found = []
    rules_found_des = []
    for rule_key, rule_desc in RULE_DESCRIPTIONS.items():
        if rule_desc.lower() in prompt.lower():
            rules_found.append(rule_key.strip())
            rules_found_des.append(rule_desc.strip())
    return rules_found, rules_found_des


def find_number_rule_variations(prompt: str) -> int:
    """find_number_rule_variations.
    - find number of rule-based variations if mentioned

    Args:
        prompt (str): prompt
    """

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    total_variations = find_variation_count(prompt)
    _, rules_selected = find_variations_rules(prompt)

    prompt = f"""
    [TASK]
    Find the percentage of rule-based variations generate by transformations: {', '.join(rules_selected)} in the following prompt.
    [[MESSAGE]]
    {prompt}
    """

    print("Prompt to find rule-based percentage:", prompt)
    messages = [
        {"role": "user", "content": prompt},
    ]
    payload = utils.prepare_payload(
        messages, schema={"rule_based_percentage": "int"}, json_format=True
    )

    response = client.chat.completions.create(**payload)
    result = json.loads(response.choices[0].message.content)
    print("Rule-based percentage:", result["rule_based_percentage"])
    return math.ceil((total_variations * result["rule_based_percentage"]) // 100)


if __name__ == "__main__":
    testcase = [
        {
            "prompt": """
Generate 14 variations of {name}, ensuring phonetic similarity and orthographic similarity. For phonetic similarity, include:
- Medium variations (50%): Remove or change middle initial or last name suffix/prefix
For orthographic similarity, include:
- Light variations (30%): Add/Remove apostrophe in surname
- Medium variations (40%): Replace double letters with single letter
- Far variations (30%): Interchange or add/remove middle initial

Approximately 22% of the total 14 variations should follow these rule-based transformations: Additionally, generate variations that: Replace double letters with a single letter..
        """,
            "result": "f71a4a25483b26439a1b94300ef2d4b1",
        },
        {
            "prompt": """
Generate 1 variations of {name}, ensuring phonetic similarity and orthographic similarity. For phonetic similarity, include:
- Light variations (10%): Replace first name with similar-sounding nickname
- Medium variations (50%): Remove or change middle initial or last name suffix/prefix
- Far variations (40%): Change entire first and/or last name to a completely different but still similar-sounding entity
For orthographic similarity, include:
- Light variations (30%): Add/Remove apostrophe in surname
- Medium variations (40%): Replace double letters with single letter
- Far variations (30%): Interchange or add/remove middle initial

Approximately 22% of the total 14 variations should follow these rule-based transformations: Additionally, generate variations that perform these transformations: Replace double letters with a single letter, Replace random consonants with different consonants
            """,
            "result": "a702c6b037e1d4e828e5e4d00c2f869a",
        },
        {
            "prompt": """
            Generate 15 variations of the name Quy Pham, ensuring phonetic similarity: {'Medium': 0.5}, and orthographic similarity: {'Medium': 0.5}, and also include 30% of variations that follow: Additionally, generate variations that perform these transformations: Convert name to initials, Replace random vowels with different vowels, and Insert a random letter.. The following address is the seed country/city to generate address variations for: Lesotho. Generate unique real addresses within the specified country/city for each variation.  The following date of birth is the seed DOB to generate variations for: 1986-11-06.

[ADDITIONAL CONTEXT]:
- Address variations should be realistic addresses within the specified country/city
- DOB variations ATLEAST one in each category (±1 day, ±3 days, ±30 days, ±90 days, ±365 days, year+month only)
- Each variation must have a different, realistic address and DOB

            """,
            "result": "6f1ed002ab5595859014ebf0951522d9",
        },
    ]

    # for item in testcase:
    #     resp = parse_validator_request(item["promt"])
    #     assert (
    #         item["result"]
    #         == hashlib.md5(resp.model_dump_json().encode("utf-8")).hexdigest()
    #     )

    # for item in testcase:
    # print(parser_variation_count(item["promt"]))
    print(find_number_rule_variations(testcase[2]["prompt"]))
