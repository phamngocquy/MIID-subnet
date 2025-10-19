# coding=utf-8
import json
import hashlib
from openai import OpenAI
from pydantic import BaseModel

VALIDATOR_PROMT_TMPL = """
Generate {variation_count} variations of {name}, ensuring phonetic similarity ({phonetic_spec}) and orthographic similarity {orthographic_spec},
Approximately {rule_percentage} of the total {variation_count} variations should follow these rule-based transformations: (Additionally, generate variations that | Additionally, generate variations that perform these transformations): {rules}
"""


class ValidatorRequest(BaseModel):
    variation_count: int
    phonetic_spec: dict[str, float]
    orthographic_spec: dict[str, float]
    rule_percentage: int
    rules: str


def _prepare_payload(
    messages,
    json_format: bool = False,
    schema: dict | None = None,
    stream: bool = False,
):
    payload = {"model": "deepseek-chat", "messages": messages, "stream": stream}
    if json_format:
        payload["response_format"] = {"type": "json_object"}
        if schema:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"You must respond with JSON that matches this schema: {schema}",
                },
            )
            payload["messages"] = messages

    return payload


def parse_validator_request(request: str):
    client = OpenAI(
        api_key="",
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
        You are a helpful assistant \n
        Just returns a json object, without any additional explanation.
        The output must be a JSON object in a format like this: {"field_a": "value_a", ...}
    """
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    payload = _prepare_payload(messages, schema=ValidatorRequest.model_json_schema())
    response = client.chat.completions.create(**payload)
    data = json.loads(response.choices[0].message.content)
    return ValidatorRequest(**data)


if __name__ == "__main__":
    testcase = [
        {
            "promt": """
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
            "promt": """
Generate 14 variations of {name}, ensuring phonetic similarity and orthographic similarity. For phonetic similarity, include:
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
    ]

    for item in testcase:
        resp = parse_validator_request(item["promt"])
        assert (
            item["result"]
            == hashlib.md5(resp.model_dump_json().encode("utf-8")).hexdigest()
        )
