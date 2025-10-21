# coding=utf-8


def prepare_payload(
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
                    "content": f"""
                        You are a helpful assistant \n
                        Just returns a json object, without any additional explanation.
                        You must respond with JSON that matches this schema: {schema}
                    """,
                },
            )
            payload["messages"] = messages

    return payload
