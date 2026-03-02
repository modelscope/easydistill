import copy
from typing import Optional, List, Dict
from openai import OpenAI


def call_llm_api(
    user_prompt: str,
    system_prompt: str,
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
):
    client = OpenAI(api_key=api_key, base_url=api_base)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        extra_body={
            "max_completion_tokens": max_tokens
        },
    )
    response_content = response.choices[0].message.content
    messages.append(
        {"role": "assistant", "content": response_content}
    )

    return messages


def call_llm_messages(
    messages: List[Dict[str, str]],
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
):
    client = OpenAI(api_key=api_key, base_url=api_base)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        extra_body={"max_completion_tokens": max_tokens}
    )
    response_content = response.choices[0].message.content
    messages.append(
        {"role": "assistant", "content": response_content}
    )

    return messages
