"""
ChatAnywhere  (OpenAI Chat Completions ) . 

most usage: 
- connectuse chatanywhere_summarize(text, api_key, endpoint)
- oruse make_chatanywhere_summarizer(api_key, endpoint) build1canuse's function
"""
from __future__ import annotations

import json
from typing import Callable

import requests


def _default_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

# ------------------------------
# version: connectinput api_key and endpoint
# ------------------------------
def chatanywhere_summarize(
    text: str,
    *,
    api_key: str,
    endpoint: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    timeout: float = 30.0,
    return_usage: bool = False,
) -> str:
    if not text or not text.strip():
        return ("", {}) if return_usage else ""
    body = {
        "model": model,
        "temperature": max(0.0, float(temperature)),
        "messages": [
            {"role": "system", "content": "youis's total. "},
            {"role": "user", "content": text},
        ],
    }
    try:
        resp = requests.post(
            endpoint,
            headers=_default_headers(api_key),
            data=json.dumps(body),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = ((data.get("choices") or [{}])[0].get("message", {}) or {}).get("content", "")
        result = str(content).strip()
        if not result:
            raise ValueError("LLM returnemptycontent")
        if return_usage:
            usage = data.get("usage", {})
            return result, usage
        return result
    except Exception as ex:
        raise RuntimeError(f"LLM API usefailed: {ex}") from ex


def make_chatanywhere_summarizer(
    *,
    api_key: str,
    endpoint: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    timeout: float = 30.0,
) -> Callable[[str], str]:
    def _fn(text: str) -> str:
        return chatanywhere_summarize(
            text,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
    return _fn
