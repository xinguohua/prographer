"""
ChatAnywhere  (OpenAI Chat Completions ) . 

most usage: 
- connectuse chatanywhere_summarize(text, api_key, endpoint)
- oruse make_chatanywhere_summarizer(api_key, endpoint) build1canuse's function
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import requests


def chat_completions_endpoint(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("LLM base_url is empty")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def recalculate_llm_cost(records: List[Dict], pricing_by_model: Dict[str, Dict]) -> Dict:
    """Recompute hosted token cost from recorded usage and explicit pricing."""
    total = 0.0
    priced_calls = 0
    unpriced_calls = 0
    for record in records:
        model = str(record.get("model", ""))
        pricing = pricing_by_model.get(model, {})
        input_rate = pricing.get("input_per_million_usd")
        output_rate = pricing.get("output_per_million_usd")
        input_tokens = record.get("input_tokens")
        output_tokens = record.get("output_tokens")
        if None in (input_rate, output_rate, input_tokens, output_tokens):
            unpriced_calls += 1
            continue
        total += (
            float(input_tokens) * float(input_rate)
            + float(output_tokens) * float(output_rate)
        ) / 1_000_000.0
        priced_calls += 1
    return {
        "cost_usd": total,
        "priced_calls": priced_calls,
        "unpriced_calls": unpriced_calls,
        "fully_recalculable": unpriced_calls == 0,
    }


def summarize_llm_calls(records: List[Dict], pricing_by_model: Dict[str, Dict]) -> Dict:
    by_stage = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0})
    missing_usage = 0
    for record in records:
        row = by_stage[str(record.get("stage", "unknown"))]
        row["calls"] += 1
        row["latency_seconds"] += float(record.get("wall_latency_seconds", 0.0) or 0.0)
        if record.get("input_tokens") is None or record.get("output_tokens") is None:
            missing_usage += 1
        else:
            row["input_tokens"] += int(record["input_tokens"])
            row["output_tokens"] += int(record["output_tokens"])
    return {
        "calls": len(records),
        "successful_calls": sum(record.get("status") == "ok" for record in records),
        "failed_calls": sum(record.get("status") != "ok" for record in records),
        "api_retries": sum(int(record.get("api_retries", 0) or 0) for record in records),
        "missing_provider_usage_calls": missing_usage,
        "wall_latency_seconds": sum(
            float(record.get("wall_latency_seconds", 0.0) or 0.0) for record in records
        ),
        "by_stage": dict(by_stage),
        "cost": recalculate_llm_cost(records, pricing_by_model),
    }


class TrackedOpenAICompatibleLLM:
    """Per-model OpenAI-compatible routing with auditable call telemetry."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        max_tokens: int,
        stop,
        timeout: float = 60.0,
        max_api_retries: int = 0,
        context_window: int = 4096,
    ):
        self.provider = str(provider)
        self.endpoint = chat_completions_endpoint(base_url)
        self.api_key = str(api_key)
        self.model = str(model)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = top_k
        self.max_tokens = int(max_tokens)
        self.stop = stop
        self.timeout = float(timeout)
        self.max_api_retries = max(0, int(max_api_retries))
        self.context_window = int(context_window)
        self._provider_tokenizer = None
        self._provider_tokenizer_checked = False
        self.records: List[Dict] = []

    def _estimate_input_tokens(self, prompt: str) -> tuple[int, str]:
        """Use an available model tokenizer, else a conservative char bound."""
        try:
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(prompt)) + 32, "tiktoken"
        except Exception:
            if not self._provider_tokenizer_checked:
                self._provider_tokenizer_checked = True
                try:
                    from transformers import AutoTokenizer
                    self._provider_tokenizer = AutoTokenizer.from_pretrained(
                        self.model, local_files_only=True, trust_remote_code=False,
                    )
                except Exception:
                    self._provider_tokenizer = None
            if self._provider_tokenizer is not None:
                return (
                    len(self._provider_tokenizer.encode(prompt, add_special_tokens=True)) + 32,
                    "provider_tokenizer",
                )
            return int(math.ceil(len(prompt) / 3.0)) + 32, "chars_per_token_3"

    def __call__(self, prompt: str, *, stage: str, attempt: int, metadata: Optional[Dict] = None) -> str:
        start = time.perf_counter()
        input_estimate, estimator = self._estimate_input_tokens(prompt)
        if input_estimate + self.max_tokens > self.context_window:
            error = ValueError(
                f"LLM context budget exceeded: estimated input {input_estimate} + "
                f"max output {self.max_tokens} > {self.context_window}"
            )
            record = {
                "stage": str(stage), "attempt": int(attempt),
                "retry": max(0, int(attempt) - 1), "provider": self.provider,
                "model": self.model, "endpoint": self.endpoint, "status": "error",
                "input_tokens": None, "output_tokens": None,
                "input_token_estimate": input_estimate,
                "input_token_estimator": estimator,
                "context_window": self.context_window,
                "max_output_tokens": self.max_tokens,
                "wall_latency_seconds": time.perf_counter() - start,
                "api_attempts": 0, "api_retries": 0, "usage": {},
                "input_characters": len(prompt), "output_characters": 0,
                "error": str(error),
            }
            record.update(dict(metadata or {}))
            self.records.append(record)
            raise error
        usage = {}
        result = ""
        error = None
        api_attempts = 0
        for api_retry in range(self.max_api_retries + 1):
            api_attempts += 1
            try:
                result, usage = chatanywhere_summarize(
                    text=prompt,
                    api_key=self.api_key,
                    endpoint=self.endpoint,
                    model=self.model,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=int(self.top_k) if self.top_k is not None else None,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                    timeout=self.timeout,
                    return_usage=True,
                )
                error = None
                break
            except Exception as exc:  # recorded and re-raised after bounded retries
                error = exc
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        record = {
            "stage": str(stage),
            "attempt": int(attempt),
            "retry": max(0, int(attempt) - 1),
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "status": "ok" if error is None else "error",
            "input_tokens": int(input_tokens) if input_tokens is not None else None,
            "output_tokens": int(output_tokens) if output_tokens is not None else None,
            "wall_latency_seconds": time.perf_counter() - start,
            "api_attempts": api_attempts,
            "api_retries": max(0, api_attempts - 1),
            "usage": dict(usage or {}),
            "input_characters": len(prompt),
            "input_token_estimate": input_estimate,
            "input_token_estimator": estimator,
            "context_window": self.context_window,
            "max_output_tokens": self.max_tokens,
            "output_characters": len(result),
            "error": str(error) if error is not None else None,
        }
        record.update(dict(metadata or {}))
        self.records.append(record)
        if error is not None:
            raise error
        return result


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
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    stop=None,
    timeout: float = 30.0,
    return_usage: bool = False,
) -> str:
    if not text or not text.strip():
        return ("", {}) if return_usage else ""
    body = {
        "model": model,
        "temperature": max(0.0, float(temperature)),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the supplied mutation instructions exactly and return only "
                    "the requested valid JSON, with no prose or Markdown."
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    if top_k is not None:
        body["top_k"] = int(top_k)
    if stop:
        body["stop"] = stop
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
    top_p: float = 0.95,
    top_k: Optional[int] = None,
    max_tokens: int = 1024,
    stop=None,
    timeout: float = 30.0,
) -> Callable[[str], str]:
    def _fn(text: str) -> str:
        return chatanywhere_summarize(
            text,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop,
            timeout=timeout,
        )
    return _fn
