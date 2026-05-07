"""LLM adapters and JSON parsing utilities for TriSkill."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLM(Protocol):
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        ...


@dataclass
class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible chat completions client using stdlib only."""

    api_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: int = 120

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        api_url = os.getenv("TRISKILL_API_URL") or os.getenv("OPENAI_BASE_URL") or ""
        model = os.getenv("TRISKILL_MODEL") or os.getenv("OPENAI_MODEL") or ""
        api_key = os.getenv("TRISKILL_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        if not api_url or not model:
            raise ValueError("Set TRISKILL_API_URL and TRISKILL_MODEL to use OpenAI-compatible generation")
        return cls(api_url=api_url, model=model, api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url.rstrip("/") + "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {details[:500]}") from exc
        return str(body["choices"][0]["message"]["content"])


def extract_json_text(text: str) -> str | None:
    source = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", source, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", source, flags=re.DOTALL | re.IGNORECASE)
    if answer:
        source = answer.group(1).strip()
    for left, right in (("{", "}"), ("[", "]")):
        start = source.find(left)
        end = source.rfind(right)
        if start != -1 and end > start:
            return source[start:end + 1]
    return None


def parse_json_lenient(text: str) -> Any | None:
    candidates = [text, extract_json_text(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


def call_llm_json(llm: LLM, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> tuple[Any | None, str]:
    raw = llm.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens)
    return parse_json_lenient(raw), raw
