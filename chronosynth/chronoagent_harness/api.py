from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_api import build_client as build_openai_compatible_client, generate_text


def call_text_generation(
    prompt: str,
    model: str = "gpt-4o-mini",
    client: Optional[OpenAI] = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_mode: str = "responses",
    retries: int = 3,
    sleep_sec: float = 1.5,
    reasoning_effort: str | None = None,
    enable_thinking: bool = False,
) -> str:
    client = client or build_client(api_key=api_key, base_url=base_url, model=model)
    return generate_text(
        prompt=prompt,
        model=model,
        client=client,
        api_mode=api_mode,
        retries=retries,
        sleep_sec=sleep_sec,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )


def build_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> OpenAI:
    return build_openai_compatible_client(api_key=api_key, base_url=base_url, model=model)
