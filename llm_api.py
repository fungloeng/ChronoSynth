from __future__ import annotations

import os
import time
import random
from typing import Any

from openai import OpenAI


def _pick_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def build_client(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> OpenAI:
    model_name = (model or "").strip().lower()
    is_deepseek = model_name.startswith("deepseek-")
    if base_url:
        resolved_base_url = base_url
    elif is_deepseek:
        resolved_base_url = _pick_env("DEEPSEEK_BASE_URL")
    else:
        resolved_base_url = _pick_env("A_BASE_URL", "OPENAI_BASE_URL", "API_BASE")
    if api_key:
        resolved_api_key = api_key
    elif is_deepseek:
        resolved_api_key = _pick_env("DEEPSEEK_API_KEY")
    else:
        resolved_api_key = _pick_env("A_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    if not resolved_api_key:
        if is_deepseek:
            raise RuntimeError(
                "DeepSeek API key is not set. Use --api_key or set DEEPSEEK_API_KEY."
            )
        raise RuntimeError(
            "API key is not set. Use --api_key or set A_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY."
        )
    timeout_sec = float(_pick_env("TKGQG_API_TIMEOUT", "OPENAI_TIMEOUT", "A_TIMEOUT") or "120")
    kwargs: dict[str, Any] = {"api_key": resolved_api_key, "timeout": timeout_sec}
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    return OpenAI(**kwargs)


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()

    output = getattr(response, "output", None) or []
    chunks: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if text_value:
                chunks.append(text_value)
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _call_responses_api(
    client: OpenAI,
    prompt: str,
    model: str,
) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        timeout=float(_pick_env("TKGQG_API_TIMEOUT", "OPENAI_TIMEOUT", "A_TIMEOUT") or "120"),
    )
    return extract_output_text(response)


def _call_chat_completions_api(
    client: OpenAI,
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    thinking_mode: str | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "timeout": float(_pick_env("TKGQG_API_TIMEOUT", "OPENAI_TIMEOUT", "A_TIMEOUT") or "120"),
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if thinking_mode in {"enabled", "disabled"}:
        kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}

    response = client.chat.completions.create(**kwargs)
    return extract_output_text(response)


def _is_model_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "model_not_found" in message or "no available channel for model" in message or "model not found" in message


def _is_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "too many requests" in message
        or "503" in message
        or "service unavailable" in message
        or "server busy" in message
        or "502" in message
        or "bad gateway" in message
        or "504" in message
        or "gateway timeout" in message
        or "connection reset" in message
        or "timeout" in message
    )


def generate_text(
    prompt: str,
    model: str,
    client: OpenAI | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_mode: str = "responses",
    retries: int = 3,
    sleep_sec: float = 1.5,
    system_prompt: str | None = None,
    reasoning_effort: str | None = None,
    enable_thinking: bool = False,
) -> str:
    client = client or build_client(api_key=api_key, base_url=base_url, model=model)
    mode = (api_mode or "responses").strip().lower()
    if mode not in {"responses", "chat", "auto"}:
        raise ValueError(f"Unsupported api_mode: {api_mode}")

    model_name = (model or "").strip().lower()
    force_no_thinking = model_name == "deepseek-v4-flash"
    effective_reasoning_effort = None if force_no_thinking else reasoning_effort
    if force_no_thinking:
        effective_thinking_mode = "disabled"
        if mode == "responses":
            mode = "chat"
    else:
        effective_thinking_mode = "enabled" if enable_thinking else None

    last_error = None
    for attempt in range(retries):
        try:
            if mode == "responses":
                return _call_responses_api(client, prompt=prompt, model=model).strip()
            if mode == "chat":
                return _call_chat_completions_api(
                    client,
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    reasoning_effort=effective_reasoning_effort,
                    thinking_mode=effective_thinking_mode,
                ).strip()

            try:
                return _call_chat_completions_api(
                    client,
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    reasoning_effort=effective_reasoning_effort,
                    thinking_mode=effective_thinking_mode,
                ).strip()
            except Exception:
                return _call_responses_api(client, prompt=prompt, model=model).strip()
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt + 1 < retries:
                if _is_retryable_error(exc):
                    base_sleep = min(60.0, 2.0 * (2 ** attempt))
                    time.sleep(base_sleep + random.uniform(0.0, 1.0))
                else:
                    time.sleep(sleep_sec * (attempt + 1))

    raise RuntimeError(f"API call failed after {retries} attempts: {last_error}")
