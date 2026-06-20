from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def answer_strings(item: dict[str, Any]) -> list[str]:
    raw = item.get("answer_text") or item.get("answers") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(value).strip() for value in raw if str(value).strip()]


def edge_time_text(edge: dict[str, Any]) -> str:
    if edge.get("time"):
        return str(edge["time"])
    start = edge.get("start_year")
    end = edge.get("end_year")
    if start and end and str(start) != str(end):
        return f"{start}-{end}"
    if start:
        return str(start)
    if end:
        return str(end)
    return ""


def parse_time_sort_key(value: str) -> tuple[int, ...]:
    numbers = [int(part) for part in re.findall(r"\d+", str(value))]
    if not numbers:
        return (9999, 99, 99)
    if len(numbers) == 1:
        return (numbers[0], 0, 0)
    if len(numbers) == 2:
        return (numbers[0], numbers[1], 0)
    return tuple(numbers[:3])


def ordered_edges(item: dict[str, Any]) -> list[dict[str, Any]]:
    edges = list((item.get("subgraph") or {}).get("edges", []) or [])
    return sorted(
        edges,
        key=lambda edge: (
            parse_time_sort_key(edge_time_text(edge)),
            str(edge.get("source") or edge.get("source_name") or ""),
            str(edge.get("relation") or ""),
            str(edge.get("target") or edge.get("target_name") or ""),
        ),
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def get_references(item: dict[str, Any]) -> list[str]:
    refs = item.get("paraphrases") or []
    if isinstance(refs, str):
        refs = [refs]
    refs = [str(ref).strip() for ref in refs if str(ref).strip()]
    if refs:
        return refs
    question = str(item.get("question") or "").strip()
    return [question] if question else []


def first_reference(item: dict[str, Any]) -> str:
    refs = get_references(item)
    return refs[0] if refs else ""


def is_time_like(text: str) -> bool:
    value = str(text).strip()
    return bool(
        re.fullmatch(r"\d{4}", value)
        or re.fullmatch(r"\d{4}-\d{2}", value)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        or re.fullmatch(r"\d{4}\s*-\s*\d{4}", value)
    )


def guess_dataset_name(path: str | Path | None = None, sample: dict[str, Any] | None = None) -> str:
    if path:
        upper = str(path).upper()
        if "MULTITQ" in upper:
            return "MULTITQ"
        if "CRONQUESTION" in upper:
            return "CRONQUESTION"
    sample = sample or {}
    for edge in (sample.get("subgraph") or {}).get("edges", []) or []:
        if "source_name" in edge or "target_name" in edge or sample.get("qtype"):
            return "MULTITQ"
    return "CRONQUESTION"


def sanitize_eval_item(item: dict[str, Any]) -> dict[str, Any]:
    edges = []
    for edge in (item.get("subgraph") or {}).get("edges", []) or []:
        cleaned = {}
        for key in ("source", "target", "relation", "start_year", "end_year", "source_name", "target_name", "time"):
            if key in edge:
                cleaned[key] = edge[key]
        edges.append(cleaned)
    return {
        "question_id": item.get("question_id"),
        "subgraph": {"edges": edges, "edge_count": len(edges)},
        "answer_text": item.get("answer_text") or item.get("answers") or [],
    }


def get_question_type(item: dict[str, Any]) -> str:
    return str(item.get("qtype") or item.get("type") or "unknown")


def get_answer_type(item: dict[str, Any]) -> str:
    answers = answer_strings(item)
    if answers and all(is_time_like(answer) for answer in answers):
        return "time"
    return "entity"


def get_time_level(item: dict[str, Any]) -> str:
    explicit = item.get("time_level")
    if explicit:
        return str(explicit)
    values = list(item.get("times") or []) + answer_strings(item)
    values.extend(edge_time_text(edge) for edge in ordered_edges(item) if edge_time_text(edge))
    for value in values:
        text = str(value)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return "day"
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return "month"
        if re.fullmatch(r"\d{4}", text) or re.fullmatch(r"\d{4}\s*-\s*\d{4}", text):
            return "year"
    return "year"


def get_edge_count(item: dict[str, Any]) -> int:
    return int((item.get("subgraph") or {}).get("edge_count") or len((item.get("subgraph") or {}).get("edges", [])))


def sample_group_key(item: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        get_question_type(item),
        get_answer_type(item),
        get_time_level(item),
        min(get_edge_count(item), 5),
    )


def _allocate_group_sizes(group_sizes: dict[Any, int], total: int) -> dict[Any, int]:
    total_population = sum(group_sizes.values())
    if total_population <= total:
        return dict(group_sizes)

    allocation = {key: 0 for key in group_sizes}
    remainders = []
    remaining = total

    for key, size in group_sizes.items():
        quota = total * size / total_population
        taken = min(size, math.floor(quota))
        allocation[key] = taken
        remaining -= taken
        remainders.append((quota - taken, key))

    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocation[key] < group_sizes[key]:
            allocation[key] += 1
            remaining -= 1

    if remaining > 0:
        for key, size in sorted(group_sizes.items(), key=lambda pair: pair[1], reverse=True):
            while remaining > 0 and allocation[key] < size:
                allocation[key] += 1
                remaining -= 1
            if remaining <= 0:
                break

    return allocation


def stratified_sample(data: list[dict[str, Any]], sample_size: int, seed: int = 42) -> list[dict[str, Any]]:
    if sample_size >= len(data):
        return list(data)

    rng = random.Random(seed)
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        grouped[sample_group_key(item)].append(item)

    allocation = _allocate_group_sizes({key: len(items) for key, items in grouped.items()}, sample_size)

    sampled = []
    for key, items in grouped.items():
        take = allocation.get(key, 0)
        if take <= 0:
            continue
        sampled.extend(rng.sample(items, take))

    if len(sampled) < sample_size:
        chosen_ids = {id(item) for item in sampled}
        leftovers = [item for item in data if id(item) not in chosen_ids]
        sampled.extend(rng.sample(leftovers, sample_size - len(sampled)))

    rng.shuffle(sampled)
    return sampled[:sample_size]
