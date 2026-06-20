from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .core import build_operator_state
from .data import answer_strings, edge_time_text, get_references, guess_dataset_name, load_json, save_json, stratified_sample


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _contains_answer_leakage(question: str, answers: list[str]) -> bool:
    lowered = question.lower()
    return any(answer and answer.lower() in lowered for answer in answers)


def _graph_lines(raw_item: dict[str, Any]) -> list[str]:
    lines = []
    for idx, edge in enumerate((raw_item.get("subgraph") or {}).get("edges", []) or [], start=1):
        source = str(edge.get("source") or edge.get("source_name") or "").strip()
        relation = str(edge.get("relation") or "").strip()
        target = str(edge.get("target") or edge.get("target_name") or "").strip()
        time_text = edge_time_text(edge)
        line = f"{idx}. {source} -- {relation} --> {target}"
        if time_text:
            line += f" [{time_text}]"
        lines.append(line)
    return lines


def _judge_prompt(raw_item: dict[str, Any], generated_question: str, dataset: str) -> str:
    state = build_operator_state(raw_item, dataset=dataset)
    answers = answer_strings(raw_item)
    refs = get_references(raw_item)
    graph_lines = _graph_lines(raw_item)
    prompt = f"""You are evaluating a temporal KG question synthesis output.

Assess whether the generated question preserves the temporal intent of the input instance.
Judge semantics, not writing style. The reference question is only for orientation; the gold intent fields below are authoritative.
Return exactly one JSON object and nothing else.

[Evaluation Rubric]
1. operator_correct = 1 only if the generated question preserves the intended temporal operator family and direction.
   - Examples of operator errors: before vs after inversion, first vs last inversion, dropping an explicit time constraint, replacing a relative comparator with a generic time question.
2. slot_correct = 1 only if the generated question asks for the same slot.
   - source: asks for the source-side entity in the focus edge
   - target: asks for the target-side entity in the focus edge
   - time: asks for the event timestamp
3. comparator_correct = 1 only if the same comparator entity or comparator time is preserved when a comparator exists.
   - If no comparator entity/time exists in the gold intent, set comparator_correct = 1.
4. temporal_faithful = 1 only if the generated question is overall consistent with the gold temporal intent.
   - It should be 0 if any major semantic error appears in operator, asked slot, comparator, or temporal scope.
5. Ignore minor wording differences if the temporal intent is preserved.

[Gold Temporal Intent]
- operator: {state.operator}
- asked_slot: {state.ask_slot}
- answer_type: {state.answer_type}
- time_level: {state.time_level}
- anchor_entity: {state.anchor_entity or 'none'}
- comparator_entity: {state.comparator_entity or 'none'}
- comparator_time: {(state.comparator_time or state.constraint_value) or 'none'}
- focus_edge: {state.focus_summary}
- answers: {answers}

[Graph Facts]
{chr(10).join(graph_lines)}

[Reference Question]
{refs[0] if refs else 'N/A'}

[Generated Question]
{generated_question}

Decide these fields using the rubric above:
- operator_correct: 0 or 1
- slot_correct: 0 or 1
- comparator_correct: 0 or 1
- temporal_faithful: 0 or 1
- brief_reason: one short sentence that names the main issue or says it is faithful

Output schema:
{{"operator_correct":0 or 1,"slot_correct":0 or 1,"comparator_correct":0 or 1,"temporal_faithful":0 or 1,"brief_reason":"..."}}"""
    return prompt


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_one(
    raw_item: dict[str, Any],
    result_item: dict[str, Any],
    dataset: str,
    client,
    call_text_generation_fn,
    model: str,
    api_mode: str,
    reasoning_effort: str | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    question = _normalize_question(result_item.get("generated_question"))
    answers = answer_strings(raw_item)
    leakage = _contains_answer_leakage(question, answers)
    prompt = _judge_prompt(raw_item, question, dataset=dataset)
    raw_text = call_text_generation_fn(
        prompt,
        model=model,
        client=client,
        api_mode=api_mode,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )
    parsed = _parse_json_object(raw_text)
    return {
        "question_id": raw_item.get("question_id"),
        "generated_question": question,
        "operator_correct": int(parsed.get("operator_correct", 0)),
        "slot_correct": int(parsed.get("slot_correct", 0)),
        "comparator_correct": int(parsed.get("comparator_correct", 0)),
        "temporal_faithful": int(parsed.get("temporal_faithful", 0)),
        "answer_leakage": int(leakage),
        "brief_reason": str(parsed.get("brief_reason", "")).strip(),
        "judge_raw": raw_text.strip(),
    }


def _build_result_index(results: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {item.get("question_id"): item for item in results}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}
    return {
        "num_samples": n,
        "operator_accuracy": sum(row["operator_correct"] for row in rows) / n,
        "slot_accuracy": sum(row["slot_correct"] for row in rows) / n,
        "comparator_accuracy": sum(row["comparator_correct"] for row in rows) / n,
        "temporal_faithfulness": sum(row["temporal_faithful"] for row in rows) / n,
        "answer_leakage_rate": sum(row["answer_leakage"] for row in rows) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge temporal faithfulness evaluation for ChronoSynth outputs")
    parser.add_argument("--input_data", required=True, help="Raw evaluation split JSON")
    parser.add_argument("--results", required=True, help="Generation results JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    parser.add_argument("--sample_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_mode", choices=["responses", "chat", "auto"], default="responses")
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()
    from .api import build_client, call_text_generation

    raw_data = load_json(args.input_data)
    results = load_json(args.results)
    dataset = guess_dataset_name(path=args.input_data, sample=raw_data[0] if raw_data else None)
    result_index = _build_result_index(results)

    matched = []
    for item in raw_data:
        qid = item.get("question_id")
        if qid in result_index:
            merged = dict(item)
            merged["_result"] = result_index[qid]
            merged["_operator"] = build_operator_state(item, dataset=dataset).operator
            matched.append(merged)
    if args.sample_size and args.sample_size < len(matched):
        grouped = {}
        for item in matched:
            grouped.setdefault(item["_operator"], []).append(item)
        sampled = []
        per_group = max(1, args.sample_size // max(1, len(grouped)))
        for _, items in grouped.items():
            sampled.extend(stratified_sample(items, min(len(items), per_group), seed=args.seed))
        if len(sampled) < args.sample_size:
            remaining_ids = {item.get("question_id") for item in sampled}
            remainder = [item for item in matched if item.get("question_id") not in remaining_ids]
            sampled.extend(stratified_sample(remainder, min(len(remainder), args.sample_size - len(sampled)), seed=args.seed + 1))
        matched = sampled[: args.sample_size]

    client = build_client(api_key=args.api_key, base_url=args.base_url, model=args.judge_model)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                judge_one,
                item,
                item["_result"],
                dataset,
                client,
                call_text_generation,
                args.judge_model,
                args.api_mode,
                args.reasoning_effort,
                args.enable_thinking,
            ): item
            for item in matched
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Temporal faithfulness judge"):
            item = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "question_id": item.get("question_id"),
                        "generated_question": _normalize_question(item["_result"].get("generated_question")),
                        "operator_correct": 0,
                        "slot_correct": 0,
                        "comparator_correct": 0,
                        "temporal_faithful": 0,
                        "answer_leakage": int(_contains_answer_leakage(_normalize_question(item["_result"].get("generated_question")), answer_strings(item))),
                        "brief_reason": f"judge_error: {exc}",
                        "judge_raw": "",
                    }
                )

    payload = {
        "input_data": str(Path(args.input_data)),
        "results": str(Path(args.results)),
        "judge_model": args.judge_model,
        "dataset": dataset,
        "summary": _summary(rows),
        "details": sorted(rows, key=lambda row: str(row.get("question_id"))),
    }
    save_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
