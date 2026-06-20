from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "chronosynth"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from chronoagent_harness.api import build_client, call_text_generation
from chronoagent_harness.core import (
    build_memory_bank,
    build_memory_bank_cache_metadata,
    build_operator_state,
    build_prompt,
    load_memory_bank,
    save_memory_bank,
)
from chronoagent_harness.data import get_references, guess_dataset_name, load_json, sanitize_eval_item, save_json
from chronoagent_harness.metrics import evaluate, load as load_eval_pairs
from chronoagent_harness.progress import ProgressTracker


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: str) -> int:
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    # Lightweight estimate that does not require extra tokenizer dependencies.
    return max(1, math.ceil(len(stripped) / 4))


def clean_question(text: str) -> str:
    text = " ".join(str(text).strip().split())
    if not text:
        return ""
    if text.lower().startswith("question:"):
        text = text.split(":", 1)[1].strip()
    if "\n" in text:
        text = text.splitlines()[-1].strip()
    return text if text.endswith("?") else f"{text}?"


def prepare_memory_bank(train_path: str | Path, dataset: str, memory_bank_cache: str | Path | None):
    cache_path = Path(memory_bank_cache) if memory_bank_cache else None
    metadata = build_memory_bank_cache_metadata(train_path, dataset)
    cache_used = False
    cache_built = False
    build_started_at = time.perf_counter()
    if cache_path is not None:
        cached_bank = load_memory_bank(cache_path, expected_metadata=metadata)
        if cached_bank is not None:
            return cached_bank, {
                "cache_path": str(cache_path),
                "cache_used": True,
                "cache_built": False,
                "cache_exists": True,
                "build_seconds": time.perf_counter() - build_started_at,
                "cache_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
            }

    train_data = load_json(train_path)
    memory_bank = build_memory_bank(train_data, dataset=dataset)
    cache_built = cache_path is not None
    if cache_path is not None:
        save_memory_bank(memory_bank, cache_path, metadata=metadata)
    build_seconds = time.perf_counter() - build_started_at
    return memory_bank, {
        "cache_path": str(cache_path) if cache_path is not None else None,
        "cache_used": cache_used,
        "cache_built": cache_built,
        "cache_exists": bool(cache_path and cache_path.exists()),
        "build_seconds": build_seconds,
        "cache_size_bytes": cache_path.stat().st_size if cache_path and cache_path.exists() else 0,
    }


def generate_one(
    index: int,
    item: dict[str, Any],
    raw_item: dict[str, Any],
    memory_bank,
    dataset: str,
    client,
    model: str,
    ablation: str,
    api_mode: str,
    reasoning_effort: str | None,
    enable_thinking: bool,
    include_prompt: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    operator_state = build_operator_state(item, dataset=dataset)
    retrieval_mode = "full"
    use_drafts = True
    if ablation == "no_memory":
        retrieval_mode = "none"
    elif ablation == "relation_only":
        retrieval_mode = "relation_only"
    elif ablation == "no_drafts":
        use_drafts = False

    prompt = build_prompt(
        operator_state,
        memory_bank,
        retrieval_mode=retrieval_mode,
        use_drafts=use_drafts,
    )
    prompt_chars = len(prompt)
    prompt_token_est = estimate_tokens(prompt)

    started = time.perf_counter()
    started_at = utc_now()
    status = "ok"
    error_message = None
    try:
        raw_output = call_text_generation(
            prompt,
            model=model,
            client=client,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )
        generated_question = clean_question(raw_output)
    except Exception as exc:
        status = "error"
        error_message = str(exc)
        raw_output = f"ERROR: {exc}"
        generated_question = raw_output
    latency_seconds = time.perf_counter() - started
    output_chars = len(raw_output)
    output_token_est = estimate_tokens(raw_output)

    result = {
        "question_id": raw_item.get("question_id"),
        "generated_question": generated_question,
        "reference_questions": get_references(raw_item),
    }
    if include_prompt:
        result["prompt"] = prompt
        result["operator_state"] = {
            "operator": operator_state.operator,
            "ask_slot": operator_state.ask_slot,
            "focus_relation": operator_state.focus_relation,
            "focus_summary": operator_state.focus_summary,
            "anchor_entity": operator_state.anchor_entity,
            "comparator_entity": operator_state.comparator_entity,
            "comparator_time": operator_state.comparator_time or operator_state.constraint_value,
            "answer_type": operator_state.answer_type,
            "time_level": operator_state.time_level,
            "edge_count": operator_state.edge_count,
            "graph_shape": operator_state.graph_shape,
        }

    stats = {
        "sample_index": index,
        "question_id": raw_item.get("question_id"),
        "dataset": dataset,
        "status": status,
        "error_message": error_message,
        "latency_seconds": round(latency_seconds, 6),
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "prompt_chars": prompt_chars,
        "prompt_tokens_est": prompt_token_est,
        "output_chars": output_chars,
        "output_tokens_est": output_token_est,
        "total_tokens_est": prompt_token_est + output_token_est,
        "reference_count": len(result["reference_questions"]),
        "ablation": ablation,
        "model": model,
        "api_mode": api_mode,
        "include_prompt": include_prompt,
        "operator": operator_state.operator,
        "ask_slot": operator_state.ask_slot,
        "answer_type": operator_state.answer_type,
        "time_level": operator_state.time_level,
        "edge_count": operator_state.edge_count,
        "graph_shape": operator_state.graph_shape,
        "focus_relation": operator_state.focus_relation,
    }
    return result, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_stats(per_sample: list[dict[str, Any]], total_wall_seconds: float) -> dict[str, Any]:
    ok = [row for row in per_sample if row["status"] == "ok"]
    latencies = [row["latency_seconds"] for row in ok]
    prompt_tokens = [row["prompt_tokens_est"] for row in ok]
    output_tokens = [row["output_tokens_est"] for row in ok]
    total_tokens = [row["total_tokens_est"] for row in ok]
    return {
        "num_samples": len(per_sample),
        "num_success": len(ok),
        "num_error": len(per_sample) - len(ok),
        "success_rate": round(len(ok) / len(per_sample), 6) if per_sample else 0.0,
        "wall_time_seconds": round(total_wall_seconds, 6),
        "throughput_samples_per_min": round((len(ok) / total_wall_seconds) * 60.0, 6) if total_wall_seconds > 0 else 0.0,
        "avg_latency_seconds": round(statistics.mean(latencies), 6) if latencies else None,
        "p50_latency_seconds": round(statistics.median(latencies), 6) if latencies else None,
        "p95_latency_seconds": round(sorted(latencies)[max(0, min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1))], 6) if latencies else None,
        "avg_prompt_tokens_est": round(statistics.mean(prompt_tokens), 3) if prompt_tokens else None,
        "avg_output_tokens_est": round(statistics.mean(output_tokens), 3) if output_tokens else None,
        "avg_total_tokens_est": round(statistics.mean(total_tokens), 3) if total_tokens else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChronoHarness supplementary experiment with richer logging")
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--shard_size", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--include_prompt", action="store_true")
    parser.add_argument("--memory_bank_cache", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_mode", choices=["responses", "chat", "auto"], default="responses")
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--ablation", choices=["full", "no_memory", "relation_only", "no_drafts"], default="full")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = guess_dataset_name(path=args.input)
    eval_data = load_json(args.input)
    sanitized = [sanitize_eval_item(item) for item in eval_data]

    run_name = args.run_name or output_path.stem
    stem = output_path.with_suffix("")
    run_config_path = Path(f"{stem}.run_config.json")
    per_sample_path = Path(f"{stem}.per_sample.jsonl")
    summary_path = Path(f"{stem}.summary.json")
    metrics_path = Path(f"{stem}.metrics.json")

    run_config = {
        "run_name": run_name,
        "started_at_utc": utc_now(),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "dataset": dataset,
        "train_path": str(Path(args.train).resolve()),
        "input_path": str(Path(args.input).resolve()),
        "output_path": str(output_path.resolve()),
        "model": args.model,
        "workers": args.workers,
        "shard_size": args.shard_size,
        "save_every": args.save_every,
        "resume": not args.no_resume,
        "include_prompt": args.include_prompt,
        "memory_bank_cache": str(Path(args.memory_bank_cache).resolve()) if args.memory_bank_cache else None,
        "api_mode": args.api_mode,
        "reasoning_effort": args.reasoning_effort,
        "enable_thinking": args.enable_thinking,
        "ablation": args.ablation,
        "num_input_samples": len(eval_data),
        "python": sys.version,
        "env_summary": {
            "A_BASE_URL_set": bool(os.getenv("A_BASE_URL")),
            "A_API_KEY_set": bool(os.getenv("A_API_KEY")),
            "DEEPSEEK_BASE_URL_set": bool(os.getenv("DEEPSEEK_BASE_URL")),
            "DEEPSEEK_API_KEY_set": bool(os.getenv("DEEPSEEK_API_KEY")),
        },
    }
    save_json(run_config_path, run_config)

    memory_bank, cache_info = prepare_memory_bank(args.train, dataset, args.memory_bank_cache)
    client = build_client(api_key=args.api_key, base_url=args.base_url, model=args.model)
    tracker = ProgressTracker(
        output_path=output_path,
        eval_data=eval_data,
        shard_size=args.shard_size,
        save_every=args.save_every,
        resume=not args.no_resume,
        retry_errors=True,
    )

    pending_indices = tracker.pending_indices()
    per_sample_records: dict[int, dict[str, Any]] = {}
    if per_sample_path.exists() and not args.no_resume:
        with open(per_sample_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = row.get("sample_index")
                if isinstance(idx, int):
                    per_sample_records[idx] = row

    if not pending_indices:
        metrics = evaluate(load_eval_pairs(output_path), compute_cider=True)
        save_json(metrics_path, metrics)
        summary = summarize_stats([per_sample_records[idx] for idx in sorted(per_sample_records)], 0.0)
        summary.update({"cache_info": cache_info, "metrics_path": str(metrics_path), "run_config_path": str(run_config_path)})
        save_json(summary_path, summary)
        print(f"No pending samples. Existing output reused: {output_path}")
        return

    wall_started = time.perf_counter()
    desc = f"supplementary {dataset} {args.ablation}"
    pbar = tqdm(total=len(pending_indices), desc=desc)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(
                generate_one,
                idx,
                sanitized[idx],
                eval_data[idx],
                memory_bank,
                dataset,
                client,
                args.model,
                args.ablation,
                args.api_mode,
                args.reasoning_effort,
                args.enable_thinking,
                args.include_prompt,
            ): idx
            for idx in pending_indices
        }

        completed = 0
        success = 0
        error = 0
        latency_samples: list[float] = []
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                result, stats = future.result()
            except Exception as exc:
                result = {
                    "question_id": eval_data[idx].get("question_id"),
                    "generated_question": f"ERROR: {exc}",
                    "reference_questions": get_references(eval_data[idx]),
                }
                stats = {
                    "sample_index": idx,
                    "question_id": eval_data[idx].get("question_id"),
                    "dataset": dataset,
                    "status": "error",
                    "error_message": str(exc),
                    "latency_seconds": None,
                    "started_at_utc": None,
                    "finished_at_utc": utc_now(),
                    "prompt_chars": None,
                    "prompt_tokens_est": None,
                    "output_chars": None,
                    "output_tokens_est": None,
                    "total_tokens_est": None,
                    "reference_count": len(result["reference_questions"]),
                    "ablation": args.ablation,
                    "model": args.model,
                    "api_mode": args.api_mode,
                    "include_prompt": args.include_prompt,
                    "operator": None,
                    "ask_slot": None,
                    "answer_type": None,
                    "time_level": None,
                    "edge_count": None,
                    "graph_shape": None,
                    "focus_relation": None,
                }
            tracker.record(idx, result)
            per_sample_records[idx] = stats
            completed += 1
            if stats["status"] == "ok":
                success += 1
                latency_samples.append(stats["latency_seconds"])
            else:
                error += 1
            avg_latency = statistics.mean(latency_samples) if latency_samples else 0.0
            pbar.set_postfix(success=success, error=error, avg_lat=f"{avg_latency:.2f}s")
            pbar.update(1)
            if completed % args.save_every == 0:
                write_jsonl(per_sample_path, [per_sample_records[i] for i in sorted(per_sample_records)])
    pbar.close()
    tracker.finalize()
    write_jsonl(per_sample_path, [per_sample_records[i] for i in sorted(per_sample_records)])
    wall_seconds = time.perf_counter() - wall_started

    metrics = evaluate(load_eval_pairs(output_path), compute_cider=True)
    save_json(metrics_path, metrics)

    summary = summarize_stats([per_sample_records[idx] for idx in sorted(per_sample_records)], wall_seconds)
    summary.update(
        {
            "run_name": run_name,
            "finished_at_utc": utc_now(),
            "cache_info": cache_info,
            "metrics": metrics,
            "metrics_path": str(metrics_path),
            "run_config_path": str(run_config_path),
            "per_sample_path": str(per_sample_path),
            "output_path": str(output_path),
        }
    )
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
