from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    idx = max(0, min(len(values) - 1, math.ceil(len(values) * q) - 1))
    return sorted(values)[idx]


def parse_name(stem: str) -> tuple[str, str, str]:
    if stem.startswith("chrono_full_"):
        rest = stem[len("chrono_full_"):]
        return "Chrono-full", "CRONQUESTION" if "CRONQUESTION" in rest else "MULTITQ", "demo100" if rest.endswith("demo100") else "test"
    if stem.startswith("chrono_no_memory_"):
        rest = stem[len("chrono_no_memory_"):]
        return "Chrono-no_memory", "CRONQUESTION" if "CRONQUESTION" in rest else "MULTITQ", "demo100" if rest.endswith("demo100") else "test"
    if stem.startswith("chrono_relation_only_"):
        rest = stem[len("chrono_relation_only_"):]
        return "Chrono-relation_only", "CRONQUESTION" if "CRONQUESTION" in rest else "MULTITQ", "demo100" if rest.endswith("demo100") else "test"
    if stem.startswith("kqg_cot_"):
        rest = stem[len("kqg_cot_"):]
        return "KQG-CoT", "CRONQUESTION" if "CRONQUESTION" in rest else "MULTITQ", "demo100" if rest.endswith("demo100") else "test"
    return stem, "", ""


def load_run(result_path: Path, per_sample_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = load_json(result_path)
    per_sample = load_jsonl(per_sample_path)
    by_index = {int(row["sample_index"]): row for row in per_sample}
    merged: list[dict[str, Any]] = []
    for item in results:
        sample_index = int(item["sample_index"])
        stats = by_index.get(sample_index)
        if not stats:
            continue
        if stats.get("status") != "ok":
            continue
        refs = item.get("reference_questions") or []
        if isinstance(refs, str):
            refs = [refs]
        hyp = str(item.get("generated_question", "")).strip()
        refs = [str(ref).strip() for ref in refs if str(ref).strip()]
        if not hyp or hyp.startswith("ERROR:") or not refs:
            continue
        merged.append(
            {
                "sample_index": sample_index,
                "question_id": item.get("question_id"),
                "generated_question": hyp,
                "reference_questions": refs,
                **stats,
            }
        )
    method, dataset, split = parse_name(result_path.stem)
    meta = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "result_path": str(result_path),
        "per_sample_path": str(per_sample_path),
    }
    return meta, merged


def build_group_rows(
    meta: dict[str, Any],
    rows: list[dict[str, Any]],
    group_keys: list[str],
    min_group_size: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_key in tqdm(group_keys, desc=f"{meta['method']} {meta['dataset']} groups", leave=False):
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            value = row.get(group_key)
            if value is None or value == "":
                value = "UNKNOWN"
            value = str(value)
            buckets.setdefault(value, []).append(row)
        for group_value, bucket in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(bucket) < min_group_size:
                continue
            pairs = [(entry["generated_question"], entry["reference_questions"]) for entry in bucket]
            metrics = evaluate(pairs, compute_cider=True, verbose=False)
            latencies = [float(entry["latency_seconds"]) for entry in bucket]
            total_tokens = [float(entry["total_tokens_est"]) for entry in bucket]
            prompt_tokens = [float(entry["prompt_tokens_est"]) for entry in bucket]
            out.append(
                {
                    "method": meta["method"],
                    "dataset": meta["dataset"],
                    "split": meta["split"],
                    "group_key": group_key,
                    "group_value": group_value,
                    "num_samples": len(bucket),
                    "BLEU-4": metrics.get("BLEU-4"),
                    "ROUGE-L": metrics.get("ROUGE-L"),
                    "CIDEr": metrics.get("CIDEr"),
                    "Distinct-1": metrics.get("Distinct-1"),
                    "Distinct-2": metrics.get("Distinct-2"),
                    "avg_latency_seconds": statistics.mean(latencies),
                    "p95_latency_seconds": percentile(latencies, 0.95),
                    "avg_prompt_tokens_est": statistics.mean(prompt_tokens),
                    "avg_total_tokens_est": statistics.mean(total_tokens),
                }
            )
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = [
        "method",
        "dataset",
        "split",
        "group_key",
        "group_value",
        "num_samples",
        "BLEU-4",
        "ROUGE-L",
        "CIDEr",
        "Distinct-1",
        "Distinct-2",
        "avg_latency_seconds",
        "p95_latency_seconds",
        "avg_prompt_tokens_est",
        "avg_total_tokens_est",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["dataset"], row["split"], row["group_key"], row["group_value"], row["method"]): row
        for row in rows
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] != "Chrono-full":
            continue
        key = (row["dataset"], row["split"], row["group_key"], row["group_value"], "Chrono-no_memory")
        base = index.get(key)
        if not base:
            continue
        out.append(
            {
                "dataset": row["dataset"],
                "split": row["split"],
                "group_key": row["group_key"],
                "group_value": row["group_value"],
                "num_samples": row["num_samples"],
                "full_BLEU-4": row["BLEU-4"],
                "no_memory_BLEU-4": base["BLEU-4"],
                "delta_BLEU-4": row["BLEU-4"] - base["BLEU-4"],
                "full_CIDEr": row["CIDEr"],
                "no_memory_CIDEr": base["CIDEr"],
                "delta_CIDEr": row["CIDEr"] - base["CIDEr"],
                "full_avg_latency_seconds": row["avg_latency_seconds"],
                "no_memory_avg_latency_seconds": base["avg_latency_seconds"],
                "delta_avg_latency_seconds": row["avg_latency_seconds"] - base["avg_latency_seconds"],
                "full_avg_total_tokens_est": row["avg_total_tokens_est"],
                "no_memory_avg_total_tokens_est": base["avg_total_tokens_est"],
                "delta_avg_total_tokens_est": row["avg_total_tokens_est"] - base["avg_total_tokens_est"],
            }
        )
    return sorted(out, key=lambda r: (r["split"], r["dataset"], r["group_key"], -r["delta_CIDEr"]))


def write_markdown(rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]], path: Path) -> None:
    lines: list[str] = ["# Grouped Full-Test Analysis", ""]
    focus_group_keys = ["operator", "answer_type", "time_level", "edge_count", "graph_shape"]
    for dataset in ("CRONQUESTION", "MULTITQ"):
        lines.append(f"## {dataset}")
        lines.append("")
        for group_key in focus_group_keys:
            subset = [
                row for row in delta_rows
                if row["dataset"] == dataset and row["split"] == "test" and row["group_key"] == group_key
            ]
            if not subset:
                continue
            lines.append(f"### {group_key}")
            lines.append("")
            lines.append("| Group | N | Full CIDEr | No-memory CIDEr | Delta CIDEr | Full BLEU-4 | No-memory BLEU-4 | Delta BLEU-4 | Full Avg Lat (s) | No-memory Avg Lat (s) |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in subset[:12]:
                lines.append(
                    f"| {row['group_value']} | {fmt(row['num_samples'],0)} | {fmt(row['full_CIDEr'])} | {fmt(row['no_memory_CIDEr'])} | {fmt(row['delta_CIDEr'])} | {fmt(row['full_BLEU-4'])} | {fmt(row['no_memory_BLEU-4'])} | {fmt(row['delta_BLEU-4'])} | {fmt(row['full_avg_latency_seconds'],3)} | {fmt(row['no_memory_avg_latency_seconds'],3)} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped analysis for Chrono supplementary runs")
    parser.add_argument("--result-dir", default=str(ROOT / "result" / "paper_runs" / "chrono_full"))
    parser.add_argument("--split", default="test", choices=["demo100", "test"])
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument(
        "--group-keys",
        nargs="+",
        default=["operator", "answer_type", "time_level", "edge_count", "graph_shape", "ask_slot"],
    )
    parser.add_argument("--out-dir", default=str(ROOT / "supplementary_experiments" / "results"))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_paths = sorted(result_dir.glob(f"*_{args.split}.json"))
    all_rows: list[dict[str, Any]] = []
    for result_path in tqdm(result_paths, desc="Runs"):
        if result_path.name.endswith(".metrics.json") or result_path.name.endswith(".summary.json") or result_path.name.endswith(".run_config.json"):
            continue
        per_sample_path = result_path.with_suffix(".per_sample.jsonl")
        if not per_sample_path.exists():
            continue
        meta, rows = load_run(result_path, per_sample_path)
        all_rows.extend(build_group_rows(meta, rows, args.group_keys, args.min_group_size))

    csv_path = out_dir / f"grouped_analysis_{args.split}.csv"
    md_path = out_dir / f"grouped_analysis_{args.split}.md"
    delta_path = out_dir / f"grouped_analysis_{args.split}.delta.json"
    write_csv(all_rows, csv_path)
    delta_rows = build_delta_rows(all_rows)
    write_markdown(all_rows, delta_rows, md_path)
    delta_path.write_text(json.dumps(delta_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {delta_path}")


if __name__ == "__main__":
    main()
