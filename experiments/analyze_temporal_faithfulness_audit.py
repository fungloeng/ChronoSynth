from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chronosynth"))

from chronoagent_harness.core import build_operator_state  # noqa: E402
from chronoagent_harness.data import load_json  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "paper_results" / "temporal_faithfulness_audit"
)

DEFAULT_DATA = {
    "CRONQUESTION": ROOT / "data" / "CRONQUESTION" / "test.json",
    "MULTITQ": ROOT / "data" / "MULTITQ" / "test.json",
}

METHOD_ORDER = ["direct", "kqg_cot", "no_memory", "full"]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean_ci(values: list[float], seed: int = 42, rounds: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        value = values[0]
        return value, value
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(rounds):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def paired_bootstrap_diff(
    left: list[float], right: list[float], seed: int = 42, rounds: int = 5000
) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    if len(left) != len(right):
        raise ValueError("paired bootstrap requires equally sized lists")
    if len(left) == 1:
        diff = right[0] - left[0]
        return diff, diff
    rng = random.Random(seed)
    diffs = []
    n = len(left)
    for _ in range(rounds):
        idxs = [rng.randrange(n) for _ in range(n)]
        draw = sum(right[i] - left[i] for i in idxs) / n
        diffs.append(draw)
    return _percentile(diffs, 0.025), _percentile(diffs, 0.975)


def exact_two_sided_sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _summary_rows(payload: dict[str, Any]) -> dict[str, float]:
    summary = payload["summary"]
    return {
        "num_samples": int(summary["num_samples"]),
        "operator_accuracy": float(summary["operator_accuracy"]),
        "slot_accuracy": float(summary["slot_accuracy"]),
        "comparator_accuracy": float(summary["comparator_accuracy"]),
        "temporal_faithfulness": float(summary["temporal_faithfulness"]),
        "answer_leakage_rate": float(summary["answer_leakage_rate"]),
    }


def _load_judge(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paired_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_by_id = {row["question_id"]: row for row in left}
    right_by_id = {row["question_id"]: row for row in right}
    ids = sorted(set(left_by_id) & set(right_by_id))
    return [(left_by_id[qid], right_by_id[qid]) for qid in ids]


def _paired_metric(pairs: list[tuple[dict[str, Any], dict[str, Any]]], metric: str) -> dict[str, Any]:
    left = [float(a[metric]) for a, _ in pairs]
    right = [float(b[metric]) for _, b in pairs]
    diff = [r - l for l, r in zip(left, right)]
    wins = sum(1 for value in diff if value > 0)
    losses = sum(1 for value in diff if value < 0)
    ties = sum(1 for value in diff if value == 0)
    mean_delta = sum(diff) / len(diff) if diff else 0.0
    ci_low, ci_high = paired_bootstrap_diff(left, right)
    return {
        "n": len(diff),
        "left_mean": sum(left) / len(left) if left else 0.0,
        "right_mean": sum(right) / len(right) if right else 0.0,
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_value": exact_two_sided_sign_test(wins, losses),
    }


def _paired_operator_breakdown(
    raw_data: list[dict[str, Any]],
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    dataset: str,
) -> list[dict[str, Any]]:
    raw_by_id = {item["question_id"]: item for item in raw_data}
    left_by_id = {row["question_id"]: row for row in left}
    right_by_id = {row["question_id"]: row for row in right}
    rows_by_operator: dict[str, list[int]] = defaultdict(list)
    for qid in sorted(set(left_by_id) & set(right_by_id) & set(raw_by_id)):
        op = build_operator_state(raw_by_id[qid], dataset=dataset).operator
        rows_by_operator[op].append(qid)

    output = []
    for op, qids in sorted(rows_by_operator.items(), key=lambda item: (-len(item[1]), item[0])):
        left_vals = [left_by_id[qid]["temporal_faithful"] for qid in qids]
        right_vals = [right_by_id[qid]["temporal_faithful"] for qid in qids]
        left_leak = [left_by_id[qid]["answer_leakage"] for qid in qids]
        right_leak = [right_by_id[qid]["answer_leakage"] for qid in qids]
        output.append(
            {
                "operator": op,
                "n": len(qids),
                "left_faithful": sum(left_vals) / len(left_vals),
                "right_faithful": sum(right_vals) / len(right_vals),
                "faithful_delta": (sum(right_vals) - sum(left_vals)) / len(qids),
                "left_leakage": sum(left_leak) / len(left_leak),
                "right_leakage": sum(right_leak) / len(right_leak),
                "leakage_delta": (sum(right_leak) - sum(left_leak)) / len(qids),
            }
        )
    return output


def _format_ci(low: float, high: float) -> str:
    return f"[{_pct(low)}%, {_pct(high)}%]"


def _method_row(dataset: str, method_key: str, method: str, summary: dict[str, float]) -> str:
    return (
        f"| {dataset} | {method} | {summary['num_samples']} | "
        f"{_pct(summary['temporal_faithfulness'])} | {_pct(summary['answer_leakage_rate'])} | "
        f"{_pct(summary['operator_accuracy'])} | {_pct(summary['slot_accuracy'])} | "
        f"{_pct(summary['comparator_accuracy'])} |"
    )


def _write_report(
    rows: list[dict[str, Any]],
    paired_stats: dict[str, Any],
    operator_breakdown: dict[str, list[dict[str, Any]]],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Temporal Faithfulness Audit")
    lines.append("")
    lines.append("This report reuses the existing judge outputs and compares methods on the same `question_id` set.")
    lines.append("The core comparison is `ChronoSynth-no-memory` vs `ChronoSynth-full`.")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append("| Dataset | Method | Samples | Faithful | Leakage | Op. match | Slot match | Comparator match |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(_method_row(row["dataset"], row["method_key"], row["method"], row["summary"]))
    lines.append("")
    lines.append("## Paired Comparison: Full vs No-Memory")
    lines.append("")
    lines.append("| Dataset | Metric | Mean delta (full - no-memory) | 95% CI | Wins | Losses | Ties | p-value |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset, metrics in paired_stats.items():
        for metric_name in ["temporal_faithful", "answer_leakage", "operator_correct", "slot_correct", "comparator_correct"]:
            metric = metrics[metric_name]
            lines.append(
                f"| {dataset} | {metric_name} | {_pct(metric['mean_delta'])} | "
                f"{_format_ci(metric['ci_low'], metric['ci_high'])} | {metric['wins']} | {metric['losses']} | {metric['ties']} | {metric['p_value']:.4g} |"
            )
    lines.append("")
    lines.append("## Operator Breakdown: Full vs No-Memory")
    lines.append("")
    lines.append("| Dataset | Operator | N | Faithful(no-memory) | Faithful(full) | Delta | Leakage(no-memory) | Leakage(full) | Delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, rows_by_op in operator_breakdown.items():
        for row in rows_by_op:
            lines.append(
                f"| {dataset} | {row['operator']} | {row['n']} | {_pct(row['left_faithful'])} | {_pct(row['right_faithful'])} | "
                f"{_pct(row['faithful_delta'])} | {_pct(row['left_leakage'])} | {_pct(row['right_leakage'])} | {_pct(row['leakage_delta'])} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The audit is paired on the same sampled `question_id`s, so the comparison is not driven by different subsets.")
    lines.append("- `ChronoSynth-no-memory` is higher on raw temporal faithfulness in both datasets.")
    lines.append("- `ChronoSynth-full` is lower on leakage, so the design currently shows a faithfulness-vs-leakage trade-off rather than monotonic improvement.")
    lines.append("- If the paper needs a stronger claim, the claim itself must change or the method must be improved; post-hoc metrics cannot make a contradicted result true.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze the temporal-faithfulness audit outputs.")
    parser.add_argument(
        "--judge_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing the judge JSON outputs produced by run_temporal_faithfulness_table.py",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where analysis artifacts should be written",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_paths = {
        "CRONQUESTION": {
            "direct": judge_dir / "cronquestion_direct_judge.json",
            "kqg_cot": judge_dir / "cronquestion_kqg_cot_judge.json",
            "no_memory": judge_dir / "cronquestion_no_memory_judge.json",
            "full": judge_dir / "cronquestion_full_judge.json",
        },
        "MULTITQ": {
            "direct": judge_dir / "multitq_direct_judge.json",
            "kqg_cot": judge_dir / "multitq_kqg_cot_judge.json",
            "no_memory": judge_dir / "multitq_no_memory_judge.json",
            "full": judge_dir / "multitq_full_judge.json",
        },
    }

    rows: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for dataset, methods in method_paths.items():
        for method_key in METHOD_ORDER:
            payload = _load_judge(methods[method_key])
            payloads[f"{dataset}:{method_key}"] = payload
            rows.append(
                {
                    "dataset": dataset,
                    "method_key": method_key,
                    "method": {
                        "direct": "Direct Prompting",
                        "kqg_cot": "KQG-CoT",
                        "no_memory": "ChronoSynth-no-memory",
                        "full": "ChronoSynth-full",
                    }[method_key],
                    "summary": _summary_rows(payload),
                }
            )

    paired_stats: dict[str, dict[str, Any]] = {}
    operator_breakdown: dict[str, list[dict[str, Any]]] = {}
    for dataset in ["CRONQUESTION", "MULTITQ"]:
        left = payloads[f"{dataset}:no_memory"]["details"]
        right = payloads[f"{dataset}:full"]["details"]
        paired = _paired_rows(left, right)
        paired_stats[dataset] = {}
        for metric in [
            "temporal_faithful",
            "answer_leakage",
            "operator_correct",
            "slot_correct",
            "comparator_correct",
        ]:
            paired_stats[dataset][metric] = _paired_metric(paired, metric)
        raw_data = load_json(DEFAULT_DATA[dataset])
        operator_breakdown[dataset] = _paired_operator_breakdown(raw_data, left, right, dataset=dataset)

    report_path = output_dir / "temporal_faithfulness_paired_report.md"
    stats_path = output_dir / "temporal_faithfulness_paired_stats.json"
    _write_report(rows, paired_stats, operator_breakdown, report_path)
    stats_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "paired_stats": paired_stats,
                "operator_breakdown": operator_breakdown,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[done] report: {report_path}")
    print(f"[done] stats: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
