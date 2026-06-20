from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "supplementary_experiments" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def collect_variance_rows() -> list[dict[str, Any]]:
    # run1 comes from chrono_demo100; rerun2/3 come from chrono_remaining
    variants = [
        (
            "Chrono-full",
            "result/paper_runs/chrono_demo100/chrono_full_MULTITQ_demo100.summary.json",
            "result/paper_runs/chrono_remaining/chrono_full_MULTITQ_demo100_rerun2.summary.json",
            "result/paper_runs/chrono_remaining/chrono_full_MULTITQ_demo100_rerun3.summary.json",
        ),
        (
            "Chrono-no_memory",
            "result/paper_runs/chrono_demo100/chrono_no_memory_MULTITQ_demo100.summary.json",
            "result/paper_runs/chrono_remaining/chrono_no_memory_MULTITQ_demo100_rerun2.summary.json",
            "result/paper_runs/chrono_remaining/chrono_no_memory_MULTITQ_demo100_rerun3.summary.json",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for method, p1, p2, p3 in variants:
        paths = [ROOT / p1, ROOT / p2, ROOT / p3]
        payloads = [load_json(path) for path in paths if path.exists()]
        if len(payloads) != 3:
            continue

        bleu4 = [float(p["metrics"]["BLEU-4"]) for p in payloads]
        rouge = [float(p["metrics"]["ROUGE-L"]) for p in payloads]
        cider = [float(p["metrics"]["CIDEr"]) for p in payloads]
        lat = [float(p["avg_latency_seconds"]) for p in payloads]
        p95 = [float(p["p95_latency_seconds"]) for p in payloads]
        tok = [float(p["avg_total_tokens_est"]) for p in payloads]

        bleu_m, bleu_s = mean_std(bleu4)
        rouge_m, rouge_s = mean_std(rouge)
        cider_m, cider_s = mean_std(cider)
        lat_m, lat_s = mean_std(lat)
        p95_m, p95_s = mean_std(p95)
        tok_m, tok_s = mean_std(tok)

        rows.append(
            {
                "method": method,
                "dataset": "MULTITQ",
                "split": "demo100",
                "num_runs": 3,
                "BLEU-4_mean": bleu_m,
                "BLEU-4_std": bleu_s,
                "ROUGE-L_mean": rouge_m,
                "ROUGE-L_std": rouge_s,
                "CIDEr_mean": cider_m,
                "CIDEr_std": cider_s,
                "avg_latency_seconds_mean": lat_m,
                "avg_latency_seconds_std": lat_s,
                "p95_latency_seconds_mean": p95_m,
                "p95_latency_seconds_std": p95_s,
                "avg_total_tokens_est_mean": tok_m,
                "avg_total_tokens_est_std": tok_s,
                "run1_path": str(paths[0]),
                "run2_path": str(paths[1]),
                "run3_path": str(paths[2]),
            }
        )
    return rows


def collect_cache_row() -> dict[str, Any] | None:
    miss_path = ROOT / "result/paper_runs/chrono_remaining/chrono_full_MULTITQ_demo100_cache_miss.summary.json"
    hit_path = ROOT / "result/paper_runs/chrono_remaining/chrono_full_MULTITQ_demo100_cache_hit.summary.json"
    if not (miss_path.exists() and hit_path.exists()):
        return None

    miss = load_json(miss_path)
    hit = load_json(hit_path)
    miss_wall = float(miss["wall_time_seconds"])
    hit_wall = float(hit["wall_time_seconds"])
    miss_build = float((miss.get("cache_info") or {}).get("build_seconds") or 0.0)
    hit_build = float((hit.get("cache_info") or {}).get("build_seconds") or 0.0)
    cache_size = float((hit.get("cache_info") or {}).get("cache_size_bytes") or 0.0)

    return {
        "dataset": "MULTITQ",
        "split": "demo100",
        "method": "Chrono-full",
        "miss_wall_time_seconds": miss_wall,
        "hit_wall_time_seconds": hit_wall,
        "miss_cache_build_seconds": miss_build,
        "hit_cache_build_seconds": hit_build,
        "wall_time_speedup": (miss_wall / hit_wall) if hit_wall > 0 else None,
        "cache_build_reduction_seconds": miss_build - hit_build,
        "cache_size_bytes": cache_size,
        "cache_size_mb": cache_size / (1024 * 1024),
        "miss_summary_path": str(miss_path),
        "hit_summary_path": str(hit_path),
    }


def write_variance_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "dataset",
        "split",
        "num_runs",
        "BLEU-4_mean",
        "BLEU-4_std",
        "ROUGE-L_mean",
        "ROUGE-L_std",
        "CIDEr_mean",
        "CIDEr_std",
        "avg_latency_seconds_mean",
        "avg_latency_seconds_std",
        "p95_latency_seconds_mean",
        "p95_latency_seconds_std",
        "avg_total_tokens_est_mean",
        "avg_total_tokens_est_std",
        "run1_path",
        "run2_path",
        "run3_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cache_csv(row: dict[str, Any] | None, path: Path) -> None:
    fields = [
        "dataset",
        "split",
        "method",
        "miss_wall_time_seconds",
        "hit_wall_time_seconds",
        "miss_cache_build_seconds",
        "hit_cache_build_seconds",
        "wall_time_speedup",
        "cache_build_reduction_seconds",
        "cache_size_bytes",
        "cache_size_mb",
        "miss_summary_path",
        "hit_summary_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if row is not None:
            writer.writerow(row)


def write_md(variance_rows: list[dict[str, Any]], cache_row: dict[str, Any] | None, path: Path) -> None:
    lines: list[str] = ["# Variance and Cache Summary", ""]

    lines.append("## Variance (3 runs, MULTITQ demo100)")
    lines.append("")
    lines.append("| Method | BLEU-4 (mean±std) | ROUGE-L (mean±std) | CIDEr (mean±std) | Avg Lat (s, mean±std) | P95 Lat (s, mean±std) | Avg Total Tok (mean±std) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in variance_rows:
        lines.append(
            f"| {row['method']} | {fmt(row['BLEU-4_mean'])}±{fmt(row['BLEU-4_std'])} | {fmt(row['ROUGE-L_mean'])}±{fmt(row['ROUGE-L_std'])} | {fmt(row['CIDEr_mean'])}±{fmt(row['CIDEr_std'])} | {fmt(row['avg_latency_seconds_mean'],3)}±{fmt(row['avg_latency_seconds_std'],3)} | {fmt(row['p95_latency_seconds_mean'],3)}±{fmt(row['p95_latency_seconds_std'],3)} | {fmt(row['avg_total_tokens_est_mean'],2)}±{fmt(row['avg_total_tokens_est_std'],2)} |"
        )

    lines.append("")
    lines.append("## Cache Ablation (MULTITQ demo100, Chrono-full)")
    lines.append("")
    lines.append("| Miss Wall (s) | Hit Wall (s) | Wall Speedup (x) | Miss Build (s) | Hit Build (s) | Build Reduction (s) | Cache Size (MB) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    if cache_row is not None:
        lines.append(
            f"| {fmt(cache_row['miss_wall_time_seconds'],3)} | {fmt(cache_row['hit_wall_time_seconds'],3)} | {fmt(cache_row['wall_time_speedup'],3)} | {fmt(cache_row['miss_cache_build_seconds'],3)} | {fmt(cache_row['hit_cache_build_seconds'],3)} | {fmt(cache_row['cache_build_reduction_seconds'],3)} | {fmt(cache_row['cache_size_mb'],2)} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    variance_rows = collect_variance_rows()
    cache_row = collect_cache_row()

    var_csv = OUT_DIR / "variance_multitq_demo100_summary.csv"
    cache_csv = OUT_DIR / "cache_ablation_multitq_demo100_summary.csv"
    md_path = OUT_DIR / "variance_cache_summary.md"

    write_variance_csv(variance_rows, var_csv)
    write_cache_csv(cache_row, cache_csv)
    write_md(variance_rows, cache_row, md_path)

    print(f"Wrote {var_csv}")
    print(f"Wrote {cache_csv}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
