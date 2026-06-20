from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCALABILITY_DIR = ROOT / "result" / "paper_runs" / "scalability"
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


def extract_scale(name: str) -> int | None:
    m = re.search(r"scale(\d+)", name)
    return int(m.group(1)) if m else None


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(SCALABILITY_DIR.glob("*.summary.json")):
        payload = load_json(path)
        scale = extract_scale(path.name)
        if scale is None:
            continue
        metrics = payload.get("metrics") or {}
        cache = payload.get("cache_info") or {}
        rows.append(
            {
                "scale": scale,
                "num_samples": payload.get("num_samples"),
                "BLEU-4": metrics.get("BLEU-4"),
                "ROUGE-L": metrics.get("ROUGE-L"),
                "CIDEr": metrics.get("CIDEr"),
                "avg_latency_seconds": payload.get("avg_latency_seconds"),
                "p95_latency_seconds": payload.get("p95_latency_seconds"),
                "throughput_samples_per_min": payload.get("throughput_samples_per_min"),
                "avg_total_tokens_est": payload.get("avg_total_tokens_est"),
                "cache_build_seconds": cache.get("build_seconds"),
                "cache_size_bytes": cache.get("cache_size_bytes"),
                "cache_size_mb": (cache.get("cache_size_bytes") or 0) / (1024 * 1024),
                "summary_path": str(path),
            }
        )
    return sorted(rows, key=lambda r: r["scale"])


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "scale",
        "num_samples",
        "BLEU-4",
        "ROUGE-L",
        "CIDEr",
        "avg_latency_seconds",
        "p95_latency_seconds",
        "throughput_samples_per_min",
        "avg_total_tokens_est",
        "cache_build_seconds",
        "cache_size_bytes",
        "cache_size_mb",
        "summary_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# MULTITQ Scalability Summary",
        "",
        "| Train Scale | BLEU-4 | ROUGE-L | CIDEr | Avg Lat (s) | P95 Lat (s) | Throughput (/min) | Avg Total Tok | Cache Build (s) | Cache Size (MB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {fmt(row['scale'],0)} | {fmt(row['BLEU-4'])} | {fmt(row['ROUGE-L'])} | {fmt(row['CIDEr'])} | {fmt(row['avg_latency_seconds'],3)} | {fmt(row['p95_latency_seconds'],3)} | {fmt(row['throughput_samples_per_min'],3)} | {fmt(row['avg_total_tokens_est'],2)} | {fmt(row['cache_build_seconds'],3)} | {fmt(row['cache_size_mb'],2)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    csv_path = OUT_DIR / "scalability_multitq_summary.csv"
    md_path = OUT_DIR / "scalability_multitq_summary.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
