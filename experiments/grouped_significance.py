from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = sorted(values)
    idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * q))))
    return arr[idx]


def fmt(v: Any, d: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def load_joined(result_json: Path, per_sample_jsonl: Path) -> list[dict[str, Any]]:
    results = load_json(result_json)
    stats = {int(r["sample_index"]): r for r in load_jsonl(per_sample_jsonl)}
    out: list[dict[str, Any]] = []
    for row in results:
        idx = int(row["sample_index"])
        s = stats.get(idx)
        if not s or s.get("status") != "ok":
            continue
        refs = row.get("reference_questions") or []
        if isinstance(refs, str):
            refs = [refs]
        refs = [str(x).strip() for x in refs if str(x).strip()]
        hyp = str(row.get("generated_question", "")).strip()
        if not refs or not hyp or hyp.startswith("ERROR:"):
            continue
        out.append(
            {
                "question_id": row.get("question_id"),
                "generated_question": hyp,
                "reference_questions": refs,
                "operator": s.get("operator"),
                "answer_type": s.get("answer_type"),
                "time_level": s.get("time_level"),
                "edge_count": str(s.get("edge_count")),
                "graph_shape": s.get("graph_shape"),
            }
        )
    return out


def bootstrap_delta(
    full_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    reps: int,
    seed: int,
    sample_size: int,
) -> dict[str, float]:
    full_by_qid = {str(r["question_id"]): r for r in full_rows}
    base_by_qid = {str(r["question_id"]): r for r in base_rows}
    qids = sorted(set(full_by_qid) & set(base_by_qid))
    if not qids:
        return {}

    def eval_metric(sample_qids: list[str]) -> tuple[float, float]:
        full_pairs = [(full_by_qid[q]["generated_question"], full_by_qid[q]["reference_questions"]) for q in sample_qids]
        base_pairs = [(base_by_qid[q]["generated_question"], base_by_qid[q]["reference_questions"]) for q in sample_qids]
        mf = evaluate(full_pairs, compute_cider=True, verbose=False)
        mb = evaluate(base_pairs, compute_cider=True, verbose=False)
        return float(mf["BLEU-4"] - mb["BLEU-4"]), float(mf["CIDEr"] - mb["CIDEr"])

    rng = random.Random(seed)
    delta_bleu: list[float] = []
    delta_cider: list[float] = []
    draw_n = max(1, min(sample_size, len(qids)))
    for _ in range(reps):
        sample = [qids[rng.randrange(len(qids))] for _ in range(draw_n)]
        b, c = eval_metric(sample)
        delta_bleu.append(b)
        delta_cider.append(c)

    # point estimate on full set
    point_bleu, point_cider = eval_metric(qids)
    return {
        "n": len(qids),
        "delta_BLEU-4": point_bleu,
        "ci95_low_BLEU-4": percentile(delta_bleu, 0.025),
        "ci95_high_BLEU-4": percentile(delta_bleu, 0.975),
        "significant_BLEU-4": percentile(delta_bleu, 0.025) > 0,
        "delta_CIDEr": point_cider,
        "ci95_low_CIDEr": percentile(delta_cider, 0.025),
        "ci95_high_CIDEr": percentile(delta_cider, 0.975),
        "significant_CIDEr": percentile(delta_cider, 0.025) > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap significance for grouped full-vs-no_memory deltas")
    parser.add_argument("--result-dir", default=str(ROOT / "result" / "paper_runs" / "chrono_full"))
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--out-dir", default=str(ROOT / "supplementary_experiments" / "results"))
    parser.add_argument(
        "--group-keys",
        nargs="+",
        default=["operator", "answer_type", "time_level", "edge_count", "graph_shape"],
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["CRONQUESTION", "MULTITQ"]
    group_keys = args.group_keys

    out_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        full_json = result_dir / f"chrono_full_{dataset}_test.json"
        full_ps = result_dir / f"chrono_full_{dataset}_test.per_sample.jsonl"
        base_json = result_dir / f"chrono_no_memory_{dataset}_test.json"
        base_ps = result_dir / f"chrono_no_memory_{dataset}_test.per_sample.jsonl"
        if not (full_json.exists() and full_ps.exists() and base_json.exists() and base_ps.exists()):
            continue
        full_rows = load_joined(full_json, full_ps)
        base_rows = load_joined(base_json, base_ps)
        print(f"[significance] dataset={dataset}")
        for gk in group_keys:
            print(f"[significance]   group_key={gk}")
            # group using full side; then intersect with base by qid inside bootstrap_delta
            groups: dict[str, list[dict[str, Any]]] = {}
            for r in full_rows:
                gv = str(r.get(gk) or "UNKNOWN")
                groups.setdefault(gv, []).append(r)
            for gv, fr in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                if len(fr) < args.min_group_size:
                    continue
                full_ids = {str(x["question_id"]) for x in fr}
                br = [x for x in base_rows if str(x["question_id"]) in full_ids]
                stats = bootstrap_delta(fr, br, reps=args.reps, seed=args.seed, sample_size=args.sample_size)
                if not stats:
                    continue
                out_rows.append(
                    {
                        "dataset": dataset,
                        "group_key": gk,
                        "group_value": gv,
                        **stats,
                    }
                )

    csv_path = out_dir / "grouped_significance_test.csv"
    md_path = out_dir / "grouped_significance_test.md"
    json_path = out_dir / "grouped_significance_test.json"

    import csv

    fields = [
        "dataset",
        "group_key",
        "group_value",
        "n",
        "delta_BLEU-4",
        "ci95_low_BLEU-4",
        "ci95_high_BLEU-4",
        "significant_BLEU-4",
        "delta_CIDEr",
        "ci95_low_CIDEr",
        "ci95_high_CIDEr",
        "significant_CIDEr",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    lines = ["# Grouped Significance (Bootstrap CI)", ""]
    for dataset in datasets:
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append("| Group Key | Group Value | N | Delta CIDEr | CI95 CIDEr | Sig CIDEr | Delta BLEU-4 | CI95 BLEU-4 | Sig BLEU-4 |")
        lines.append("|---|---|---:|---:|---|---:|---:|---|---:|")
        subset = [r for r in out_rows if r["dataset"] == dataset]
        subset = sorted(subset, key=lambda x: (x["group_key"], -x["delta_CIDEr"]))
        for r in subset:
            ci_cider = f"[{fmt(r['ci95_low_CIDEr'])}, {fmt(r['ci95_high_CIDEr'])}]"
            ci_bleu = f"[{fmt(r['ci95_low_BLEU-4'])}, {fmt(r['ci95_high_BLEU-4'])}]"
            lines.append(
                f"| {r['group_key']} | {r['group_value']} | {fmt(r['n'],0)} | {fmt(r['delta_CIDEr'])} | {ci_cider} | {1 if r['significant_CIDEr'] else 0} | {fmt(r['delta_BLEU-4'])} | {ci_bleu} | {1 if r['significant_BLEU-4'] else 0} |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
