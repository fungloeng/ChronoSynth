from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "chronosynth"))
from chronoagent_harness.generator import run_generation


DEFAULT_MODEL = os.getenv("TKGQG_KGQG_MODEL", "gpt-4o-mini")
DEFAULT_WORKERS = int(os.getenv("TKGQG_PROMPT_WORKERS", "20"))


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def _prepared_paths(dataset: str) -> tuple[Path, Path]:
    base = ROOT / "data" / "chronosynth_kgqg" / dataset
    return base / "train.json", base / "test.json"


def _strategy_suffix(static_strategy: str) -> str:
    return "" if static_strategy == "transfer_v1" else f"_{static_strategy}"


def _write_static_summary(raw_metrics_path: Path, summary_path: Path) -> None:
    raw = json.loads(raw_metrics_path.read_text(encoding="utf-8"))
    summary = {}
    for key in ("BLEU-4", "ROUGE-L"):
        if key in raw:
            summary[key] = raw[key] * 100.0
    if "BLEU-4" in summary and "ROUGE-L" in summary:
        summary["Overall"] = (summary["BLEU-4"] + summary["ROUGE-L"]) / 2.0
    summary["num_samples"] = raw.get("num_samples")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_valid_results(output_path: Path) -> int:
    data = json.loads(output_path.read_text(encoding="utf-8"))
    valid = 0
    for item in data:
        hyp = str(item.get("generated_question", "")).strip()
        raw_refs = item.get("reference_questions", [])
        if isinstance(raw_refs, str):
            raw_refs = [raw_refs]
        refs = [ref for ref in raw_refs if isinstance(ref, str) and ref.strip()]
        if hyp and not hyp.startswith("ERROR:") and refs:
            valid += 1
    return valid


def _write_eval_failure_summary(summary_path: Path, output_path: Path, reason: str) -> None:
    summary = {
        "status": "evaluation_failed",
        "reason": reason,
        "num_valid_samples": _count_valid_results(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ChronoSynth on static KGQG datasets.")
    parser.add_argument("--datasets", nargs="+", default=["WQ", "PQ"], choices=["WQ", "CWQ", "PQ"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_mode", choices=["responses", "chat", "auto"], default="responses")
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--static_strategy", choices=["transfer_v1", "transfer_v1b", "sota_v2", "sota_v3", "sota_v4"], default="transfer_v1")
    args = parser.parse_args()

    if not args.skip_prepare:
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_chronosynth_kgqg.py"),
                "--datasets",
                *args.datasets,
            ]
        )

    out_dir = ROOT / "result" / "full_chrono_kgqg"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "memory_bank_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        train_path, test_path = _prepared_paths(dataset)
        suffix = _strategy_suffix(args.static_strategy)
        output_path = out_dir / f"chrono_kgqg_{dataset}_full{suffix}.json"
        cache_path = cache_dir / f"chrono_{dataset.lower()}_memory_bank.pkl"
        print(
            f"[static-kgqg] dataset={dataset} model={args.model} workers={args.workers} "
            f"output={output_path}",
            flush=True,
        )
        run_generation(
            train_path=train_path,
            input_path=test_path,
            output_path=output_path,
            model=args.model,
            workers=args.workers,
            shard_size=args.shard_size,
            save_every=args.save_every,
            resume=not args.no_resume,
            include_prompt=False,
            memory_bank_cache=cache_path,
            api_key=args.api_key,
            base_url=args.base_url,
            api_mode=args.api_mode,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
            dataset=dataset,
            static_strategy=args.static_strategy,
        )

        if not args.no_eval:
            raw_metrics_path = out_dir / f"chrono_kgqg_{dataset}_full{suffix}.raw.metrics.json"
            metrics_path = out_dir / f"chrono_kgqg_{dataset}_full{suffix}.metrics.json"
            try:
                _run(
                    [
                        sys.executable,
                        str(ROOT / "evaluate.py"),
                        str(output_path),
                        "--out",
                        str(raw_metrics_path),
                    ]
                )
                _write_static_summary(raw_metrics_path, metrics_path)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[warn] evaluation failed for {dataset}: {exc}. "
                    f"Keeping generation output and continuing.",
                    flush=True,
                )
                _write_eval_failure_summary(metrics_path, output_path, str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
