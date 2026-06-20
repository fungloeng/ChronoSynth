from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "chronosynth"))
from chronoagent_harness.generator import run_generation

DEFAULT_MODEL = os.getenv("TKGQG_PROMPT_MODEL", "Qwen3.5-9B")
DEFAULT_WORKERS = int(os.getenv("TKGQG_PROMPT_WORKERS", "20"))
SHARD_SIZE = 1000
SAVE_EVERY = 100


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ChronoHarness prompt baseline on TKGQG")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_mode", choices=["responses", "chat", "auto"], default="responses")
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--memory_bank_cache_dir",
        default=str(ROOT / "result" / "full_chrono" / "memory_bank_cache"),
        help="Directory containing per-dataset ChronoHarness memory bank caches",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CRONQUESTION", "MULTITQ"],
        choices=["CRONQUESTION", "MULTITQ"],
    )
    args = parser.parse_args()

    tasks = []
    for dataset in args.datasets:
        tasks.append(
            (
                dataset,
                ROOT / "data" / dataset / "train.json",
                ROOT / "data" / dataset / "test.json",
            )
        )

    out_dir = ROOT / "result" / "full_chrono"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.memory_bank_cache_dir)

    for dataset, train_path, test_path in tasks:
        output_path = out_dir / f"chrono_harness_{dataset}_full.json"
        cache_path = cache_dir / f"chrono_{dataset.lower()}_memory_bank.pkl"
        run_generation(
            train_path=train_path,
            input_path=test_path,
            output_path=output_path,
            model=args.model,
            workers=args.workers,
            shard_size=args.shard_size,
            save_every=args.save_every,
            resume=True,
            include_prompt=False,
            api_key=args.api_key,
            base_url=args.base_url,
            api_mode=args.api_mode,
            reasoning_effort=args.reasoning_effort,
            enable_thinking=args.enable_thinking,
            memory_bank_cache=cache_path,
        )


if __name__ == "__main__":
    main()
