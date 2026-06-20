from __future__ import annotations

import argparse
from pathlib import Path

from .core import build_memory_bank, build_memory_bank_cache_metadata, guess_dataset_name, save_memory_bank
from .data import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist a ChronoAgentHarness memory bank cache")
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    train_path = Path(args.train)
    dataset = args.dataset or guess_dataset_name(path=str(train_path))
    train_data = load_json(train_path)
    memory_bank = build_memory_bank(train_data, dataset=dataset)
    metadata = build_memory_bank_cache_metadata(train_path, dataset)
    cache_path = save_memory_bank(memory_bank, args.output, metadata=metadata)
    print(f"Saved memory bank cache to {cache_path}")


if __name__ == "__main__":
    main()
