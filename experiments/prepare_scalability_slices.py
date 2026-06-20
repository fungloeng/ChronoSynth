from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train slices for scalability experiments")
    parser.add_argument("--train", default="data/MULTITQ/train.json")
    parser.add_argument("--out_dir", default="data/MULTITQ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scales", nargs="+", type=int, default=[20, 40, 80])
    args = parser.parse_args()

    train_path = Path(args.train)
    out_dir = Path(args.out_dir)
    data = load_json(train_path)

    rng = random.Random(args.seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    shuffled = [data[i] for i in indices]

    for scale in args.scales:
        size = max(1, int(len(shuffled) * (scale / 100.0)))
        out_path = out_dir / f"train_scale{scale}.json"
        save_json(out_path, shuffled[:size])
        print(f"Wrote {out_path} ({size} / {len(shuffled)})")


if __name__ == "__main__":
    main()
