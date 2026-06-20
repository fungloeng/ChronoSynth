from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


class ProgressTracker:
    def __init__(
        self,
        output_path: str | Path,
        eval_data: list[dict[str, Any]],
        shard_size: int = 1000,
        save_every: int = 100,
        resume: bool = True,
        retry_errors: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.eval_data = eval_data
        self.shard_size = max(1, shard_size)
        self.save_every = max(1, save_every)
        self.resume = resume
        self.retry_errors = retry_errors
        self.shard_dir = self.output_path.parent / f"{self.output_path.stem}.shards"

        self.results_by_index: dict[int, dict[str, Any]] = {}
        self._completed_since_flush = 0
        self._dirty_shards: set[int] = set()

        if self.resume:
            self._load_existing()

    @staticmethod
    def _is_error_result(item: dict[str, Any]) -> bool:
        generated = item.get("generated_question")
        return isinstance(generated, str) and generated.startswith("ERROR:")

    def _load_existing(self) -> None:
        if self.output_path.exists():
            for item in _load_json(self.output_path):
                idx = item.get("sample_index")
                if isinstance(idx, int) and not (self.retry_errors and self._is_error_result(item)):
                    self.results_by_index[idx] = item

        if not self.results_by_index and self.shard_dir.exists():
            for shard_file in sorted(self.shard_dir.glob("shard_*.json")):
                for item in _load_json(shard_file):
                    idx = item.get("sample_index")
                    if isinstance(idx, int) and not (self.retry_errors and self._is_error_result(item)):
                        self.results_by_index[idx] = item

    def record(self, index: int, result: dict[str, Any]) -> None:
        result["sample_index"] = index
        self.results_by_index[index] = result
        self._completed_since_flush += 1
        self._dirty_shards.add(index // self.shard_size)
        if self._completed_since_flush >= self.save_every:
            self.flush()

    def flush(self) -> None:
        if not self.results_by_index:
            return

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        for shard_id in sorted(self._dirty_shards):
            start = shard_id * self.shard_size
            stop = min((shard_id + 1) * self.shard_size, len(self.eval_data))
            shard_items = [
                self.results_by_index[idx]
                for idx in range(start, stop)
                if idx in self.results_by_index
            ]
            shard_path = self.shard_dir / f"shard_{start:05d}_{stop - 1:05d}.json"
            _save_json(shard_path, shard_items)

        ordered = [
            self.results_by_index[idx]
            for idx in range(len(self.eval_data))
            if idx in self.results_by_index
        ]
        _save_json(self.output_path, ordered)

        self._dirty_shards.clear()
        self._completed_since_flush = 0

    def finalize(self) -> None:
        self.flush()

    def pending_indices(self) -> list[int]:
        return [idx for idx in range(len(self.eval_data)) if idx not in self.results_by_index]

    def completed_count(self) -> int:
        return len(self.results_by_index)

    def error_count(self) -> int:
        count = 0
        if self.output_path.exists():
            for item in _load_json(self.output_path):
                if self._is_error_result(item):
                    count += 1
        return count

    def summary(self) -> str:
        total = len(self.eval_data)
        done = len(self.results_by_index)
        shards = ceil(total / self.shard_size)
        return f"{done}/{total} complete, shards={shards}, output={self.output_path}"
