from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "chronosynth_kgqg"


def _clean_surface(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split()).strip()


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _read_lines(path) if line.strip()]


def _load_rel_dict(path: Path) -> dict[str, str]:
    rel_map: dict[str, str] = {}
    for line in _read_lines(path):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        rel_map[parts[0]] = parts[1].strip()
    return rel_map


def _load_ent_dict_list(path: Path) -> list[dict[str, list[str]]]:
    out: list[dict[str, list[str]]] = []
    for line in _read_lines(path):
        value = line.strip()
        if not value:
            continue
        out.append(ast.literal_eval(value))
    return out


def _choose_alias(aliases: list[str], fallback: str) -> str:
    for alias in aliases:
        text = _clean_surface(alias)
        if text and text.lower() != "none":
            return text
    return _clean_surface(fallback)


def _parse_cwq_entity_token(token: str) -> tuple[str, str]:
    parts = token.split("￨")
    base = parts[0].strip()
    tag = parts[1].strip() if len(parts) > 1 else ""
    return base, tag


def _parse_cwq_split(
    src_path: Path,
    tgt_path: Path,
    rel_map: dict[str, str],
    ent_dict_list: list[dict[str, list[str]]],
    offset: int,
    split: str,
) -> list[dict[str, Any]]:
    src_lines = _read_lines(src_path)
    tgt_lines = _read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        print(
            f"[warn] {split}: line count mismatch {len(src_lines)} vs {len(tgt_lines)}; "
            f"truncating to {min(len(src_lines), len(tgt_lines))}"
        )
    n = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:n]
    tgt_lines = tgt_lines[:n]

    items: list[dict[str, Any]] = []
    for idx, (src_line, tgt_line) in enumerate(
        tqdm(zip(src_lines, tgt_lines), total=n, desc=f"{split} CWQ", leave=False)
    ):
        entity_alias_map = ent_dict_list[offset + idx]
        edges = []
        answers = []
        triples = [chunk.strip() for chunk in src_line.split("<t>") if chunk.strip()]
        for triple in triples:
            tokens = triple.split()
            if len(tokens) < 3:
                continue
            head_raw, rel_raw, tail_raw = tokens[:3]
            head_base, head_tag = _parse_cwq_entity_token(head_raw)
            rel_base = rel_raw.split("￨", 1)[0].strip()
            tail_base, tail_tag = _parse_cwq_entity_token(tail_raw)
            head_key = f"Ent_{head_base[:-1] if head_base.endswith('e') else head_base}"
            tail_key = f"Ent_{tail_base[:-1] if tail_base.endswith('e') else tail_base}"
            head_surface = _choose_alias(entity_alias_map.get(head_key, [head_key]), head_key)
            tail_surface = _choose_alias(entity_alias_map.get(tail_key, [tail_key]), tail_key)
            relation = rel_map.get(rel_base[:-1] if rel_base.endswith("r") else rel_base, rel_base)
            edges.append(
                {
                    "source": head_surface,
                    "relation": relation,
                    "target": tail_surface,
                }
            )
            if head_tag == "A":
                answers.append(head_surface)
            if tail_tag == "A":
                answers.append(tail_surface)
        if not answers and edges:
            answers.append(edges[-1]["target"])
        items.append(
            {
                "question_id": offset + idx,
                "question": tgt_line.strip(),
                "answers": list(dict.fromkeys(answers)),
                "answer_text": list(dict.fromkeys(answers)),
                "paraphrases": [tgt_line.strip()],
                "subgraph": {
                    "edges": edges,
                    "edge_count": len(edges),
                },
                "source_text": src_line.strip(),
                "dataset": "CWQ",
                "split": split,
            }
        )
    return items


def _parse_pq_line(src_line: str) -> tuple[list[dict[str, str]], list[str]]:
    tokens = [token.strip() for token in src_line.split() if token.strip()]
    edges = []
    for start in range(0, max(len(tokens) - 2, 0), 2):
        head = _clean_surface(tokens[start])
        relation = _clean_surface(tokens[start + 1])
        tail = _clean_surface(tokens[start + 2])
        edges.append({"source": head, "relation": relation, "target": tail})
    answers = [edges[-1]["target"]] if edges else []
    return edges, answers


def _parse_pq_split(src_path: Path, tgt_path: Path, split: str, dataset_name: str) -> list[dict[str, Any]]:
    src_lines = _read_lines(src_path)
    tgt_lines = _read_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        print(
            f"[warn] {split}: line count mismatch {len(src_lines)} vs {len(tgt_lines)}; "
            f"truncating to {min(len(src_lines), len(tgt_lines))}"
        )
    n = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:n]
    tgt_lines = tgt_lines[:n]

    items: list[dict[str, Any]] = []
    for idx, (src_line, tgt_line) in enumerate(
        tqdm(zip(src_lines, tgt_lines), total=n, desc=f"{split} {dataset_name}", leave=False)
    ):
        edges, answers = _parse_pq_line(src_line)
        items.append(
            {
                "question_id": idx,
                "question": tgt_line.strip(),
                "answers": list(dict.fromkeys(answers)),
                "answer_text": list(dict.fromkeys(answers)),
                "paraphrases": [tgt_line.strip()],
                "subgraph": {
                    "edges": edges,
                    "edge_count": len(edges),
                },
                "source_text": src_line.strip(),
                "dataset": dataset_name,
                "split": split,
            }
        )
    return items


def _write_split(items: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def _sgsh_edges(item: dict[str, Any]) -> list[dict[str, str]]:
    graph = item.get("inGraph") or {}
    names = graph.get("g_node_names") or {}
    edge_types = graph.get("g_edge_types") or {}
    adj = graph.get("g_adj") or {}
    edges: list[dict[str, str]] = []
    for source_id, targets in adj.items():
        for target_id, rels in targets.items():
            rel_list = rels if isinstance(rels, list) else [rels]
            for rel in rel_list:
                edges.append(
                    {
                        "source": _clean_surface(names.get(source_id, source_id)),
                        "relation": _clean_surface(edge_types.get(rel, rel).replace("/", " ")),
                        "target": _clean_surface(names.get(target_id, target_id)),
                    }
                )
    return edges


def _prepare_sgsh_split(path: Path, split: str, dataset_name: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    items: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"{split} {dataset_name}", leave=False):
        edges = _sgsh_edges(row)
        answers = [_clean_surface(value) for value in row.get("answers", []) if _clean_surface(value)]
        question = _clean_surface(row.get("outSeq", ""))
        items.append(
            {
                "question_id": row.get("qId"),
                "question": question,
                "answers": answers,
                "answer_text": answers,
                "paraphrases": [question],
                "subgraph": {
                    "edges": edges,
                    "edge_count": len(edges),
                },
                "source_text": json.dumps(row.get("inGraph", {}), ensure_ascii=False, sort_keys=True),
                "dataset": dataset_name,
                "split": split,
            }
        )
    return items


def prepare_cwq(out_root: Path) -> dict[str, int]:
    src_root = ROOT / "org_data" / "mhqg" / "final_data" / "cwq" / "ae+de"
    rel_map = _load_rel_dict(ROOT / "org_data" / "mhqg" / "final_data" / "cwq" / "common_data" / "rel_dict.txt")
    ent_dict_list = _load_ent_dict_list(src_root / "ent_dict_list.txt")

    offsets = {"train": 0}
    train_count = len(_read_lines(src_root / "train.src"))
    dev_count = len(_read_lines(src_root / "dev.src"))
    offsets["dev"] = train_count
    offsets["test"] = train_count + dev_count

    summary: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        items = _parse_cwq_split(
            src_root / f"{split}.src",
            src_root / f"{split}.tgt",
            rel_map,
            ent_dict_list,
            offsets[split],
            split,
        )
        _write_split(items, out_root / "CWQ" / f"{split}.json")
        summary[split] = len(items)
    return summary


def prepare_pq(out_root: Path) -> dict[str, int]:
    src_root = ROOT / "SGSH" / "dataset" / "PQ"
    summary: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        items = _prepare_sgsh_split(src_root / f"{split}.json", split, "PQ")
        _write_split(items, out_root / "PQ" / f"{split}.json")
        summary[split] = len(items)
    return summary


def prepare_wq(out_root: Path) -> dict[str, int]:
    src_root = ROOT / "SGSH" / "dataset" / "WQ"
    summary: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        items = _prepare_sgsh_split(src_root / f"{split}.json", split, "WQ")
        _write_split(items, out_root / "WQ" / f"{split}.json")
        summary[split] = len(items)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ChronoSynth KGQG data for WQ/CWQ and PQ.")
    parser.add_argument("--datasets", nargs="+", default=["WQ", "PQ"], choices=["WQ", "CWQ", "PQ"])
    parser.add_argument("--out_root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    summary: dict[str, dict[str, int]] = {}
    if "WQ" in args.datasets:
        summary["WQ"] = prepare_wq(args.out_root)
    if "CWQ" in args.datasets:
        summary["CWQ"] = prepare_cwq(args.out_root)
    if "PQ" in args.datasets:
        summary["PQ"] = prepare_pq(args.out_root)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
