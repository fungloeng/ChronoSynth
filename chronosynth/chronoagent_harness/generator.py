from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from .api import build_client, call_text_generation
from .core import (
    build_operator_state,
    build_memory_bank,
    build_memory_bank_cache_metadata,
    build_prompt,
    load_memory_bank,
    save_memory_bank,
)
from .data import first_reference, get_references, guess_dataset_name, load_json, sanitize_eval_item
from .progress import ProgressTracker


def _clean_question(text: str) -> str:
    text = " ".join(str(text).strip().split())
    if not text:
        return ""
    if text.lower().startswith("question:"):
        text = text.split(":", 1)[1].strip()
    if "\n" in text:
        text = text.splitlines()[-1].strip()
    return text if text.endswith("?") else f"{text}?"


@dataclass
class StaticPatternExample:
    prototype: str
    exact_signature: str
    relation_signature: str
    edge_count: int
    entities: list[str]
    question: str
    edges: list[tuple[str, str, str]]


@dataclass
class StaticSkeletonPlanner:
    dataset: str
    train_items: list[dict]
    train_skeletons: list[str]
    test_skeletons: list[str]
    similarity_idx: list[list[int]]


@dataclass
class StaticBaselinePrototype:
    question: str
    relation_signature: str
    edge_count: int


class StaticPatternIndex:
    def __init__(self, dataset: str, examples: list[StaticPatternExample]):
        self.dataset = dataset
        self.examples = examples
        self.by_exact: dict[str, list[StaticPatternExample]] = {}
        self.by_relation: dict[tuple[str, int], list[StaticPatternExample]] = {}
        self.by_single_relation: dict[str, list[StaticPatternExample]] = {}
        for ex in examples:
            self.by_exact.setdefault(ex.exact_signature, []).append(ex)
            self.by_relation.setdefault((ex.relation_signature, ex.edge_count), []).append(ex)
            for relation in {rel for _, rel, _ in ex.edges}:
                self.by_single_relation.setdefault(relation, []).append(ex)


def _normalize_space(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split()).strip()


def _static_question_type(text: str) -> str:
    text = str(text).strip().lower()
    for prefix in ("who", "what", "where", "when", "which", "how", "is", "are"):
        if text.startswith(prefix):
            return prefix
    parts = text.split()
    return parts[0] if parts else ""


def _extract_static_edges(item: dict, dataset: str) -> list[tuple[str, str, str]]:
    edges = []
    for edge in (item.get("subgraph") or {}).get("edges", []) or []:
        source = _normalize_space(edge.get("source", ""))
        relation = _normalize_space(edge.get("relation", "")).lower()
        target = _normalize_space(edge.get("target", ""))
        if source or relation or target:
            edges.append((source, relation, target))
    edges.sort()
    return edges


def _canonicalize_edges(edges: list[tuple[str, str, str]], entity_prefix: str = "__ENT") -> tuple[str, str, list[str]]:
    entity_map: dict[str, str] = {}
    ordered_entities: list[str] = []

    def canon_entity(value: str) -> str:
        key = value.lower()
        if key not in entity_map:
            entity_map[key] = f"{entity_prefix}{len(entity_map) + 1}__"
            ordered_entities.append(value)
        return entity_map[key]

    canonical_edges = []
    relation_names = []
    for source, relation, target in edges:
        canonical_edges.append((canon_entity(source), relation, canon_entity(target)))
        relation_names.append(relation)
    exact_signature = " | ".join(f"{s} -- {r} --> {t}" for s, r, t in canonical_edges)
    relation_signature = " | ".join(sorted(relation_names))
    return exact_signature, relation_signature, ordered_entities


def _delexicalize_static_question(question: str, entities: list[str]) -> str:
    text = str(question).strip()
    for idx, entity in sorted(enumerate(entities, start=1), key=lambda pair: len(pair[1]), reverse=True):
        entity = str(entity).strip()
        if not entity or entity.lower() == "none":
            continue
        pattern = re.compile(re.escape(entity), flags=re.IGNORECASE)
        text = pattern.sub(f"__ENT{idx}__", text)
    return " ".join(text.split()).strip()


def _adapt_static_prototype(prototype: str, entities: list[str]) -> str:
    text = prototype
    for idx, entity in enumerate(entities, start=1):
        text = text.replace(f"__ENT{idx}__", entity)
    return _clean_question(text)


def _replace_entities_with_mapping(question: str, mapping: dict[str, str]) -> str:
    text = str(question).strip()
    ordered = sorted(
        ((src, dst) for src, dst in mapping.items() if src and dst),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for source, target in ordered:
        pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
        text = pattern.sub(target, text)
    return _clean_question(text)


def _node_signature(edges: list[tuple[str, str, str]], node: str) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    outgoing = sorted(relation for source, relation, _ in edges if source == node)
    incoming = sorted(relation for _, relation, target in edges if target == node)
    return (tuple(outgoing), tuple(incoming), len(outgoing) + len(incoming))


def _find_graph_mapping(
    source_edges: list[tuple[str, str, str]],
    target_edges: list[tuple[str, str, str]],
) -> dict[str, str] | None:
    source_nodes = sorted({node for edge in source_edges for node in (edge[0], edge[2])})
    target_nodes = sorted({node for edge in target_edges for node in (edge[0], edge[2])})
    if len(source_nodes) != len(target_nodes):
        return None

    source_edge_set = {(s, r, t) for s, r, t in source_edges}
    target_edge_set = {(s, r, t) for s, r, t in target_edges}
    source_sig = {node: _node_signature(source_edges, node) for node in source_nodes}
    target_sig = {node: _node_signature(target_edges, node) for node in target_nodes}

    ordered_sources = sorted(source_nodes, key=lambda node: source_sig[node], reverse=True)
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()

    def backtrack(index: int) -> bool:
        if index == len(ordered_sources):
            mapped_edges = {(mapping[s], r, mapping[t]) for s, r, t in source_edge_set}
            return mapped_edges == target_edge_set

        source_node = ordered_sources[index]
        candidates = [node for node in target_nodes if target_sig[node] == source_sig[source_node] and node not in used_targets]
        if not candidates:
            candidates = [node for node in target_nodes if node not in used_targets]

        for target_node in candidates:
            mapping[source_node] = target_node
            used_targets.add(target_node)

            partial_ok = True
            for source, relation, target in source_edge_set:
                if source in mapping and target in mapping:
                    if (mapping[source], relation, mapping[target]) not in target_edge_set:
                        partial_ok = False
                        break
            if partial_ok and backtrack(index + 1):
                return True

            used_targets.remove(target_node)
            del mapping[source_node]
        return False

    return dict(mapping) if backtrack(0) else None


def _build_static_index(train_data: list[dict], dataset: str) -> StaticPatternIndex:
    examples: list[StaticPatternExample] = []
    for item in train_data:
        edges = _extract_static_edges(item, dataset)
        if not edges:
            continue
        exact_signature, relation_signature, entities = _canonicalize_edges(edges)
        question = str(item.get("question") or "").strip()
        if not question:
            continue
        prototype = _delexicalize_static_question(question, entities)
        examples.append(
            StaticPatternExample(
                prototype=prototype,
                exact_signature=exact_signature,
                relation_signature=relation_signature,
                edge_count=len(edges),
                entities=entities,
                question=question,
                edges=edges,
            )
        )
    return StaticPatternIndex(dataset=dataset, examples=examples)


def _graph_fact_lines(item: dict) -> list[str]:
    lines = []
    for source, relation, target in _extract_static_edges(item, str(item.get("dataset") or "")):
        lines.append(f"<{source}, {relation}, {target}>")
    return lines


def _static_answers(raw_item: dict) -> list[str]:
    values = raw_item.get("answer_text") or raw_item.get("answers") or []
    return [_normalize_space(value) for value in values if _normalize_space(value)]


def _static_answer_type(raw_item: dict) -> str:
    answers = _static_answers(raw_item)
    if not answers:
        return "unknown"
    if all(re.fullmatch(r"\d{4}([-/]\d{1,2}([-/]\d{1,2})?)?", answer) for answer in answers):
        return "date"
    if all(answer.replace(".", "", 1).isdigit() for answer in answers):
        return "number"
    return "entity"


def _build_static_structured_prompt(raw_item: dict) -> str:
    return "\n".join(
        [
            f"You generate one {raw_item.get('dataset', 'KGQG')}-style question from a knowledge-graph subgraph.",
            "Use the triples faithfully.",
            "Do not reveal the answer string in the question.",
            "Return only the final question.",
            "",
            "Triples:",
            "\n".join(_graph_fact_lines(raw_item)),
            f"Answer for planning only: {', '.join(_static_answers(raw_item))}",
            "Question:",
        ]
    )


def _build_static_state_prompt(raw_item: dict) -> str:
    edges = _extract_static_edges(raw_item, str(raw_item.get("dataset") or ""))
    relations = sorted({relation for _, relation, _ in edges})
    return "\n".join(
        [
            f"You generate one {raw_item.get('dataset', 'KGQG')}-style question from a knowledge-graph subgraph.",
            "Use the intent state to preserve what the question should ask.",
            "Do not reveal the answer string in the question.",
            "Return only the final question.",
            "",
            "Intent state:",
            f"- answer_type: {_static_answer_type(raw_item)}",
            f"- edge_count: {len(edges)}",
            f"- relation_pattern: {' | '.join(relations) if relations else 'none'}",
            "",
            "Triples:",
            "\n".join(_graph_fact_lines(raw_item)),
            f"Answer for planning only: {', '.join(_static_answers(raw_item))}",
            "Question:",
        ]
    )


def _load_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _build_static_skeleton_planner(dataset: str, train_data: list[dict]) -> StaticSkeletonPlanner | None:
    root = Path(__file__).resolve().parents[2] / "SGSH" / "dataset" / dataset
    train_skeleton_path = root / "train_skeleton.txt"
    test_skeleton_path = root / "predict_test_skeleton.txt"
    similarity_path = root / "similarity_by_embeddings_and_skeleton_top_16_idx.npy"
    if not (train_skeleton_path.exists() and test_skeleton_path.exists() and similarity_path.exists()):
        return None
    import numpy as np

    train_skeletons = _load_text_lines(train_skeleton_path)
    test_skeletons = _load_text_lines(test_skeleton_path)
    similarity_idx = np.load(similarity_path, allow_pickle=True).tolist()
    return StaticSkeletonPlanner(
        dataset=dataset,
        train_items=train_data,
        train_skeletons=train_skeletons,
        test_skeletons=test_skeletons,
        similarity_idx=similarity_idx,
    )


def _target_skeleton_type(
    skeleton_planner: StaticSkeletonPlanner | None,
    sample_index: int | None,
) -> str | None:
    if skeleton_planner is None or sample_index is None:
        return None
    if sample_index < 0 or sample_index >= len(skeleton_planner.test_skeletons):
        return None
    return _static_question_type(skeleton_planner.test_skeletons[sample_index])


def _build_static_skeleton_prompt(
    raw_item: dict,
    sample_index: int,
    planner: StaticSkeletonPlanner,
    fewshot_k: int = 4,
    candidate_question: str | None = None,
) -> str:
    skeleton = planner.test_skeletons[sample_index]
    fewshot_indices = planner.similarity_idx[sample_index][:fewshot_k]
    sections = [
        f"You generate one {planner.dataset}-style question from a knowledge-graph subgraph.",
        "Use the provided skeleton as the outer wording plan.",
        "Keep entity names faithful to the graph.",
        "Preserve the relation semantics in the triples.",
        "Do not reveal the answer string in the question.",
        "Return only the final question.",
        "",
    ]
    for rank, train_idx in enumerate(fewshot_indices, start=1):
        train_item = planner.train_items[train_idx]
        train_skeleton = planner.train_skeletons[train_idx]
        sections.extend(
            [
                f"[Example {rank}]",
                "Triples:",
                "\n".join(_graph_fact_lines(train_item)),
                f"Answer: {', '.join(train_item.get('answer_text') or train_item.get('answers') or [])}",
                f"Skeleton: {train_skeleton}",
                f"Question: {train_item.get('question', '').strip()}",
                "",
            ]
        )
    sections.extend(
        [
            "[Target]",
            "Triples:",
            "\n".join(_graph_fact_lines(raw_item)),
            f"Answer for planning only: {', '.join(raw_item.get('answer_text') or raw_item.get('answers') or [])}",
            f"Skeleton: {skeleton}",
            (
                f"Prototype draft: {candidate_question}"
                if candidate_question
                else "Prototype draft: none"
            ),
            "Use the prototype draft only when it matches the graph facts and the skeleton.",
            "Question:",
        ]
    )
    return "\n".join(section for section in sections if section is not None)


def _build_static_refine_prompt(
    raw_item: dict,
    skeleton: str,
    draft_question: str,
    candidate_question: str | None = None,
) -> str:
    sections = [
        "Revise the draft question so that it exactly matches the knowledge-graph facts.",
        "Keep the wording natural and concise.",
        "Follow the skeleton style when possible.",
        "Do not reveal the answer string in the final question.",
        "Return only the revised final question.",
        "",
        "Triples:",
        "\n".join(_graph_fact_lines(raw_item)),
        f"Answer for planning only: {', '.join(raw_item.get('answer_text') or raw_item.get('answers') or [])}",
        f"Skeleton: {skeleton}",
    ]
    if candidate_question:
        sections.append(f"Prototype draft: {candidate_question}")
    sections.extend(
        [
            f"Current draft: {draft_question}",
            "Final question:",
        ]
    )
    return "\n".join(sections)


def _build_static_merge_prompt(
    raw_item: dict,
    skeleton: str,
    draft_question: str,
    prototype_question: str,
) -> str:
    return "\n".join(
        [
            f"You refine one {raw_item.get('dataset', 'KGQG')}-style question from a knowledge-graph subgraph.",
            "You are given two candidate questions.",
            "Candidate A usually matches the target skeleton.",
            "Candidate B usually stays closer to the graph structure.",
            "Write one final question that keeps the graph facts correct and keeps the wording natural.",
            "Prefer the skeleton style of Candidate A unless it conflicts with the graph facts.",
            "Do not reveal the answer string in the final question.",
            "Return only the final question.",
            "",
            "Triples:",
            "\n".join(_graph_fact_lines(raw_item)),
            f"Answer for planning only: {', '.join(raw_item.get('answer_text') or raw_item.get('answers') or [])}",
            f"Skeleton: {skeleton}",
            f"Candidate A: {_clean_question(draft_question)}",
            f"Candidate B: {_clean_question(prototype_question)}",
            "Final question:",
        ]
    )


def _build_static_polish_prompt(
    raw_item: dict,
    draft_question: str,
) -> str:
    return "\n".join(
        [
            f"You lightly rewrite one {raw_item.get('dataset', 'KGQG')}-style question.",
            "Keep the same graph meaning and answer target.",
            "Keep the relation path and entity roles unchanged.",
            "Only improve wording, remove awkward repetition, and keep the final question concise.",
            "Do not reveal the answer string in the final question.",
            "Return only the final question.",
            "",
            "Triples:",
            "\n".join(_graph_fact_lines(raw_item)),
            f"Answer for planning only: {', '.join(raw_item.get('answer_text') or raw_item.get('answers') or [])}",
            f"Draft question: {_clean_question(draft_question)}",
            "Final question:",
        ]
    )


def _static_similarity(
    target_exact: str,
    target_relation: str,
    target_edges: int,
    candidate: StaticPatternExample,
) -> float:
    score = 0.0
    if candidate.exact_signature == target_exact:
        score += 100.0
    if candidate.relation_signature == target_relation:
        score += 10.0
    target_rel_set = set(target_relation.split(" | ")) if target_relation else set()
    cand_rel_set = set(candidate.relation_signature.split(" | ")) if candidate.relation_signature else set()
    if target_rel_set or cand_rel_set:
        score += 5.0 * (len(target_rel_set & cand_rel_set) / max(len(target_rel_set | cand_rel_set), 1))
    score -= abs(candidate.edge_count - target_edges) * 0.5
    return score


def _rank_static_examples(
    item: dict,
    static_index: StaticPatternIndex | None,
    top_k: int = 4,
    relation_only: bool = False,
) -> list[StaticPatternExample]:
    if static_index is None:
        return []
    edges = _extract_static_edges(item, str(item.get("dataset") or ""))
    if not edges:
        return []
    exact_signature, relation_signature, _ = _canonicalize_edges(edges)
    if relation_only:
        candidates = list(static_index.by_relation.get((relation_signature, len(edges)), []))
    else:
        candidates = list(static_index.by_exact.get(exact_signature, []))
        if not candidates:
            candidates = list(static_index.by_relation.get((relation_signature, len(edges)), []))
    if not candidates:
        overlap_candidates: list[StaticPatternExample] = []
        seen_ids: set[int] = set()
        for relation in {rel for _, rel, _ in edges}:
            for example in static_index.by_single_relation.get(relation, []):
                marker = id(example)
                if marker not in seen_ids:
                    overlap_candidates.append(example)
                    seen_ids.add(marker)
        candidates = overlap_candidates
    if not candidates:
        candidates = static_index.examples[:]
    ranked = sorted(
        candidates,
        key=lambda ex: _static_similarity(exact_signature, relation_signature, len(edges), ex),
        reverse=True,
    )
    return ranked[:top_k]


def _sample_static_examples(
    static_index: StaticPatternIndex | None,
    raw_item: dict,
    sample_index: int | None,
    top_k: int = 4,
) -> list[StaticPatternExample]:
    if static_index is None or not static_index.examples:
        return []
    start = (sample_index or 0) % len(static_index.examples)
    selected: list[StaticPatternExample] = []
    forbidden_question = str(raw_item.get("question") or "").strip().lower()
    for offset in range(len(static_index.examples)):
        example = static_index.examples[(start + offset) % len(static_index.examples)]
        if example.question.strip().lower() == forbidden_question:
            continue
        selected.append(example)
        if len(selected) >= top_k:
            break
    return selected


def _prototype_to_baseline(example: StaticPatternExample, raw_item: dict) -> StaticBaselinePrototype:
    return StaticBaselinePrototype(
        question=_adapt_static_prototype(example.prototype, _canonicalize_edges(_extract_static_edges(raw_item, str(raw_item.get("dataset") or "")))[2]),
        relation_signature=example.relation_signature,
        edge_count=example.edge_count,
    )


def _build_static_prototype_prompt(
    raw_item: dict,
    prototypes: list[StaticBaselinePrototype],
    prompt_name: str,
) -> str:
    sections = [
        f"You generate one {raw_item.get('dataset', 'KGQG')}-style question from a knowledge-graph subgraph.",
        f"Use the provided {prompt_name} examples only as wording guidance.",
        "Keep the graph semantics faithful.",
        "Do not reveal the answer string in the question.",
        "Return only the final question.",
        "",
    ]
    for idx, prototype in enumerate(prototypes, start=1):
        sections.extend(
            [
                f"[Prototype {idx}]",
                f"Question pattern: {prototype.question}",
                f"Relation pattern: {prototype.relation_signature}",
                f"Edge count: {prototype.edge_count}",
                "",
            ]
        )
    sections.extend(
        [
            "[Target]",
            "Triples:",
            "\n".join(_graph_fact_lines(raw_item)),
            f"Answer for planning only: {', '.join(_static_answers(raw_item))}",
            "Question:",
        ]
    )
    return "\n".join(sections)


def _posthoc_validate_static_question(raw_item: dict, question: str, fallback: str | None = None) -> str:
    clean = _clean_question(question)
    lower = clean.lower()
    for answer in _static_answers(raw_item):
        if answer and answer.lower() in lower:
            return fallback or "What is the correct question?"
    if len(clean.split()) < 3:
        return fallback or clean
    return clean


def _generate_static_by_pattern(
    item: dict,
    dataset: str,
    static_index: StaticPatternIndex | None,
    skeleton_planner: StaticSkeletonPlanner | None = None,
    sample_index: int | None = None,
    prefer_short_question: bool = False,
) -> str | None:
    if static_index is None:
        return None
    edges = _extract_static_edges(item, dataset)
    if not edges:
        return None
    exact_signature, relation_signature, entities = _canonicalize_edges(edges)
    target_type = _target_skeleton_type(skeleton_planner, sample_index)
    exact_hits = static_index.by_exact.get(exact_signature)
    if exact_hits:
        ranked_exact = sorted(exact_hits, key=lambda ex: len(ex.prototype), reverse=True)
        if target_type:
            typed = [ex for ex in ranked_exact if _static_question_type(ex.question) == target_type]
            if typed:
                ranked_exact = typed
        if prefer_short_question:
            ranked_exact = sorted(ranked_exact, key=lambda ex: (len(ex.question.split()), len(ex.prototype)))
        best = ranked_exact[0]
        mapping = _find_graph_mapping(best.edges, edges)
        if mapping:
            return _replace_entities_with_mapping(best.question, mapping)
        return _adapt_static_prototype(best.prototype, entities)
    relation_hits = static_index.by_relation.get((relation_signature, len(edges)), [])
    candidates = relation_hits
    if not candidates:
        overlap_candidates: list[StaticPatternExample] = []
        seen_ids: set[int] = set()
        for relation in {rel for _, rel, _ in edges}:
            for example in static_index.by_single_relation.get(relation, []):
                marker = id(example)
                if marker not in seen_ids:
                    overlap_candidates.append(example)
                    seen_ids.add(marker)
        candidates = overlap_candidates[:256]
    if not candidates:
        candidates = static_index.examples[:256]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda ex: _static_similarity(exact_signature, relation_signature, len(edges), ex),
        reverse=True,
    )[:32]
    if target_type:
        typed = [ex for ex in ranked if _static_question_type(ex.question) == target_type]
        if typed:
            ranked = typed
    if prefer_short_question:
        ranked = sorted(ranked, key=lambda ex: (len(ex.question.split()), len(ex.prototype)))
    for best in ranked:
        mapping = _find_graph_mapping(best.edges, edges)
        if mapping:
            return _replace_entities_with_mapping(best.question, mapping)
    return _adapt_static_prototype(ranked[0].prototype, entities)


def _generate_static_exact_pattern(item: dict, dataset: str, static_index: StaticPatternIndex | None) -> str | None:
    if static_index is None:
        return None
    edges = _extract_static_edges(item, dataset)
    if not edges:
        return None
    exact_signature, _, entities = _canonicalize_edges(edges)
    exact_hits = static_index.by_exact.get(exact_signature)
    if not exact_hits:
        return None
    best = max(exact_hits, key=lambda ex: len(ex.prototype))
    mapping = _find_graph_mapping(best.edges, edges)
    if mapping:
        return _replace_entities_with_mapping(best.question, mapping)
    return _adapt_static_prototype(best.prototype, entities)


def _prepare_memory_bank(
    train_path: str | Path,
    dataset: str,
    memory_bank_cache: str | Path | None = None,
):
    cache_path = Path(memory_bank_cache) if memory_bank_cache else None
    metadata = build_memory_bank_cache_metadata(train_path, dataset)
    if cache_path is not None:
        cached_bank = load_memory_bank(cache_path, expected_metadata=metadata)
        if cached_bank is not None:
            print(f"Loaded memory bank cache from {cache_path}")
            return cached_bank
        if cache_path.exists():
            print(f"Memory bank cache at {cache_path} is stale or unreadable; rebuilding")

    train_data = load_json(train_path)
    memory_bank = build_memory_bank(train_data, dataset=dataset)
    if cache_path is not None:
        save_memory_bank(memory_bank, cache_path, metadata=metadata)
        print(f"Saved memory bank cache to {cache_path}")
    return memory_bank


def generate_one(
    item: dict,
    raw_item: dict,
    memory_bank,
    dataset: str,
    client,
    model: str,
    static_index: StaticPatternIndex | None = None,
    skeleton_planner: StaticSkeletonPlanner | None = None,
    sample_index: int | None = None,
    include_prompt: bool = False,
    ablation: str = "full",
    api_mode: str = "responses",
    reasoning_effort: str | None = None,
    enable_thinking: bool = False,
    static_strategy: str = "transfer_v1",
    baseline_mode: str = "chronosynth",
) -> dict:
    if dataset in {"WQ", "PQ", "CWQ"} and baseline_mode != "chronosynth":
        fallback_question = _generate_static_by_pattern(raw_item, dataset, static_index)
        prompt = None
        if baseline_mode == "structured_prompt":
            prompt = _build_static_structured_prompt(raw_item)
        elif baseline_mode == "operatorstate_prompt":
            prompt = _build_static_state_prompt(raw_item)
        elif baseline_mode == "random_prototype_prompt":
            examples = [
                _prototype_to_baseline(example, raw_item)
                for example in _sample_static_examples(static_index, raw_item, sample_index, top_k=4)
            ]
            prompt = _build_static_prototype_prompt(raw_item, examples, "random prototype")
        elif baseline_mode == "relation_only_retrieval":
            examples = [
                _prototype_to_baseline(example, raw_item)
                for example in _rank_static_examples(raw_item, static_index, top_k=4, relation_only=True)
            ]
            prompt = _build_static_prototype_prompt(raw_item, examples, "relation-matched prototype")
        elif baseline_mode == "prototype_icl":
            examples = [
                _prototype_to_baseline(example, raw_item)
                for example in _rank_static_examples(raw_item, static_index, top_k=4, relation_only=False)
            ]
            prompt = _build_static_prototype_prompt(raw_item, examples, "retrieved prototype")
        elif baseline_mode == "direct_prompt_verifier":
            prompt = _build_static_structured_prompt(raw_item)
        else:
            raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")

        question = call_text_generation(
            prompt,
            model=model,
            client=client,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )
        clean_question = _clean_question(question)
        if baseline_mode == "direct_prompt_verifier":
            clean_question = _posthoc_validate_static_question(
                raw_item,
                clean_question,
                fallback=fallback_question or clean_question,
            )
        result = {
            "question_id": raw_item.get("question_id"),
            "generated_question": clean_question,
            "reference_questions": get_references(raw_item),
        }
        if include_prompt:
            result["prompt"] = prompt
        return result

    if dataset == "WQ" and static_strategy == "sota_v3" and ablation == "full":
        exact_question = _generate_static_exact_pattern(raw_item, dataset, static_index)
        if exact_question:
            result = {
                "question_id": raw_item.get("question_id"),
                "generated_question": exact_question,
                "reference_questions": get_references(raw_item),
            }
            if include_prompt:
                result["prompt"] = "[static-prototype] exact pattern reuse"
            return result

    if dataset == "WQ" and skeleton_planner is not None and sample_index is not None and ablation == "full":
        candidate_question = None
        if static_strategy in {"transfer_v1", "transfer_v1b", "sota_v2", "sota_v4"}:
            candidate_question = _generate_static_by_pattern(
                raw_item,
                dataset,
                static_index,
                skeleton_planner=skeleton_planner,
                sample_index=sample_index,
                prefer_short_question=True,
            )
        prompt = _build_static_skeleton_prompt(
            raw_item,
            sample_index=sample_index,
            planner=skeleton_planner,
            candidate_question=candidate_question,
        )
        question = call_text_generation(
            prompt,
            model=model,
            client=client,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )
        if static_strategy == "sota_v2":
            refine_prompt = _build_static_refine_prompt(
                raw_item=raw_item,
                skeleton=skeleton_planner.test_skeletons[sample_index],
                draft_question=_clean_question(question),
                candidate_question=candidate_question,
            )
            question = call_text_generation(
                refine_prompt,
                model=model,
                client=client,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
        elif static_strategy == "sota_v4" and candidate_question:
            merge_prompt = _build_static_merge_prompt(
                raw_item=raw_item,
                skeleton=skeleton_planner.test_skeletons[sample_index],
                draft_question=_clean_question(question),
                prototype_question=candidate_question,
            )
            question = call_text_generation(
                merge_prompt,
                model=model,
                client=client,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
        result = {
            "question_id": raw_item.get("question_id"),
            "generated_question": _clean_question(question),
            "reference_questions": get_references(raw_item),
        }
        if include_prompt:
            result["prompt"] = prompt
        return result

    if dataset in {"PQ", "CWQ"} and ablation == "full" and static_strategy in {"transfer_v1", "transfer_v1b", "sota_v3"}:
        static_question = _generate_static_by_pattern(
            raw_item,
            dataset,
            static_index,
            skeleton_planner=skeleton_planner,
            sample_index=sample_index,
            prefer_short_question=(dataset == "PQ"),
        )
        if static_question:
            result = {
                "question_id": raw_item.get("question_id"),
                "generated_question": static_question,
                "reference_questions": get_references(raw_item),
            }
            if include_prompt:
                result["prompt"] = "[static-prototype] direct pattern adaptation"
            return result

    if dataset == "PQ" and ablation == "full" and static_strategy == "sota_v4":
        static_question = _generate_static_by_pattern(raw_item, dataset, static_index)
        if static_question:
            polish_prompt = _build_static_polish_prompt(raw_item, static_question)
            polished = call_text_generation(
                polish_prompt,
                model=model,
                client=client,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
            result = {
                "question_id": raw_item.get("question_id"),
                "generated_question": _posthoc_validate_static_question(
                    raw_item,
                    polished,
                    fallback=static_question,
                ),
                "reference_questions": get_references(raw_item),
            }
            if include_prompt:
                result["prompt"] = polish_prompt
            return result

    if dataset == "PQ" and skeleton_planner is not None and sample_index is not None and ablation == "full":
        candidate_question = _generate_static_by_pattern(raw_item, dataset, static_index)
        prompt = _build_static_skeleton_prompt(
            raw_item,
            sample_index=sample_index,
            planner=skeleton_planner,
            candidate_question=candidate_question,
        )
        question = call_text_generation(
            prompt,
            model=model,
            client=client,
            api_mode=api_mode,
            reasoning_effort=reasoning_effort,
            enable_thinking=enable_thinking,
        )
        if static_strategy == "sota_v2":
            refine_prompt = _build_static_refine_prompt(
                raw_item=raw_item,
                skeleton=skeleton_planner.test_skeletons[sample_index],
                draft_question=_clean_question(question),
                candidate_question=candidate_question,
            )
            question = call_text_generation(
                refine_prompt,
                model=model,
                client=client,
                api_mode=api_mode,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
            )
        result = {
            "question_id": raw_item.get("question_id"),
            "generated_question": _clean_question(question),
            "reference_questions": get_references(raw_item),
        }
        if include_prompt:
            result["prompt"] = prompt
        return result

    operator_state = build_operator_state(item, dataset=dataset)
    retrieval_mode = "full"
    use_drafts = True
    validate_placeholders = True
    validate_leakage = True
    use_default_templates = True
    enforce_temporal_rule = True
    if ablation == "no_memory":
        retrieval_mode = "none"
    elif ablation == "relation_only":
        retrieval_mode = "relation_only"
    elif ablation == "no_drafts":
        use_drafts = False
    elif ablation == "no_draft_filtering":
        validate_placeholders = False
        validate_leakage = False
    elif ablation == "no_leakage_check":
        validate_leakage = False
    elif ablation == "no_default_templates":
        use_default_templates = False
    elif ablation == "no_temporal_rule":
        enforce_temporal_rule = False
    prompt = build_prompt(
        operator_state,
        memory_bank,
        retrieval_mode=retrieval_mode,
        use_drafts=use_drafts,
        validate_placeholders=validate_placeholders,
        validate_leakage=validate_leakage,
        use_default_templates=use_default_templates,
        enforce_temporal_rule=enforce_temporal_rule,
    )
    question = call_text_generation(
        prompt,
        model=model,
        client=client,
        api_mode=api_mode,
        reasoning_effort=reasoning_effort,
        enable_thinking=enable_thinking,
    )
    result = {
        "question_id": raw_item.get("question_id"),
        "generated_question": _clean_question(question),
        "reference_questions": get_references(raw_item),
    }
    if include_prompt:
        result["prompt"] = prompt
        result["operator_state"] = {
            "operator": operator_state.operator,
            "ask_slot": operator_state.ask_slot,
            "focus_relation": operator_state.focus_relation,
            "focus_summary": operator_state.focus_summary,
            "anchor_entity": operator_state.anchor_entity,
            "comparator_entity": operator_state.comparator_entity,
            "comparator_time": operator_state.comparator_time or operator_state.constraint_value,
        }
    return result


def run_generation(
    train_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    model: str = "gpt-4o-mini",
    workers: int = 20,
    shard_size: int = 1000,
    save_every: int = 100,
    resume: bool = True,
    include_prompt: bool = False,
    memory_bank_cache: str | Path | None = None,
    ablation: str = "full",
    api_key: str | None = None,
    base_url: str | None = None,
    api_mode: str = "responses",
    reasoning_effort: str | None = None,
    enable_thinking: bool = False,
    retry_errors: bool = True,
    dataset: str | None = None,
    static_strategy: str = "transfer_v1",
    baseline_mode: str = "chronosynth",
) -> str:
    dataset = dataset or guess_dataset_name(path=str(input_path))
    eval_data = load_json(input_path)
    sanitized = [sanitize_eval_item(item) for item in eval_data]
    train_data = load_json(train_path)
    memory_bank = _prepare_memory_bank(
        train_path=train_path,
        dataset=dataset,
        memory_bank_cache=memory_bank_cache,
    )
    static_index = _build_static_index(train_data, dataset) if dataset in {"WQ", "PQ", "CWQ"} else None
    skeleton_planner = (
        _build_static_skeleton_planner(dataset, train_data)
        if dataset in {"WQ", "PQ"} and ablation == "full"
        else None
    )
    needs_client = (
        baseline_mode != "chronosynth"
        or dataset in {"WQ", "PQ"}
        or static_index is None
        or ablation != "full"
        or static_strategy in {"sota_v2", "sota_v4"}
    )
    client = build_client(api_key=api_key, base_url=base_url, model=model) if needs_client else None

    tracker = ProgressTracker(
        output_path=output_path,
        eval_data=eval_data,
        shard_size=shard_size,
        save_every=save_every,
        resume=resume,
        retry_errors=retry_errors,
    )
    pending_indices = tracker.pending_indices()
    if not pending_indices:
        tracker.finalize()
        return str(output_path)

    desc = f"ChronoAgentHarness {dataset}"
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                generate_one,
                sanitized[idx],
                eval_data[idx],
                memory_bank,
                dataset,
                client,
                model,
                static_index,
                skeleton_planner,
                idx,
                include_prompt,
                ablation,
                api_mode,
                reasoning_effort,
                enable_thinking,
                static_strategy,
                baseline_mode,
            ): idx
            for idx in pending_indices
        }
        for future in tqdm(as_completed(future_map), total=len(future_map), desc=desc):
            idx = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "question_id": eval_data[idx].get("question_id"),
                    "generated_question": f"ERROR: {exc}",
                    "reference_questions": get_references(eval_data[idx]),
                }
            tracker.record(idx, result)

    tracker.finalize()
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone ChronoAgentHarness generator")
    parser.add_argument("--train", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--include_prompt", action="store_true")
    parser.add_argument("--memory_bank_cache", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_mode", choices=["responses", "chat", "auto"], default="responses")
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--no_retry_errors", action="store_true", help="Do not rerun existing ERROR samples when resuming")
    parser.add_argument("--dataset", default=None, help="Override dataset name used for prompt construction.")
    parser.add_argument("--static_strategy", choices=["transfer_v1", "transfer_v1b", "sota_v2", "sota_v3", "sota_v4"], default="transfer_v1")
    parser.add_argument(
        "--baseline_mode",
        choices=[
            "chronosynth",
            "structured_prompt",
            "operatorstate_prompt",
            "random_prototype_prompt",
            "relation_only_retrieval",
            "prototype_icl",
            "direct_prompt_verifier",
        ],
        default="chronosynth",
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "full",
            "no_memory",
            "relation_only",
            "no_drafts",
            "no_draft_filtering",
            "no_leakage_check",
            "no_default_templates",
            "no_temporal_rule",
        ],
        default="full",
    )
    args = parser.parse_args()

    run_generation(
        train_path=args.train,
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        workers=args.workers,
        shard_size=args.shard_size,
        save_every=args.save_every,
        resume=not args.no_resume,
        include_prompt=args.include_prompt,
        memory_bank_cache=args.memory_bank_cache,
        ablation=args.ablation,
        api_key=args.api_key,
        base_url=args.base_url,
        api_mode=args.api_mode,
        reasoning_effort=args.reasoning_effort,
        enable_thinking=args.enable_thinking,
        retry_errors=not args.no_retry_errors,
        dataset=args.dataset,
        static_strategy=args.static_strategy,
        baseline_mode=args.baseline_mode,
    )


if __name__ == "__main__":
    main()
