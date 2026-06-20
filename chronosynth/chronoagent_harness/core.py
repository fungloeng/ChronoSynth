from __future__ import annotations

import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PLACEHOLDERS = (
    "{ANSWER}",
    "{ANCHOR}",
    "{CMP_ENTITY}",
    "{CMP_TIME}",
    "{FOCUS_SOURCE}",
    "{FOCUS_TARGET}",
    "{FOCUS_TIME}",
)


RELATION_HINTS = {
    "member of sports team": "played for",
    "position held": "became",
    "spouse": "married",
    "award received": "received",
    "employer": "worked for",
    "express intent to meet or negotiate": "wanted to meet",
    "make an appeal or request": "appealed to",
    "praise or endorse": "praised",
    "host a visit": "hosted a visit from",
    "accuse": "accused",
    "criticize or denounce": "criticised",
    "discuss by telephone": "telephoned",
    "sign formal agreement": "signed a formal agreement with",
    "express intent to cooperate": "wanted to cooperate with",
    "engage in negotiation": "negotiated with",
    "consult": "consulted",
    "reject": "rejected",
    "make a visit": "visited",
    "threaten": "threatened",
    "make statement": "made a statement to",
    "fight with small arms and light weapons": "engaged in armed conflict with",
    "use unconventional violence": "used force against",
    "use unconventional force": "used force against",
    "use conventional military force": "deployed military force against",
    "engage in mass violence": "carried out mass action against",
    "assault": "acted against",
    "attack": "acted against",
}

MEMORY_BANK_CACHE_VERSION = 1
STATIC_DATASETS = {"CWQ", "PQ", "WQ"}


@dataclass(frozen=True)
class EdgeRecord:
    source: str
    relation: str
    target: str
    time_text: str
    is_constraint: bool = False


@dataclass
class OperatorState:
    dataset: str
    question_id: int | str | None
    answers: list[str]
    answer_type: str
    answer_count: int
    time_level: str
    ask_slot: str
    operator: str
    focus_relation: str
    focus_source: str
    focus_target: str
    focus_time: str
    anchor_entity: str
    comparator_entity: str
    comparator_time: str
    constraint_mode: str
    constraint_value: str
    edge_count: int
    graph_shape: str
    edges: list[EdgeRecord]
    question: str = ""
    opening: str = ""
    prototype: str = ""
    summary: str = ""
    focus_summary: str = ""


@dataclass
class MemoryExample:
    question_id: int | str | None
    question: str
    opening: str
    prototype: str
    dataset: str
    ask_slot: str
    operator: str
    answer_type: str
    time_level: str
    relation_key: str
    edge_bucket: int
    graph_shape: str
    summary: str
    focus_summary: str
    placeholder_count: int
    comparator_kind: str
    anchor_tokens: set[str] = field(default_factory=set)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def first_reference(item: dict[str, Any]) -> str:
    refs = item.get("paraphrases") or []
    if isinstance(refs, str):
        refs = [refs]
    for ref in refs:
        if isinstance(ref, str) and ref.strip():
            return ref.strip()
    question = str(item.get("question") or "").strip()
    return question


def answer_strings(item: dict[str, Any]) -> list[str]:
    raw = item.get("answer_text") or item.get("answers") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(value).strip() for value in raw if str(value).strip()]


def is_time_like(text: str) -> bool:
    value = str(text).strip()
    return bool(
        re.fullmatch(r"\d{4}", value)
        or re.fullmatch(r"\d{4}-\d{2}", value)
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        or re.fullmatch(r"\d{4}\s*-\s*\d{4}", value)
    )


def guess_dataset_name(path: str | None = None, sample: dict[str, Any] | None = None) -> str:
    if path:
        upper = str(path).upper()
        if "CWQ" in upper:
            return "CWQ"
        if "PATHQUESTION" in upper or "PQ" in upper:
            return "PQ"
        if "MULTITQ" in upper:
            return "MULTITQ"
        if "CRONQUESTION" in upper:
            return "CRONQUESTION"
    sample = sample or {}
    for edge in (sample.get("subgraph") or {}).get("edges", []) or []:
        if "source_name" in edge or "target_name" in edge or sample.get("qtype"):
            return "MULTITQ"
    return "CRONQUESTION"


def is_static_dataset(dataset: str | None) -> bool:
    return str(dataset or "").upper() in STATIC_DATASETS


def edge_time_text(edge: dict[str, Any]) -> str:
    if edge.get("time"):
        return str(edge["time"])
    start = edge.get("start_year")
    end = edge.get("end_year")
    if start and end and str(start) != str(end):
        return f"{start}-{end}"
    if start:
        return str(start)
    if end:
        return str(end)
    return ""


def parse_time_key(text: str) -> tuple[int, ...]:
    parts = [int(value) for value in re.findall(r"\d+", str(text))]
    if not parts:
        return (9999, 99, 99)
    if len(parts) == 1:
        return (parts[0], 0, 0)
    if len(parts) == 2:
        return (parts[0], parts[1], 0)
    return tuple(parts[:3])


def ordered_edges(item: dict[str, Any]) -> list[EdgeRecord]:
    edges = []
    for raw in (item.get("subgraph") or {}).get("edges", []) or []:
        source = str(raw.get("source") or raw.get("source_name") or "").strip()
        target = str(raw.get("target") or raw.get("target_name") or "").strip()
        relation = str(raw.get("relation") or "").strip()
        time_text = edge_time_text(raw)
        lower_source = source.lower()
        lower_target = target.lower()
        is_constraint = "time constraint" in {lower_source, lower_target}
        if not is_constraint and relation.lower() in {"before", "after"}:
            other = target if lower_source == "time constraint" else source if lower_target == "time constraint" else ""
            is_constraint = is_time_like(other)
        edges.append(
            EdgeRecord(
                source=source,
                relation=relation,
                target=target,
                time_text=time_text,
                is_constraint=is_constraint,
            )
        )
    return sorted(
        edges,
        key=lambda edge: (
            edge.is_constraint,
            parse_time_key(edge.time_text),
            edge.source,
            edge.relation,
            edge.target,
        ),
    )


def infer_time_level(item: dict[str, Any], answers: list[str], edges: list[EdgeRecord]) -> str:
    explicit = item.get("time_level")
    if explicit:
        return str(explicit)
    values = list(item.get("times") or []) + answers + [edge.time_text for edge in edges if edge.time_text]
    for value in values:
        text = str(value)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return "day"
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return "month"
        if re.fullmatch(r"\d{4}", text) or re.fullmatch(r"\d{4}\s*-\s*\d{4}", text):
            return "year"
    return "year"


def canonical_graph_shape(edges: list[EdgeRecord]) -> str:
    entity_ids: dict[str, str] = {}
    next_id = 1
    parts = []

    def canon(value: str) -> str:
        nonlocal next_id
        if value not in entity_ids:
            entity_ids[value] = f"e{next_id}"
            next_id += 1
        return entity_ids[value]

    for edge in edges:
        if edge.is_constraint:
            continue
        parts.append(f"({canon(edge.source)},r,{canon(edge.target)},{'t' if edge.time_text else 'x'})")
    return " ".join(parts) if parts else "(e)"


def graph_summary(edges: list[EdgeRecord], limit: int = 5) -> str:
    lines = []
    for edge in edges:
        if edge.is_constraint:
            continue
        line = f"{edge.source} -- {edge.relation} --> {edge.target}"
        if edge.time_text:
            line += f" [{edge.time_text}]"
        lines.append(line)
        if len(lines) >= limit:
            break
    return "; ".join(lines)


def _entity_surface_score(text: str) -> float:
    value = str(text or "").strip()
    if not value:
        return -5.0
    score = min(len(value.split()), 4) * 0.4
    if "(" not in value and ")" not in value:
        score += 0.6
    if any(char.isdigit() for char in value):
        score -= 0.2
    return score


def _relation_key(edge: EdgeRecord) -> str:
    return normalize_text(edge.relation)


def _time_match(edge_time: str, answer_time: str) -> bool:
    edge_time = str(edge_time).strip()
    answer_time = str(answer_time).strip()
    if not edge_time or not answer_time:
        return False
    if edge_time == answer_time:
        return True
    if re.fullmatch(r"\d{4}", answer_time):
        if edge_time.startswith(answer_time):
            return True
        if re.fullmatch(r"\d{4}\s*-\s*\d{4}", edge_time):
            start, end = [part.strip() for part in edge_time.split("-", 1)]
            return answer_time in {start, end}
        return False
    if re.fullmatch(r"\d{4}-\d{2}", answer_time):
        return edge_time.startswith(answer_time)
    return False


def _matching_edges(edges: list[EdgeRecord], answers: list[str], answer_type: str) -> list[EdgeRecord]:
    answer_set = set(answers)
    if answer_type == "time":
        return [edge for edge in edges if edge.time_text and any(_time_match(edge.time_text, ans) for ans in answer_set)]
    return [edge for edge in edges if edge.source in answer_set or edge.target in answer_set]


def _choose_focus_edge(
    candidates: list[EdgeRecord],
    all_edges: list[EdgeRecord],
    ask_slot: str,
    answers: list[str],
) -> EdgeRecord | None:
    if not candidates:
        return None
    answer_set = set(answers)

    def score(edge: EdgeRecord) -> tuple[float, tuple[int, ...], str, str]:
        if ask_slot == "source":
            anchor = edge.target
        elif ask_slot == "target":
            anchor = edge.source
        else:
            anchor = edge.source if _entity_surface_score(edge.source) >= _entity_surface_score(edge.target) else edge.target
        family = [
            other
            for other in all_edges
            if not other.is_constraint
            and _relation_key(other) == _relation_key(edge)
            and anchor in {other.source, other.target}
        ]
        value = len(family) * 1.5 + _entity_surface_score(anchor)
        if ask_slot == "source" and edge.source in answer_set:
            value += 1.0
        if ask_slot == "target" and edge.target in answer_set:
            value += 1.0
        if edge.time_text:
            value += 0.2
        return (value, parse_time_key(edge.time_text), edge.source, edge.target)

    return max(candidates, key=score)


def _constraint_signal(edges: list[EdgeRecord]) -> tuple[str, str]:
    for edge in edges:
        if not edge.is_constraint:
            continue
        relation = edge.relation.lower()
        other = edge.target if edge.source.lower() == "time constraint" else edge.source
        if relation in {"before", "after"} and is_time_like(other):
            return (relation, other)
    return ("", "")


def _extract_opening(question: str) -> str:
    tokens = question.strip().split()
    if not tokens:
        return ""
    lowered = [token.lower() for token in tokens[:4]]
    if lowered[:3] == ["in", "which", "year"]:
        return "In which year"
    if lowered[:2] == ["with", "whom"]:
        return "With whom"
    if lowered[0] in {"who", "when", "which", "what", "where", "did", "was"}:
        return " ".join(tokens[: min(4, len(tokens))]).strip()
    return " ".join(tokens[: min(4, len(tokens))]).strip()


def _safe_replace(text: str, old: str, new: str) -> str:
    old = str(old or "").strip()
    new = str(new or "").strip()
    if not old or old == new:
        return text
    pattern = re.escape(old)
    return re.sub(pattern, new, text)


def _make_prototype(question: str, operator_state: OperatorState) -> str:
    if not question:
        return ""
    text = question
    replacements = []
    for answer in sorted(operator_state.answers, key=len, reverse=True):
        replacements.append((answer, "{ANSWER}"))
    replacements.extend(
        [
            (operator_state.anchor_entity, "{ANCHOR}"),
            (operator_state.comparator_entity, "{CMP_ENTITY}"),
            (operator_state.comparator_time or operator_state.constraint_value, "{CMP_TIME}"),
            (operator_state.focus_source, "{FOCUS_SOURCE}"),
            (operator_state.focus_target, "{FOCUS_TARGET}"),
            (operator_state.focus_time, "{FOCUS_TIME}"),
        ]
    )
    seen = set()
    ordered = []
    for old, new in replacements:
        key = (old, new)
        if key in seen or not old:
            continue
        seen.add(key)
        ordered.append((old, new))
    ordered.sort(key=lambda pair: len(pair[0]), reverse=True)
    for old, new in ordered:
        text = _safe_replace(text, old, new)
    return " ".join(text.split()).strip()


def _humanize_relation(relation: str) -> str:
    key = normalize_text(relation)
    if key in RELATION_HINTS:
        return RELATION_HINTS[key]
    return relation.replace("_", " ").strip().lower()


def _default_templates(operator_state: OperatorState, openings: list[str]) -> list[str]:
    opening = openings[0] if openings else ""
    relation_phrase = _humanize_relation(operator_state.focus_relation)
    drafts: list[str] = []

    if operator_state.ask_slot == "time":
        starter = "What year" if operator_state.time_level == "year" else "When"
        if opening.lower().startswith("when"):
            starter = "When"
        if opening.lower().startswith("in which year"):
            starter = "In which year"
        if opening.lower().startswith("what year"):
            starter = "What year"
        if operator_state.operator == "first":
            drafts.append(f"{starter} was the first time {operator_state.focus_source} {relation_phrase} {operator_state.focus_target}?")
        elif operator_state.operator == "last":
            drafts.append(f"{starter} was the last time {operator_state.focus_source} {relation_phrase} {operator_state.focus_target}?")
        else:
            if starter in {"What year", "In which year"}:
                drafts.append(f"{starter} did {operator_state.focus_source} {relation_phrase} {operator_state.focus_target}?")
            else:
                drafts.append(f"When did {operator_state.focus_source} {relation_phrase} {operator_state.focus_target}?")
        return drafts

    if operator_state.operator == "after_time" and operator_state.constraint_value:
        drafts.append(f"After {operator_state.constraint_value}, who {relation_phrase} {operator_state.anchor_entity}?")
    elif operator_state.operator == "before_time" and operator_state.constraint_value:
        drafts.append(f"Before {operator_state.constraint_value}, who {relation_phrase} {operator_state.anchor_entity}?")
    elif operator_state.operator == "before_last" and operator_state.comparator_entity:
        drafts.append(f"Who was the last to {relation_phrase} {operator_state.anchor_entity} before {operator_state.comparator_entity}?")
    elif operator_state.operator == "after_first" and operator_state.comparator_entity:
        drafts.append(f"After {operator_state.comparator_entity}, who first {relation_phrase} {operator_state.anchor_entity}?")
    elif operator_state.operator == "first":
        drafts.append(f"Who first {relation_phrase} {operator_state.anchor_entity}?")
    elif operator_state.operator == "last":
        drafts.append(f"Who last {relation_phrase} {operator_state.anchor_entity}?")
    else:
        drafts.append(f"Who {relation_phrase} {operator_state.anchor_entity}?")

    if opening:
        drafts.append(f"{opening} {relation_phrase} {operator_state.anchor_entity}?")
    return drafts


def build_operator_state(item: dict[str, Any], dataset: str | None = None) -> OperatorState:
    dataset = dataset or guess_dataset_name(sample=item)
    edges = ordered_edges(item)
    usable_edges = [edge for edge in edges if not edge.is_constraint]
    answers = answer_strings(item)
    answer_type = "time" if answers and all(is_time_like(answer) for answer in answers) else "entity"
    ask_slot = "time"
    matched = _matching_edges(usable_edges, answers, answer_type)
    comparator_entity = ""
    comparator_time = ""
    constraint_mode = ""
    constraint_value = ""
    operator = "plain" if answer_type == "entity" else "when"

    if is_static_dataset(dataset):
        time_level = "static"
        operator = "static"
        if answer_type == "entity":
            answer_set = set(answers)
            source_hits = sum(1 for edge in matched if edge.source in answer_set)
            target_hits = sum(1 for edge in matched if edge.target in answer_set)
            if source_hits >= target_hits and source_hits > 0:
                ask_slot = "source"
            elif target_hits > 0:
                ask_slot = "target"
            else:
                ask_slot = "source"
        focus_edge = _choose_focus_edge(matched or usable_edges[:1], usable_edges, ask_slot, answers)
        if focus_edge is None:
            focus_edge = EdgeRecord("", "", "", "")
        anchor_entity = focus_edge.target if ask_slot == "source" else focus_edge.source
    else:
        time_level = infer_time_level(item, answers, usable_edges)

        if answer_type == "entity":
            answer_set = set(answers)
            source_hits = sum(1 for edge in matched if edge.source in answer_set)
            target_hits = sum(1 for edge in matched if edge.target in answer_set)
            if source_hits >= target_hits and source_hits > 0:
                ask_slot = "source"
            elif target_hits > 0:
                ask_slot = "target"
            else:
                ask_slot = "source"

        focus_edge = _choose_focus_edge(matched or usable_edges[:1], usable_edges, ask_slot, answers)
        if focus_edge is None:
            focus_edge = EdgeRecord("", "", "", "")

        if ask_slot == "source":
            anchor_entity = focus_edge.target
        elif ask_slot == "target":
            anchor_entity = focus_edge.source
        else:
            anchor_entity = focus_edge.source

        constraint_mode, constraint_value = _constraint_signal(edges)
        relation_key = _relation_key(focus_edge)
        family = [
            edge
            for edge in usable_edges
            if _relation_key(edge) == relation_key
            and anchor_entity
            and anchor_entity in {edge.source, edge.target}
        ]
        focus_key = parse_time_key(focus_edge.time_text)
        answer_set = set(answers)
        answer_family = []
        for edge in family:
            if ask_slot == "source" and edge.source in answer_set:
                answer_family.append(edge)
            elif ask_slot == "target" and edge.target in answer_set:
                answer_family.append(edge)
            elif ask_slot == "time" and any(_time_match(edge.time_text, answer) for answer in answer_set):
                answer_family.append(edge)

        earlier = [edge for edge in family if parse_time_key(edge.time_text) < focus_key and edge not in answer_family]
        later = [edge for edge in family if parse_time_key(edge.time_text) > focus_key and edge not in answer_family]

        if constraint_mode:
            operator = f"{constraint_mode}_time"
            comparator_time = constraint_value
        elif answer_type == "entity" and later:
            comparator_edge = min(later, key=lambda edge: parse_time_key(edge.time_text))
            comparator_entity = comparator_edge.target if comparator_edge.source == anchor_entity else comparator_edge.source
            comparator_time = comparator_edge.time_text
            before_answer_times = [parse_time_key(edge.time_text) for edge in answer_family if parse_time_key(edge.time_text) < parse_time_key(comparator_edge.time_text)]
            operator = "before_last" if before_answer_times and focus_key == max(before_answer_times) else "before"
        elif answer_type == "entity" and earlier:
            comparator_edge = max(earlier, key=lambda edge: parse_time_key(edge.time_text))
            comparator_entity = comparator_edge.target if comparator_edge.source == anchor_entity else comparator_edge.source
            comparator_time = comparator_edge.time_text
            after_answer_times = [parse_time_key(edge.time_text) for edge in answer_family if parse_time_key(edge.time_text) > parse_time_key(comparator_edge.time_text)]
            operator = "after_first" if after_answer_times and focus_key == min(after_answer_times) else "after"
        else:
            family_times = [parse_time_key(edge.time_text) for edge in family if edge.time_text]
            if len(family_times) > 1 and focus_edge.time_text:
                if focus_key == min(family_times):
                    operator = "first"
                elif focus_key == max(family_times):
                    operator = "last"

    question = first_reference(item)
    operator_state = OperatorState(
        dataset=dataset,
        question_id=item.get("question_id"),
        answers=answers,
        answer_type=answer_type,
        answer_count=len(answers),
        time_level=time_level,
        ask_slot=ask_slot,
        operator=operator,
        focus_relation=focus_edge.relation,
        focus_source=focus_edge.source,
        focus_target=focus_edge.target,
        focus_time=focus_edge.time_text,
        anchor_entity=anchor_entity,
        comparator_entity=comparator_entity,
        comparator_time=comparator_time,
        constraint_mode=constraint_mode,
        constraint_value=constraint_value,
        edge_count=len(usable_edges),
        graph_shape=canonical_graph_shape(edges),
        edges=usable_edges,
        question=question,
        opening=_extract_opening(question),
        summary=graph_summary(edges),
    )
    operator_state.prototype = _make_prototype(question, operator_state)
    focus_line = f"{operator_state.focus_source} -- {operator_state.focus_relation} --> {operator_state.focus_target}"
    if operator_state.focus_time:
        focus_line += f" [{operator_state.focus_time}]"
    operator_state.focus_summary = focus_line
    return operator_state


class HarnessMemoryBank:
    def __init__(self, bucket_cap: int = 192, global_cap: int = 512):
        self.bucket_cap = bucket_cap
        self.global_cap = global_cap
        self._bucket: dict[tuple[str, str, str, str, str, int], list[MemoryExample]] = defaultdict(list)
        self._relation: dict[tuple[str, str, str], list[MemoryExample]] = defaultdict(list)
        self._shape: dict[tuple[str, str], list[MemoryExample]] = defaultdict(list)
        self._openings: dict[tuple[str, str, str, str, str, int], Counter[str]] = defaultdict(Counter)
        self._global: list[MemoryExample] = []

    def _bucket_key(self, operator_state: OperatorState) -> tuple[str, str, str, str, str, int]:
        return (
            operator_state.dataset,
            operator_state.operator,
            operator_state.ask_slot,
            operator_state.answer_type,
            operator_state.time_level,
            min(operator_state.edge_count, 6),
        )

    def add(self, item: dict[str, Any], dataset: str | None = None) -> None:
        operator_state = build_operator_state(item, dataset=dataset)
        if not operator_state.question:
            return
        example = MemoryExample(
            question_id=operator_state.question_id,
            question=operator_state.question,
            opening=operator_state.opening,
            prototype=operator_state.prototype,
            dataset=operator_state.dataset,
            ask_slot=operator_state.ask_slot,
            operator=operator_state.operator,
            answer_type=operator_state.answer_type,
            time_level=operator_state.time_level,
            relation_key=normalize_text(operator_state.focus_relation),
            edge_bucket=min(operator_state.edge_count, 6),
            graph_shape=operator_state.graph_shape,
            summary=operator_state.summary,
            focus_summary=operator_state.focus_summary,
            placeholder_count=sum(1 for token in PLACEHOLDERS if token in operator_state.prototype),
            comparator_kind="time" if operator_state.comparator_time or operator_state.constraint_value else "entity" if operator_state.comparator_entity else "none",
            anchor_tokens=set(normalize_text(operator_state.anchor_entity).split()),
        )
        bucket = self._bucket[self._bucket_key(operator_state)]
        if len(bucket) < self.bucket_cap:
            bucket.append(example)
        relation_bucket = self._relation[(operator_state.dataset, example.relation_key, operator_state.ask_slot)]
        if len(relation_bucket) < self.bucket_cap:
            relation_bucket.append(example)
        shape_bucket = self._shape[(operator_state.dataset, operator_state.graph_shape)]
        if len(shape_bucket) < self.bucket_cap:
            shape_bucket.append(example)
        if operator_state.opening:
            self._openings[self._bucket_key(operator_state)][operator_state.opening] += 1
        if len(self._global) < self.global_cap:
            self._global.append(example)

    def top_openings(self, operator_state: OperatorState, limit: int = 4) -> list[str]:
        return [opening for opening, _ in self._openings[self._bucket_key(operator_state)].most_common(limit)]

    def retrieve(
        self,
        operator_state: OperatorState,
        limit: int = 6,
        mode: str = "full",
    ) -> list[MemoryExample]:
        if mode == "none":
            return []

        candidates: list[MemoryExample] = []
        relation_key = normalize_text(operator_state.focus_relation)
        if mode == "relation_only":
            candidates.extend(self._relation.get((operator_state.dataset, relation_key, operator_state.ask_slot), []))
        else:
            candidates.extend(self._bucket.get(self._bucket_key(operator_state), []))
            candidates.extend(self._relation.get((operator_state.dataset, relation_key, operator_state.ask_slot), []))
            candidates.extend(self._shape.get((operator_state.dataset, operator_state.graph_shape), []))
        if len(candidates) < 24:
            candidates.extend(self._global)

        target_anchor_tokens = set(normalize_text(operator_state.anchor_entity).split())
        seen: set[int | str | None] = set()
        ranked: list[tuple[float, MemoryExample]] = []
        for example in candidates:
            if example.question_id in seen:
                continue
            seen.add(example.question_id)
            score = 0.0
            if example.dataset == operator_state.dataset:
                score += 2.0
            if mode == "relation_only":
                if example.relation_key == relation_key:
                    score += 3.5
            else:
                if example.operator == operator_state.operator:
                    score += 4.0
                elif example.operator.split("_", 1)[0] == operator_state.operator.split("_", 1)[0]:
                    score += 1.5
                if example.ask_slot == operator_state.ask_slot:
                    score += 3.0
                if example.answer_type == operator_state.answer_type:
                    score += 2.5
                if example.time_level == operator_state.time_level:
                    score += 1.5
                if example.relation_key == relation_key:
                    score += 3.5
                if example.graph_shape == operator_state.graph_shape:
                    score += 2.0
            score += 1.0 / (1 + abs(example.edge_bucket - min(operator_state.edge_count, 6)))
            score += min(len(target_anchor_tokens & example.anchor_tokens), 3) * 0.4
            score += min(example.placeholder_count, 4) * 0.2
            ranked.append((score, example))

        ranked.sort(key=lambda pair: (-pair[0], pair[1].edge_bucket, str(pair[1].question_id)))
        return [example for _, example in ranked[:limit]]

    def adapt_drafts(
        self,
        operator_state: OperatorState,
        examples: list[MemoryExample],
        limit: int = 4,
        validate_placeholders: bool = True,
        validate_leakage: bool = True,
        use_default_templates: bool = True,
    ) -> list[str]:
        drafts: list[str] = []
        substitutions = {
            "{ANCHOR}": operator_state.anchor_entity,
            "{CMP_ENTITY}": operator_state.comparator_entity,
            "{CMP_TIME}": operator_state.comparator_time or operator_state.constraint_value,
            "{FOCUS_SOURCE}": operator_state.focus_source,
            "{FOCUS_TARGET}": operator_state.focus_target,
            "{FOCUS_TIME}": operator_state.focus_time,
        }
        for example in examples:
            text = example.prototype or example.question
            if not text:
                continue
            for placeholder, value in substitutions.items():
                if placeholder in text and value:
                    text = text.replace(placeholder, value)
            if validate_placeholders and any(token in text for token in PLACEHOLDERS):
                continue
            text = " ".join(text.split()).strip()
            if not text:
                continue
            if validate_leakage and any(answer and answer in text for answer in operator_state.answers):
                continue
            if not text.endswith("?"):
                text += "?"
            if text not in drafts:
                drafts.append(text)
            if len(drafts) >= limit:
                break
        if use_default_templates and len(drafts) < limit:
            for draft in _default_templates(operator_state, self.top_openings(operator_state)):
                draft = " ".join(draft.split()).strip()
                if draft and draft not in drafts:
                    drafts.append(draft)
                if len(drafts) >= limit:
                    break
        return drafts[:limit]


def build_memory_bank(train_data: Iterable[dict[str, Any]], dataset: str | None = None) -> HarnessMemoryBank:
    bank = HarnessMemoryBank()
    for item in train_data:
        bank.add(item, dataset=dataset)
    return bank


def build_memory_bank_cache_metadata(train_path: str | Path, dataset: str) -> dict[str, Any]:
    path = Path(train_path).resolve()
    stat = path.stat()
    return {
        "cache_version": MEMORY_BANK_CACHE_VERSION,
        "dataset": dataset,
        "train_path": str(path),
        "train_size": stat.st_size,
        "train_mtime_ns": stat.st_mtime_ns,
    }


def save_memory_bank(
    memory_bank: HarnessMemoryBank,
    cache_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MEMORY_BANK_CACHE_VERSION,
        "metadata": dict(metadata),
        "memory_bank": memory_bank,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_memory_bank(
    cache_path: str | Path,
    expected_metadata: dict[str, Any] | None = None,
) -> HarnessMemoryBank | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (AttributeError, EOFError, ModuleNotFoundError, OSError, pickle.PickleError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != MEMORY_BANK_CACHE_VERSION:
        return None
    if expected_metadata is not None and payload.get("metadata") != expected_metadata:
        return None
    memory_bank = payload.get("memory_bank")
    if not isinstance(memory_bank, HarnessMemoryBank):
        return None
    return memory_bank


def _sanitize_prompt(text: str) -> str:
    for raw, neutral in RELATION_HINTS.items():
        pattern = re.compile(re.escape(raw), re.IGNORECASE)
        text = pattern.sub(neutral, text)
    return text


def build_prompt(
    operator_state: OperatorState,
    memory_bank: HarnessMemoryBank,
    retrieval_mode: str = "full",
    use_drafts: bool = True,
    validate_placeholders: bool = True,
    validate_leakage: bool = True,
    use_default_templates: bool = True,
    enforce_temporal_rule: bool = True,
) -> str:
    retrieved = memory_bank.retrieve(operator_state, limit=4, mode=retrieval_mode)
    openings = memory_bank.top_openings(operator_state, limit=4) if retrieval_mode != "none" else []
    drafts = (
        memory_bank.adapt_drafts(
            operator_state,
            retrieved,
            limit=4,
            validate_placeholders=validate_placeholders,
            validate_leakage=validate_leakage,
            use_default_templates=use_default_templates,
        )
        if use_drafts
        else []
    )

    graph_lines = []
    for idx, edge in enumerate(operator_state.edges, start=1):
        line = f"{idx}. {edge.source} -- {edge.relation} --> {edge.target}"
        if edge.time_text:
            line += f" [{edge.time_text}]"
        graph_lines.append(line)

    memory_lines = []
    for idx, example in enumerate(retrieved, start=1):
        memory_lines.append(
            "\n".join(
                [
                    f"Memory {idx}",
                    f"- Focus: {example.focus_summary}",
                    f"- Question: {example.question}",
                    f"- Prototype: {example.prototype or example.question}",
                ]
            )
        )

    static_mode = operator_state.dataset in STATIC_DATASETS
    if static_mode:
        constraint_lines = [
            f"Dataset dialect: {operator_state.dataset}",
            f"Ask slot: {operator_state.ask_slot}",
            f"Answer type: {operator_state.answer_type}",
            f"Answer count: {operator_state.answer_count}",
            f"Relation focus: {operator_state.focus_relation or 'none'}",
            f"Focus edge: {operator_state.focus_summary}",
            f"Anchor entity: {operator_state.anchor_entity or 'none'}",
            f"Known answer for planning only: {', '.join(operator_state.answers) or 'unknown'}",
        ]
    else:
        constraint_lines = [
            f"Dataset dialect: {operator_state.dataset}",
            f"Ask slot: {operator_state.ask_slot}",
            f"Answer type: {operator_state.answer_type}",
            f"Answer count: {operator_state.answer_count}",
            f"Temporal mode: {operator_state.operator}",
            f"Time granularity: {operator_state.time_level}",
            f"Focus edge: {operator_state.focus_summary}",
            f"Anchor entity: {operator_state.anchor_entity or 'none'}",
            f"Comparator entity: {operator_state.comparator_entity or 'none'}",
            f"Comparator time: {(operator_state.comparator_time or operator_state.constraint_value) or 'none'}",
            f"Known answer for planning only: {', '.join(operator_state.answers) or 'unknown'}",
        ]
    if openings:
        constraint_lines.append(f"Frequent openings in this bucket: {', '.join(openings)}")

    if static_mode:
        header = "You are ChronoSynth, a KG question-generation harness."
        hard_rule = "- Keep the question faithful to the graph relation names and entity names.\n"
    else:
        header = "You are ChronoAgentHarness, a temporal KG question-generation harness."
        hard_rule = "- Preserve the temporal cue exactly: first/last/before/after and time constraints must stay faithful.\n"

    prompt = f"""{header}

The operator state below is already distilled from the training split. Use it directly instead of rediscovering the task from scratch.

[Operator State]
{chr(10).join(f"- {line}" for line in constraint_lines)}

[Graph Facts]
{chr(10).join(graph_lines)}

[Training Memory]
{chr(10).join(memory_lines) if memory_lines else 'No close memories found.'}

[Draft Candidates]
{chr(10).join(f"{idx}. {draft}" for idx, draft in enumerate(drafts, start=1)) if drafts else 'No draft candidates.'}

[Hard Rules]
- Output exactly one natural question.
- Never reveal the answer string in the question.
- {('Temporal cues may be rewritten if needed for fluency.' if not enforce_temporal_rule and not static_mode else 'Preserve the temporal cue exactly: first/last/before/after and time constraints must stay faithful.' if not static_mode else 'Keep the question faithful to the graph relation names and entity names.')}
- Mention entities and timestamps exactly as they appear in the graph.
- Prefer the wording style suggested by the retrieved memories and draft candidates.
- Do not output analysis, labels, bullets, or alternatives.

Return only the final question."""
    return _sanitize_prompt(prompt)


analyze_item = build_operator_state
