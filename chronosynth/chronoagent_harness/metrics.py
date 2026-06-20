from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _tok(text: str) -> list[str]:
    return text.lower().split()


def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _max_ref_ngrams(refs, n):
    merged = {}
    for ref in refs:
        for ng, cnt in _ngram_counts(ref, n).items():
            merged[ng] = max(merged.get(ng, 0), cnt)
    return merged


class _CorpusBleu:
    def __init__(self, n: int = 4):
        self.n = n
        self._correct = [0] * n
        self._guess = [0] * n
        self._hlen = self._rlen = 0

    def add(self, hyp, refs):
        h = len(hyp)
        r = len(min(refs, key=lambda r: (abs(len(r) - h), len(r))))
        self._hlen += h
        self._rlen += r
        for o in range(1, self.n + 1):
            maxc = _max_ref_ngrams(refs, o)
            hng = _ngram_counts(hyp, o)
            self._correct[o - 1] += sum(min(hng[g], maxc.get(g, 0)) for g in hng)
            self._guess[o - 1] += max(0, h - o + 1)

    def scores(self):
        tiny, eps = 1e-15, 1e-9
        log_b = 0.0
        bleus = []
        for k in range(self.n):
            log_b += math.log((self._correct[k] + tiny) / (self._guess[k] + eps))
            bleus.append(math.exp(log_b / (k + 1)))
        bp = math.exp(1 - self._rlen / self._hlen) if self._hlen < self._rlen else 1.0
        return [b * bp for b in bleus]


def _lcs(x, y):
    if len(x) < len(y):
        x, y = y, x
    prev = [0] * (len(y) + 1)
    for xi in x:
        curr = [0] * (len(y) + 1)
        for j, yj in enumerate(y, 1):
            curr[j] = prev[j - 1] + 1 if xi == yj else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(y)]


def _rouge_l(hyp, refs, beta: float = 1.2):
    precs, recs = [], []
    for ref in refs:
        lcs = _lcs(hyp, ref)
        precs.append(lcs / len(hyp) if hyp else 0.0)
        recs.append(lcs / len(ref) if ref else 0.0)
    p = max(precs) if precs else 0.0
    r = max(recs) if recs else 0.0
    if p == 0 and r == 0:
        return 0.0
    return (1 + beta**2) * p * r / (r + beta**2 * p)


class _CIDEr:
    def __init__(self, n: int = 4, sigma: float = 6.0):
        self.n, self.sigma = n, sigma

    def _cook(self, s):
        w = s.split()
        c = {}
        for k in range(1, self.n + 1):
            for i in range(len(w) - k + 1):
                ng = tuple(w[i : i + k])
                c[ng] = c.get(ng, 0) + 1
        return c

    def compute(self, pairs):
        crefs = [[self._cook(r) for r in refs] for _, refs in pairs]
        ctest = [self._cook(h) for h, _ in pairs]
        df = {}
        for refs in crefs:
            for ng in set(ng for ref in refs for ng in ref):
                df[ng] = df.get(ng, 0) + 1
        ref_len = math.log(float(len(crefs))) if crefs else 0.0

        def vec(cnts):
            v = [defaultdict(float) for _ in range(self.n)]
            norm = [0.0] * self.n
            length = 0
            for ng, tf in cnts.items():
                d = math.log(max(1.0, df.get(ng, 0)))
                o = len(ng) - 1
                val = float(tf) * (ref_len - d)
                v[o][ng] = val
                norm[o] += val * val
                if o == 1:
                    length += tf
            return v, [math.sqrt(x) for x in norm], length

        def sim(vh, vr, nh, nr, lh, lr):
            delta = float(lh - lr)
            out = []
            for o in range(self.n):
                s = sum(min(cnt, vr[o].get(ng, 0.0)) * vr[o].get(ng, 0.0) for ng, cnt in vh[o].items())
                if nh[o] and nr[o]:
                    s /= nh[o] * nr[o]
                out.append(s * math.exp(-(delta**2) / (2 * self.sigma**2)))
            return out

        scores = []
        for test, refs in zip(ctest, crefs):
            vh, nh, lh = vec(test)
            tot = [0.0] * self.n
            for ref in refs:
                vr, nr, lr = vec(ref)
                for o, s in enumerate(sim(vh, vr, nh, nr, lh, lr)):
                    tot[o] += s
            scores.append(np.mean(tot) / len(refs) * 10.0)
        return float(np.mean(scores)) if scores else 0.0


def load(path: str | Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data:
        hyp = item.get("generated_question", "").strip()
        raw_refs = item.get("reference_questions", [])
        if isinstance(raw_refs, str):
            raw_refs = [raw_refs]
        refs = [r for r in raw_refs if isinstance(r, str) and r and r.strip()]
        if not refs and item.get("question"):
            refs = [item["question"]]
        if not hyp or hyp.startswith("ERROR:") or not refs:
            continue
        pairs.append((hyp, refs))
    return pairs


def evaluate(pairs, compute_meteor: bool = False, compute_cider: bool = True) -> dict:
    if not pairs:
        raise ValueError("No valid pairs.")

    cb = _CorpusBleu()
    rouge_scores = []
    for hyp_str, ref_strs in pairs:
        hyp = _tok(hyp_str)
        refs = [_tok(r) for r in ref_strs]
        cb.add(hyp, refs)
        rouge_scores.append(_rouge_l(hyp, refs))

    bleus = cb.scores()
    results = {
        "BLEU-1": bleus[0],
        "BLEU-2": bleus[1],
        "BLEU-3": bleus[2],
        "BLEU-4": bleus[3],
        "ROUGE-L": float(np.mean(rouge_scores)),
        "num_samples": len(pairs),
    }

    if compute_meteor:
        results["METEOR"] = float("nan")

    if compute_cider:
        results["CIDEr"] = _CIDEr().compute(pairs)

    return results


def save_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standalone ChronoAgentHarness outputs")
    parser.add_argument("path")
    parser.add_argument("--out")
    parser.add_argument("--no_cider", action="store_true")
    args = parser.parse_args()

    pairs = load(args.path)
    metrics = evaluate(pairs, compute_meteor=False, compute_cider=not args.no_cider)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.out:
        save_metrics(args.out, metrics)


if __name__ == "__main__":
    main()
