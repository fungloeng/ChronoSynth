"""
Unified TKGQG Evaluation  —  single canonical implementation per metric.

Metrics
───────
  BLEU-1/2/3/4  corpus-level, brevity penalty, closest-reflen, merged maxcounts
                (= Graph2Seq4TKGQG bleu_scorer.py  option='closest')
  ROUGE-L       sentence-averaged F-beta (β=1.2, recall-weighted)
                (= Graph2Seq4TKGQG rouge/rouge.py)
  CIDEr         TF-IDF weighted n-gram cosine similarity, n=1..4, σ=6
                (= Graph2Seq4TKGQG cider/cider_scorer.py)
  Distinct-1/2  per-group unique-ngrams / total-ngrams, macro-averaged
                (= R2DQG evaluate.py  distinct_n_score / compute_distinct_n_scores)

Input format (only one, auto-validated)
────────────────────────────────────────
  JSON list of objects, each with:
    "generated_question"  : str
    "reference_questions" : [str, ...]

Usage
─────
  python evaluate.py path/to/results.json
  python evaluate.py path/to/results.json --no_cider
  python evaluate.py path/to/results.json --out metrics.json
"""

import argparse, json, math, sys
from collections import Counter, defaultdict
import numpy as np


# ── tokenisation ─────────────────────────────────────────────────────────────

def _tok(text: str) -> list:
    return text.lower().split()


# ── BLEU  (corpus-level, BP, closest reflen, merged maxcounts) ───────────────

def _ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def _max_ref_ngrams(refs, n):
    merged = {}
    for ref in refs:
        for ng, cnt in _ngram_counts(ref, n).items():
            merged[ng] = max(merged.get(ng, 0), cnt)
    return merged

class _CorpusBleu:
    def __init__(self, n=4):
        self.n = n
        self._correct = [0]*n
        self._guess   = [0]*n
        self._hlen = self._rlen = 0

    def add(self, hyp, refs):
        h = len(hyp)
        r = len(min(refs, key=lambda r: (abs(len(r)-h), len(r))))
        self._hlen += h; self._rlen += r
        for o in range(1, self.n+1):
            maxc = _max_ref_ngrams(refs, o)
            hng  = _ngram_counts(hyp, o)
            self._correct[o-1] += sum(min(hng[g], maxc.get(g,0)) for g in hng)
            self._guess[o-1]   += max(0, h-o+1)

    def scores(self):
        tiny, eps = 1e-15, 1e-9
        log_b = 0.; bleus = []
        for k in range(self.n):
            log_b += math.log((self._correct[k]+tiny)/(self._guess[k]+eps))
            bleus.append(math.exp(log_b/(k+1)))
        bp = math.exp(1-self._rlen/self._hlen) if self._hlen < self._rlen else 1.
        return [b*bp for b in bleus]


# ── ROUGE-L  (sentence-avg, F-beta β=1.2) ────────────────────────────────────

def _lcs(x, y):
    if len(x) < len(y): x, y = y, x
    prev = [0]*(len(y)+1)
    for xi in x:
        curr = [0]*(len(y)+1)
        for j, yj in enumerate(y, 1):
            curr[j] = prev[j-1]+1 if xi==yj else max(prev[j], curr[j-1])
        prev = curr
    return prev[len(y)]

def _rouge_l(hyp, refs, beta=1.2):
    # max precision and max recall separately across refs (rouge/rouge.py)
    precs, recs = [], []
    for ref in refs:
        lcs = _lcs(hyp, ref)
        precs.append(lcs/len(hyp) if hyp else 0.)
        recs.append(lcs/len(ref)  if ref  else 0.)
    P, R = max(precs) if precs else 0., max(recs) if recs else 0.
    if P==0 and R==0: return 0.
    return (1+beta**2)*P*R / (R + beta**2*P)


# ── CIDEr  (TF-IDF weighted n-gram cosine, n=1..4) ───────────────────────────

class _CIDEr:
    def __init__(self, n=4, sigma=6.):
        self.n, self.sigma = n, sigma

    def _cook(self, s):
        w = s.split(); c = {}
        for k in range(1, self.n+1):
            for i in range(len(w)-k+1):
                ng = tuple(w[i:i+k]); c[ng] = c.get(ng,0)+1
        return c

    def compute(self, pairs):
        crefs = [[self._cook(r) for r in refs] for _,refs in pairs]
        ctest = [self._cook(h)  for h,_   in pairs]
        df = {}
        for refs in crefs:
            for ng in set(ng for ref in refs for ng in ref):
                df[ng] = df.get(ng,0)+1
        ref_len = math.log(float(len(crefs))) if crefs else 0.

        def vec(cnts):
            v=[defaultdict(float) for _ in range(self.n)]; norm=[0.]*self.n; L=0
            for ng,tf in cnts.items():
                d=math.log(max(1.,df.get(ng,0))); o=len(ng)-1
                val=float(tf)*(ref_len-d); v[o][ng]=val; norm[o]+=val*val
                if o==1: L+=tf
            return v,[math.sqrt(x) for x in norm],L

        def sim(vh,vr,nh,nr,lh,lr):
            delta=float(lh-lr); out=[]
            for o in range(self.n):
                s=sum(min(cnt,vr[o].get(ng,0.))*vr[o].get(ng,0.) for ng,cnt in vh[o].items())
                if nh[o] and nr[o]: s/=nh[o]*nr[o]
                out.append(s*math.exp(-(delta**2)/(2*self.sigma**2)))
            return out

        scores=[]
        for test,refs in zip(ctest,crefs):
            vh,nh,lh=vec(test); tot=[0.]*self.n
            for ref in refs:
                vr,nr,lr=vec(ref)
                for o,s in enumerate(sim(vh,vr,nh,nr,lh,lr)): tot[o]+=s
            scores.append(np.mean(tot)/len(refs)*10.)
        return float(np.mean(scores)) if scores else 0.


# ── Distinct-N  (per-group, macro-averaged) ───────────────────────────────────

def _distinct_n(texts, n):
    """unique n-grams / total n-grams across all texts in a group."""
    ngc = Counter(); total = 0
    for t in texts:
        toks = t.split()
        ngs = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
        ngc.update(ngs); total += len(ngs)
    return len(ngc)/total if total else 0.


# ── load ─────────────────────────────────────────────────────────────────────

def load(path: str):
    """
    Load a results JSON file.  Expected format:
        [{generated_question: str, reference_questions: [str,...], ...}, ...]
    Returns list of (hyp_str, [ref_str,...]).
    Skips items where generated_question is empty or starts with 'ERROR:'.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data:
        hyp  = item.get("generated_question","").strip()
        raw_refs = item.get("reference_questions",[])
        if isinstance(raw_refs, str):
            raw_refs = [raw_refs]
        refs = [r for r in raw_refs if isinstance(r, str) and r and r.strip()]
        if not refs and item.get("question"):
            refs = [item["question"]]
        if not hyp or hyp.startswith("ERROR:") or not refs:
            continue
        pairs.append((hyp, refs))
    return pairs


# ── evaluate ─────────────────────────────────────────────────────────────────

def evaluate(pairs,
             compute_cider=True,
             verbose=False,
             **_ignored_kwargs) -> dict:
    """
    pairs : list of (hyp_str, [ref_str,...])
    Returns a flat dict of metric→float plus 'num_samples'.
    """
    if not pairs:
        raise ValueError("No valid pairs.")

    cb = _CorpusBleu()
    rouge_scores = []
    # group by position for Distinct (treat whole list as one group)
    all_hyps = []

    for hyp_str, ref_strs in pairs:
        hyp  = _tok(hyp_str)
        refs = [_tok(r) for r in ref_strs]
        cb.add(hyp, refs)
        rouge_scores.append(_rouge_l(hyp, refs))
        all_hyps.append(hyp_str)

    bleus = cb.scores()

    results = {
        "BLEU-1":  bleus[0],
        "BLEU-2":  bleus[1],
        "BLEU-3":  bleus[2],
        "BLEU-4":  bleus[3],
        "ROUGE-L": float(np.mean(rouge_scores)),
    }

    if compute_cider:
        try:
            results["CIDEr"] = _CIDEr().compute(pairs)
        except Exception as e:
            print(f"[warn] CIDEr: {e}", file=sys.stderr)
            results["CIDEr"] = float("nan")

    results["Distinct-1"] = _distinct_n(all_hyps, 1)
    results["Distinct-2"] = _distinct_n(all_hyps, 2)
    results["num_samples"] = len(pairs)

    if verbose:
        _print(results)
    return results


def _print(r):
    n = r["num_samples"]
    print(f"\n{'─'*44}")
    print(f"  Evaluation  ({n} samples)")
    print(f"{'─'*44}")
    for m in ("BLEU-1","BLEU-2","BLEU-3","BLEU-4","ROUGE-L","CIDEr",
              "Distinct-1","Distinct-2"):
        if m in r:
            v = r[m]
            s = f"{v:.4f}" if not math.isnan(v) else "  N/A"
            print(f"  {m:<12s} {s}")
    print(f"{'─'*44}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    ap.add_argument("pred", help="Results JSON file")
    ap.add_argument("--no_cider",  action="store_true")
    ap.add_argument("--out", default=None, help="Write JSON metrics to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pairs = load(args.pred)
    if not pairs:
        print("ERROR: no valid pairs loaded", file=sys.stderr); sys.exit(1)

    r = evaluate(pairs,
                 compute_cider=not args.no_cider,
                 verbose=not args.quiet)

    if args.out:
        with open(args.out,"w",encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
        print(f"Metrics written to: {args.out}")


if __name__ == "__main__":
    main()
