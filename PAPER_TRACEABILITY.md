# Paper Traceability

This document maps the final manuscript claims to the released implementation and frozen aggregate records.

## Method-to-code map

| Paper component | Implementation |
|---|---|
| Temporal Intent State | `chronosynth/chronoagent_harness/core.py` |
| Typed pattern normalization | `chronosynth/chronoagent_harness/core.py` |
| Indexed pattern memory | `chronosynth/chronoagent_harness/core.py` |
| Rule-aware retrieval score | `HarnessMemoryBank.retrieve` in `core.py` |
| Typed draft adaptation and validation | `core.py` |
| Parallel generation and cache reuse | `chronosynth/chronoagent_harness/generator.py` |
| Full/no-memory/relation-only experiments | `experiments/run_chrono_experiment.py` |
| BLEU, ROUGE-L, CIDEr | `evaluate.py` |
| Grouped RQ3 metrics | `experiments/analyze_grouped_results.py` |
| Grouped confidence intervals | `experiments/grouped_significance.py` |
| Variance and cache reuse | `experiments/summarize_variance_cache.py` |
| Memory scaling | `experiments/prepare_scalability_slices.py`, `summarize_scalability.py` |
| Temporal-faithfulness audit | `run_chrono_temporal_judge.py`, `temporal_faithfulness_eval.py`, `analyze_temporal_faithfulness_audit.py` |
| Static WQ/PQ adaptation | `scripts/prepare_chronosynth_kgqg.py`, `run_full_chrono_kgqg.py` |

## Numerical provenance

`paper_results/paper_values.json` records every ChronoSynth number used in the final manuscript. The included CSV/JSON summaries preserve the exact unrounded values for RQ2-RQ5 without exposing private workspace paths.

Baseline code and raw baseline predictions are deliberately excluded. Baseline values remain in the paper and in the RQ1 comparison record only as reported comparison numbers.

## Important audit note

The saved temporal-faithfulness outputs were produced by a fixed LLM judging rubric plus deterministic answer-leakage matching. The final manuscript calls this a "rule-based audit script." The released code reflects the actual executed procedure, so readers can see the exact audit implementation rather than a renamed substitute.
