# ChronoSynth: completed supplementary experiment results

## Status and protocol

All API-based generation and evaluation jobs completed on 2026-08-12. New generations use `gpt-4o-mini`; the reference-free primary evaluator is `gpt-4.1-mini`; `gpt-4.1` independently replicates Full/No-memory on a fixed stratified 200-example subset per dataset. The evaluator receives only the generated question (TIS Recovery) or the local graph plus generated question (Answer Recovery); it never receives the gold answer, gold TIS, reference question, or method name. All reported metrics are deterministically scored from its saved JSON response.

CRONQUESTION's common Full/No-memory set contains 972 unique IDs because the supplied demo split has duplicate question IDs; MultiTQ contains 1,000. In the 500/300 CRONQUESTION component manifests, duplicate IDs reduce unique evaluated counts to 490/298; this is reported explicitly rather than filled with duplicated evaluations.

## A. Independent semantic evaluation

Percentages. TIS-EM requires all applicable TIS fields (operator, answer slot/type, granularity, and comparator) to match. Answer F1 compares the recovered graph answer set against the hidden gold answer set.

| Dataset | Method | N | Operator | Slot | Type | Granularity | Reference | TIS-EM | Answer EM | Answer F1 | Answerable | Wrong-slot | Unsupported |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRONQUESTION | Full | 972 | 22.2 | 86.7 | 99.8 | 46.0 | 87.3 | 1.5 | 90.1 | 90.1 | 97.1 | 0.0 | 3.0 |
| CRONQUESTION | No-memory | 972 | 31.3 | 80.7 | 100.0 | 74.0 | 88.6 | 9.5 | 58.3 | 60.3 | 94.2 | 0.1 | 5.8 |
| CRONQUESTION | Vanilla-RAG | 972 | 42.9 | 79.4 | 87.9 | 33.8 | 85.7 | 4.9 | 58.1 | 66.1 | 93.8 | 0.0 | 6.5 |
| CRONQUESTION | Relation-only | 972 | 21.7 | 87.7 | 99.8 | 44.9 | 86.8 | 1.1 | 86.5 | 86.5 | 94.7 | 0.0 | 5.5 |
| CRONQUESTION | −typed placeholders | 490 | 25.9 | 85.9 | 99.8 | 32.2 | 74.7 | 2.0 | 91.6 | 91.6 | 97.3 | 0.0 | 2.7 |
| CRONQUESTION | −temporal-rule prompt | 298 | 36.2 | 87.2 | 99.7 | 27.5 | 75.8 | 3.7 | 89.9 | 89.9 | 98.0 | 0.0 | 2.0 |
| CRONQUESTION | −leakage check | 298 | 34.2 | 85.2 | 99.0 | 26.5 | 76.2 | 2.7 | 89.3 | 89.3 | 97.3 | 0.0 | 2.7 |
| CRONQUESTION | −draft filtering | 298 | 33.6 | 85.9 | 99.0 | 26.2 | 76.5 | 3.0 | 90.3 | 90.3 | 97.3 | 0.0 | 2.7 |
| MULTITQ | Full | 1000 | 42.3 | 82.6 | 100.0 | 45.7 | 86.7 | 11.1 | 67.5 | 74.0 | 86.8 | 0.0 | 13.7 |
| MULTITQ | No-memory | 1000 | 65.7 | 84.6 | 100.0 | 58.1 | 75.6 | 20.0 | 59.6 | 66.7 | 87.4 | 0.0 | 13.0 |
| MULTITQ | Vanilla-RAG | 1000 | 13.5 | 83.7 | 90.8 | 56.9 | 66.1 | 5.2 | 59.5 | 66.4 | 95.4 | 0.0 | 4.8 |
| MULTITQ | Relation-only | 1000 | 43.9 | 86.6 | 100.0 | 43.9 | 89.8 | 13.8 | 60.0 | 66.1 | 83.0 | 0.0 | 18.1 |
| MULTITQ | −typed placeholders | 500 | 47.0 | 83.8 | 100.0 | 49.2 | 80.0 | 12.6 | 62.0 | 72.9 | 87.0 | 0.0 | 13.6 |
| MULTITQ | −temporal-rule prompt | 300 | 44.0 | 82.0 | 100.0 | 49.3 | 77.7 | 11.3 | 55.0 | 69.1 | 85.7 | 0.0 | 15.7 |
| MULTITQ | −leakage check | 300 | 46.7 | 82.0 | 100.0 | 50.7 | 79.0 | 12.0 | 56.0 | 69.5 | 86.0 | 0.0 | 14.7 |
| MULTITQ | −draft filtering | 300 | 46.0 | 81.3 | 100.0 | 49.7 | 78.3 | 13.7 | 56.0 | 69.0 | 84.3 | 0.0 | 16.3 |

## B. Paired Full vs No-memory comparison

Deltas are Full − No-memory in percentage points; 95% CIs use a 10,000-resample paired bootstrap. The two-sided sign test excludes ties.

| Dataset | Metric | N | Δ | 95% CI | Wins/Losses/Ties | Sign-test p |
|---|---:|---:|---:|---:|---:|---:|
| CRONQUESTION | tis_exact_match | 972 | -7.9 | [-10.0, -6.0] | 12/89/871 | 1.08e-15 |
| CRONQUESTION | answer_f1 | 972 | +29.8 | [+26.3, +33.3] | 357/56/559 | 9.85e-55 |
| CRONQUESTION | answer_exact_match | 972 | +31.8 | [+28.3, +35.3] | 357/48/567 | 1.78e-59 |
| MULTITQ | tis_exact_match | 1000 | -8.9 | [-11.7, -6.0] | 64/153/783 | 1.37e-09 |
| MULTITQ | answer_f1 | 1000 | +7.2 | [+3.9, +10.5] | 215/132/653 | 9.79e-06 |
| MULTITQ | answer_exact_match | 1000 | +7.9 | [+4.5, +11.2] | 184/105/711 | 3.92e-06 |

## C. Secondary evaluator replication

`gpt-4.1` is the independent secondary evaluator. Raw agreement refers to agreement between evaluator-scored binary outcomes on the common 200-example stratified subset, not agreement with a human label.

| Dataset | Task | Method | N | Mini score | gpt-4.1 score | Binary raw agreement |
|---|---:|---:|---:|---:|---:|---:|
| CRONQUESTION | tis | Full | 200 | 2.0 | 8.5 | 93.5 |
| CRONQUESTION | tis | No-memory | 200 | 19.5 | 33.5 | 85.0 |
| CRONQUESTION | answer | Full | 200 | 91.5 | 84.5 | 90.0 |
| CRONQUESTION | answer | No-memory | 200 | 64.0 | 66.0 | 95.0 |
| MULTITQ | tis | Full | 200 | 12.5 | 18.0 | 94.5 |
| MULTITQ | tis | No-memory | 200 | 28.0 | 38.0 | 86.0 |
| MULTITQ | answer | Full | 200 | 55.5 | 51.0 | 86.5 |
| MULTITQ | answer | No-memory | 200 | 42.5 | 45.5 | 95.0 |

## D. Scope and reproducibility notes

- Fair-resource comparison uses the same training-question pool, `gpt-4o-mini` generator, and top-k=4 for Vanilla-RAG. Vanilla-RAG intentionally excludes TIS construction, temporal compatibility filtering, typed placeholder adaptation, and validation.
- Validator results cover only switches actually implemented in the code: temporal-rule instruction, answer-leakage check, and draft filtering. The codebase does not expose separate `V_rule`, `V_field`, or `V_cue` modules; no unimplemented ablation is claimed.
- The historical generation supervisor reported a nonzero status for the earlier slow Vanilla implementation. The optimized resumable runner subsequently produced 1,000 valid rows for each dataset, and all associated independent evaluations completed.
- Detailed machine-readable files: `reports/semantic_summary.csv`, `reports/full_vs_no_memory_paired_stats.csv`, and `reports/gpt41mini_gpt41_agreement.csv`.
