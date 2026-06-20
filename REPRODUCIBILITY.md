# Reproducibility

## Environment

- Python 3.10 or newer
- Linux recommended
- OpenAI-compatible text-generation endpoint
- `A_API_KEY` and `A_BASE_URL`

```bash
pip install -r requirements.txt
```

LLM outputs may vary with provider updates and decoding nondeterminism. `paper_results/` is the frozen aggregate record for exact manuscript traceability.

## RQ1: Full ChronoSynth Runs

```bash
PYTHONPATH=chronosynth python experiments/run_chrono_experiment.py \
  --train data/CRONQUESTION/train.json \
  --input data/CRONQUESTION/test.json \
  --output result/chronosynth_runs/chrono_full_CRONQUESTION_test.json \
  --model gpt-4o-mini --workers 30 \
  --memory_bank_cache result/chronosynth_runs/cache/cronquestion.pkl \
  --ablation full

PYTHONPATH=chronosynth python experiments/run_chrono_experiment.py \
  --train data/MULTITQ/train.json \
  --input data/MULTITQ/test.json \
  --output result/chronosynth_runs/chrono_full_MULTITQ_test.json \
  --model gpt-4o-mini --workers 30 \
  --memory_bank_cache result/chronosynth_runs/cache/multitq.pkl \
  --ablation full
```

Repeat with the paper's Qwen3.5-9B and DeepSeek-V4-Flash endpoints to reproduce the backend comparison.

## RQ2: Ablations

Use the same commands with:

```text
--ablation no_memory
--ablation relation_only
```

The paper's component panel uses a fixed 100-instance subset and reports full, no-memory, and relation-only.

## RQ3: Grouped Robustness

The runner writes `.per_sample.jsonl` records containing operator, answer type, time level, edge count, graph shape, latency, and estimated tokens.

```bash
python experiments/analyze_grouped_results.py \
  --result-dir result/chronosynth_runs \
  --split test \
  --min-group-size 100 \
  --out-dir result/summaries

python experiments/grouped_significance.py \
  --result-dir result/chronosynth_runs \
  --reps 300 \
  --sample-size 1000 \
  --seed 42 \
  --out-dir result/summaries
```

## Temporal-faithfulness Audit

`run_chrono_temporal_judge.py` applies the fixed audit rubric to aligned sampled question IDs. Answer leakage is checked deterministically; operator, slot, comparator, and overall faithfulness are judged by the configured LLM.

```bash
PYTHONPATH=chronosynth python run_chrono_temporal_judge.py \
  --input_data data/CRONQUESTION/test.json \
  --results result/chronosynth_runs/chrono_full_CRONQUESTION_test.json \
  --output result/audit/cronquestion_full_judge.json \
  --judge_model gpt-4o-mini \
  --sample_size 200 \
  --seed 42 \
  --workers 12
```

Run the same command for every compared result file, then use:

```bash
python experiments/analyze_temporal_faithfulness_audit.py \
  --judge_dir result/audit \
  --output_dir result/audit
```

## RQ4: Variance and Memory Reuse

Run the full and no-memory methods three times on the same 100-instance MultiTQ subset. For cache reuse, run full once after deleting the target cache, then run the identical command again with the same cache path.

Summarize:

```bash
python experiments/summarize_variance_cache.py
```

## RQ5: Memory Scale

Prepare deterministic 20/40/80 percent slices with `prepare_scalability_slices.py`. Run the full method on each slice and the complete train split, keeping the same 100-instance MultiTQ test subset, backend, and worker setting.

```bash
python experiments/summarize_scalability.py
```

## Static KGQG Sanity Check

```bash
python run_full_chrono_kgqg.py \
  --datasets WQ PQ \
  --model gpt-4o-mini \
  --workers 30
```

The paper uses WQ `transfer_v1` and PQ `sota_v3` aggregate metrics listed in `paper_results/paper_values.json`.
