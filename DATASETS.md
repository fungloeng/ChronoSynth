# Dataset Preparation

No dataset records are distributed in this repository.

## Temporal Benchmarks

Acquire the official releases of:

- CRONQUESTIONS, cited in the paper as Saxena et al., 2021.
- MultiTQ, cited in the paper as Chen et al., 2023.

Convert them to the JSON structure expected by `chronosynth/chronoagent_harness/data.py`, then place them at:

```text
data/
  CRONQUESTION/
    train.json
    valid.json
    test.json
  MULTITQ/
    train.json
    valid.json
    test.json
```

Expected split sizes:

| Dataset | Train | Dev | Test |
|---|---:|---:|---:|
| CRONQUESTIONS | 212,259 | 19,014 | 19,122 |
| MultiTQ | 257,687 | 39,128 | 36,318 |

Each item must provide:

```json
{
  "question_id": 1,
  "question": "reference question",
  "paraphrases": ["reference question"],
  "answer_text": ["answer"],
  "subgraph": {
    "edges": [
      {
        "source": "entity",
        "relation": "relation",
        "target": "entity",
        "time": "timestamp"
      }
    ]
  }
}
```

CRONQUESTIONS may use `start_year` and `end_year`; MultiTQ may use `source_name`, `target_name`, and day/month timestamps. The loader supports both forms.

## Scalability Slices

Create the deterministic MultiTQ memory slices used in RQ5:

```bash
python experiments/prepare_scalability_slices.py \
  --train data/MULTITQ/train.json \
  --out_dir data/MULTITQ \
  --seed 42 \
  --scales 20 40 80
```

The 100 percent condition uses `data/MULTITQ/train.json`.

## Static KGQG Sanity Check

The paper's WQ/PQ sanity check uses locally prepared graph-to-question sources. After obtaining the corresponding source files, run:

```bash
python scripts/prepare_chronosynth_kgqg.py --datasets WQ PQ
```

This writes the common ChronoSynth format under `data/chronosynth_kgqg/`. Source datasets are not redistributed here because the public artifact is intentionally code-only.
