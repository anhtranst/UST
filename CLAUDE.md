# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository accompanies the paper "Calibrated Semi-Supervised Models for Disaster Response based on Training Dynamics" (Gupta, Gautam, Sosea, Caragea & Caragea, ISCRAM 2025). It implements and evaluates semi-supervised learning (SSL) methods for few-shot text classification of disaster-related tweets into humanitarian categories.

The dataset is a subset of HumAID (10 of 19 disaster events, 2016–2019). Not all disasters have 10 classes — some have 7, 8, or 9. The code dynamically detects the actual classes per disaster at runtime via `detect_classes()`.

The key research focus is on **model calibration** (ECE) alongside classification performance (macro-F1).

## How Self-Training Works

No external teacher model (e.g., GPT-4o) is used. The same BERTweet model acts as both teacher and student:

1. **Phase 1 — Base model selection:** Fine-tune BERTweet on the small labeled set `N_base` times (default: 3) with different random initializations. Keep the best by validation F1.
2. **Phase 2 — Self-training loop (iterates `unsup_epochs` times, default: 12):**
   - The current best model predicts pseudo-labels on a sample of unlabeled data
   - Select pseudo-labeled instances (uniform: random; BALD schemes: uncertainty-based)
   - Re-train on labeled + pseudo-labeled data (50/50 weighted loss)
   - If validation F1 improves, save as new best checkpoint
   - Repeat — each round the model improves, producing better pseudo-labels
3. **Phase 3 — Evaluate** final checkpoint on test set (macro-F1 + ECE).

The unlabeled TSV files do contain a `class_label` column (ground truth), but it is never used during training — data is always loaded with `labeled=False`. Pseudo-labels are generated at runtime by the model itself.

## Running the Pipeline

Requires `PYTHONHASHSEED` environment variable (used as global seed):

```bash
PYTHONHASHSEED=42 python run_ust.py \
  --disaster california_wildfires_2018 \
  --train_file 5_set1 \
  --sample_scheme uniform \
  --sup_epochs 18 --unsup_epochs 12 \
  --T 7 --alpha 0.1 --N_base 3
```

For batch experiments across all 10 disasters × 12 splits (120 total), use `notebooks/st-experiments.ipynb`. It has resume support (skips completed experiments) and entry-weighted progress tracking.

## Dataset

Source: HumAID (Alam et al., ICWSM 2021). 10 disaster events under `data/`.

Each disaster has:
- `labeled_{k}_set{s}.tsv` — few-shot labeled splits (k=5/10/25/50 per class, s=1/2/3)
- `unlabeled_{k}_set{s}.tsv` — corresponding unlabeled pools (remainder of training data)
- `{disaster}_dev.tsv` / `{disaster}_test.tsv` — fixed validation/test splits

Class counts per disaster: california_wildfires_2018 (10), cyclone_idai_2019 (10), canada_wildfires_2016 (8), all others (9). The missing classes are typically `missing_or_found_people` and/or `injured_or_dead_people`.

## Architecture

- **run_ust.py** — CLI entry point. Parses arguments, detects actual classes via `detect_classes()`, loads TSV data, calls `train_model()`.
- **ust.py** — Core training logic. `BertModel` wraps `AutoModelForSequenceClassification` with a temperature scaling parameter. `train_model()` orchestrates all three phases. `mc_dropout_evaluate()` runs T stochastic forward passes for BALD uncertainty. `evaluate()` computes macro-F1 and ECE (n_bins=10). GPU device set via `CUDA_VISIBLE_DEVICES` env var (defaults to '0' if not set).
- **sampler.py** — Uncertainty-based sampling for pseudo-label selection. Schemes: `uniform` (random, no uncertainty), `easy_bald` (prefer low-BALD), `easy_bald_class_conf` (per-class BALD with confidence weighting), `bald_difficulty` (prefer high-BALD). Uses BALD acquisition function.
- **custom_dataset.py** — PyTorch Dataset subclasses. `CustomDataset` for basic text/label pairs; `CustomDataset_tracked` adds tweet ID tracking. Both handle `token_type_ids` fallback for RoBERTa-based tokenizers.
- **notebooks/st-experiments.ipynb** — Batch experiment runner for ST (uniform scheme) across all disasters × all 12 splits. Includes sanity check, pre-scan, resume-aware experiment loop, and summary pivot tables.

## Key Design Details

- Model checkpoints saved as `data/{disaster}/pytorch_model.bin` (state dict, overwritten each run).
- Results written as JSON to `results/{disaster}/{results_file}.txt` (separate from data/).
- Early stopping patience = 3 (hardcoded, the `patience = 5` variable on line 173 is dead code).
- Self-training mixed loss: `0.5 * supervised + 0.5 * mean(confidence_weighted_unsupervised)`.
- Confidence learning loss reweighting: `weight = -log(confidence) * alpha`.
- Dependencies listed in `requirements.txt`. Note: `torchmetrics` pinned to `>=1.0,<1.3` to avoid torchvision circular import issues. `emoji==0.6.0` required by BERTweet tokenizer.
- Environment variables in `.env` (git-ignored): `PYTHONHASHSEED`, `CUDA_VISIBLE_DEVICES`, `HF_HUB_DISABLE_SYMLINKS_WARNING`.
