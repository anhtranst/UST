# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Uncertainty-aware Self-Training (UST) for few-shot text classification of disaster-related tweets into 10 humanitarian categories. Uses BERT-based models (default: BERTweet) with MC Dropout for uncertainty estimation and pseudo-labeling of unlabeled data.

## Running the Pipeline

Requires `PYTHONHASHSEED` environment variable (used as global seed):

```bash
PYTHONHASHSEED=42 python run_ust.py \
  --disaster california_wildfires_2018 \
  --train_file S1T_5 \
  --sample_scheme easy_bald_class_conf \
  --sup_epochs 18 --unsup_epochs 12 \
  --T 7 --alpha 0.1 --N_base 3
```

Available datasets (under `data/`): california_wildfires_2018, canada_wildfires_2016, cyclone_idai_2019, hurricane_dorian_2019, hurricane_florence_2018, hurricane_harvey_2017, hurricane_irma_2017, hurricane_maria_2017, kaikoura_earthquake_2016, kerala_floods_2018. Each has few-shot labeled splits (5/10/25/50 per class), dev/test splits, and unlabeled data as TSV files.

No requirements.txt exists. Dependencies: PyTorch, Transformers (Hugging Face), scikit-learn, pandas, numpy, scipy, tqdm, torchmetrics.

## Architecture

- **run_ust.py** — CLI entry point. Parses arguments, loads TSV data into datasets, calls `train_model()`. Contains the 10-class `label_to_id` mapping. All data paths are relative to `data/{disaster}/`.
- **ust.py** — Core training logic. `BertModel` wraps `AutoModelForSequenceClassification` with a temperature scaling parameter. `train_model()` orchestrates: (1) fine-tune N_base random initializations with early stopping, pick best by dev F1; (2) run self-training iterations using MC Dropout uncertainty on unlabeled data; (3) evaluate on test set. GPU is hardcoded via `os.environ['CUDA_VISIBLE_DEVICES'] = '1'`.
- **sampler.py** — Uncertainty-based sampling for pseudo-label selection. Key schemes: `uniform`, `easy_bald`, `bald_difficulty`, `easy_bald_class_conf` (per-class BALD with confidence weighting). Uses Shannon entropy and BALD acquisition functions.
- **custom_dataset.py** — PyTorch Dataset subclasses. `CustomDataset` for basic text/label pairs; `CustomDataset_tracked` adds tweet ID tracking for the pseudo-labeling pipeline.

## Key Design Details

- Model checkpoints saved as `data/{disaster}/pytorch_model.bin` (state dict). Results written as JSON to `data/{disaster}/{results_file}.txt`.
- Self-training loop: samples `sample_size` unlabeled examples, runs T stochastic forward passes (MC Dropout), selects `unsup_size` pseudo-labeled instances per iteration.
- Confident learning loss reweighting controlled by `alpha` — pseudo-labeled samples weighted by model confidence.
